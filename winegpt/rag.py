"""enotropos — RAG module.

Retrieval-Augmented Generation chain:
1. User query → extract GI name for metadata filtering
2. Embed query → retrieve 20 candidate chunks from ChromaDB
3. Fuzzy GI name matching + hybrid reranking (embedding + keyword)
4. Build prompt with top-k context + query
5. Generate answer via DeepSeek V4 Flash (OpenCode Go)
6. Return answer with source citations
"""
from __future__ import annotations

import logging
import re
from typing import Any

from winegpt.config import (
    LLM_BASE_URL,
    LLM_MODEL,
    OPENCODE_GO_API_KEY,
    SYSTEM_PROMPT,
    TOP_K_CHUNKS,
)
from winegpt.embed import embed_texts

logger = logging.getLogger(__name__)


def _extract_gi_name(query: str) -> str | None:
    """Extract GI name from a query like 'varietats de la DOP Rioja?' → 'Rioja'."""
    match = re.search(
        r"(?:DOP|dop|IGP|igp)\s+(?:de\s+)?(?:la\s+)?(?:el\s+)?([A-ZÀ-Ü][\w\s\-']+?)(?:\s*\?|$|\.|,|\")",
        query,
    )
    if not match:
        return None
    name = match.group(1).strip()
    # Keep only words starting with uppercase (proper nouns) and known connectors
    parts = name.split()
    result: list[str] = []
    for i, part in enumerate(parts):
        # Keep if starts with uppercase OR is a known connector (de, del, los, etc.)
        is_upper = part[0].isupper() if part else False
        is_connector = part.lower() in {"de", "del", "dels", "los", "las", "la", "el", "y", "i", "d"}
        if i == 0 and not is_upper:
            return None  # First word must be proper noun
        if is_upper or is_connector:
            result.append(part)
        else:
            break  # Stop at first non-proper, non-connector lowercase word
    name = " ".join(result) if result else None
    # Normalize: replace spaces with underscores to match ChromaDB metadata format
    return name.replace(" ", "_") if name else None


def _rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = TOP_K_CHUNKS,
) -> list[dict[str, Any]]:
    """Hybrid reranking: combine embedding similarity with keyword matching.

    For each chunk, compute a keyword score based on how many query words
    appear in the chunk text. Combine with embedding distance for final rank.
    """
    if len(chunks) <= top_k:
        return chunks

    # Extract meaningful keywords from query (words > 3 chars, not stop words)
    _stop_words = {
        "que", "quines", "quins", "quin", "quina", "quina",
        "son", "esta", "estan", "amb", "per", "de", "la", "el", "els",
        "les", "a", "en", "d", "i", "o", "es", "una", "un", "uns",
        "unes", "del", "dels", "com", "que", "qui", "se", "ens",
        "what", "is", "the", "are", "of", "in", "to", "for", "and",
    }
    query_words = [w.lower() for w in re.findall(r"[a-zA-Zà-üÀ-Ü]{4,}", query)]
    keywords = [w for w in query_words if w not in _stop_words]

    # Expand keywords: add Spanish equivalents for Catalan words
    _ca_es_map = {
        "varietats": "variedades",
        "varietat": "variedad",
        "raim": "uva",
        "raïm": "uva",
        "blanc": "blanco",
        "blanca": "blanca",
        "negre": "tinto",
        "negres": "tintas",
        "envelliment": "envejecimiento",
        "criança": "crianza",
        "crianca": "crianza",
        "rendiment": "rendimiento",
        "geografics": "geograficos",
        "geogràfics": "geograficos",
        "limits": "limites",
        "límit": "limite",
        "pràctiques": "practicas",
        "pratiques": "practicas",
        "enologiques": "enologicas",
        "metodes": "metodos",
        "mètodes": "metodos",
        "periode": "periodo",
        "període": "periodo",
        "minim": "minimo",
        "mínim": "minimo",
        "maxim": "maximo",
        "màxim": "maximo",
        "produccio": "produccion",
        "producció": "produccion",
        "subzones": "subzonas",
        "qualificacio": "calificacion",
        "qualificació": "calificacion",
        "criteris": "criterios",
        "foranies": "foraneas",
        "forànies": "foraneas",
        "autoritza": "autoriza",
        "autoritzades": "autorizadas",
        "autoritzats": "autorizados",
        "permes": "permitido",
        "permès": "permitido",
        "permesos": "permitidos",
        "poda": "poda",
        "densitat": "densidad",
        "plantacio": "plantacion",
        "plantació": "plantacion",
        "extraccio": "extraccion",
        "extracció": "extraccion",
        "contingut": "contenido",
        "graduacio": "graduacion",
        "graduació": "graduacion",
        "alcoholica": "alcoholica",
        "alcohòlica": "alcoholica",
        "tipus": "tipos",
        "produeix": "produce",
        "vins": "vinos",
        "vi": "vino",
    }
    expanded_keywords = set(keywords)
    for kw in keywords:
        es = _ca_es_map.get(kw)
        if es:
            expanded_keywords.add(es)
        # Also add partial matches
        for ca, es in _ca_es_map.items():
            if ca.startswith(kw) or kw.startswith(ca):
                expanded_keywords.add(ca)
                expanded_keywords.add(es)

    # Score each chunk
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        doc_lower = chunk.get("document", "").lower()
        section_lower = chunk.get("metadata", {}).get("section", "").lower()

        # Keyword match score: count occurrences of each keyword (weighted)
        text = doc_lower + " " + section_lower
        kw_hits = 0
        for kw in expanded_keywords:
            count = text.count(kw)
            kw_hits += count
        kw_score = min(kw_hits / max(len(expanded_keywords), 1), 1.0)

        # Embedding score: invert distance
        distance = chunk.get("distance", 1.0)
        emb_score = max(0.0, 1.0 - distance)

        # Combined: 50% embedding + 50% keywords
        combined = 0.5 * emb_score + 0.5 * kw_score
        scored.append((combined, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def get_llm_client() -> OpenAI:
    from openai import OpenAI

    if not OPENCODE_GO_API_KEY:
        raise ValueError("OPENCODE_GO_API_KEY not set in .env")
    return OpenAI(base_url=LLM_BASE_URL, api_key=OPENCODE_GO_API_KEY)


def build_context(query: str, context_chunks: list[dict[str, Any]]) -> tuple[str, str]:
    """Build context text and full prompt from retrieved chunks.

    Returns (context_text, full_prompt).
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        src = f"[{i}] {meta.get('gi_name', '?')} ({meta.get('gi_type', '?')}) — {meta.get('section', '?')} — {meta.get('source_file', '?')}"
        context_parts.append(f"{src}\n{chunk['document']}")

    context_text = "\n\n---\n\n".join(context_parts)

    prompt = (
        "Basat exclusivament en els documents seguents, respon la pregunta. "
        "Cita les fonts entre claudators (ex: [1]). "
        "Si la informacio no es troba als documents, digues-ho.\n\n"
        f"## Documents de referencia\n\n{context_text}\n\n"
        f"## Pregunta\n\n{query}\n\n"
        "## Resposta"
    )
    return context_text, prompt


def generate(
    query: str,
    context_chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Generate an answer using the LLM with retrieved context.

    Uses DeepSeek V4 Flash via OpenCode Go.

    Returns (answer, citations).
    """
    context_text, _prompt = build_context(query, context_chunks)

    from openai import OpenAI

    client = get_llm_client()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _prompt},
            ],
            temperature=0.3,
            max_tokens=16000,
        )
        answer = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("LLM error: %s", e)
        return f"Error generating answer: {e}", []

    citations: list[dict[str, str]] = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        citations.append({
            "ref": str(i),
            "gi_name": meta.get("gi_name", ""),
            "gi_type": meta.get("gi_type", ""),
            "section": meta.get("section", ""),
            "source_file": meta.get("source_file", ""),
            "country": meta.get("country", ""),
        })

    return answer, citations


def query_rag(
    query: str,
    country: str | None = None,
    gi_type: str | None = None,
    top_k: int = TOP_K_CHUNKS,
) -> dict[str, Any]:
    """Full RAG pipeline: extract GI → translate → embed → retrieve → generate.

    Returns dict with answer, citations, and context_chunks.
    """
    from winegpt.store import query as store_query

    # 0. Extract GI name for metadata filtering
    gi_name = _extract_gi_name(query)
    if gi_name:
        logger.info("Detected GI name: %s", gi_name)

    # 1. Embed the query and retrieve
    embeddings = embed_texts([query])
    if not embeddings:
        return {"answer": "Error generating query embedding.", "citations": [], "context_chunks": []}
    query_embedding = embeddings[0]

    # 2. Retrieve relevant chunks (retrieve 4x for reranking)
    context_chunks = store_query(
        query_embedding,
        k=top_k * 4,
        country=country,
        gi_type=gi_type,
    )

    # 2b. Fuzzy GI name filtering: boost or filter chunks matching the query GI
    if gi_name and context_chunks:
        gi_lower = gi_name.lower().replace("_", " ")
        matched = []
        unmatched = []
        for chunk in context_chunks:
            stored_name = chunk.get("metadata", {}).get("gi_name", "").lower().replace("_", " ")
            # Check if extracted name is substring of stored name or vice versa
            if gi_lower in stored_name or stored_name in gi_lower:
                matched.append(chunk)
            else:
                unmatched.append(chunk)
        if matched:
            # Prefer matched chunks, but keep unmatched as fallback
            context_chunks = matched + unmatched
        else:
            logger.info("No fuzzy GI name match for '%s', using all results", gi_name)

    if not context_chunks:
        return {
            "answer": "No relevant documents found for your query.",
            "citations": [],
            "context_chunks": [],
        }

    # 2b. Rerank chunks with LLM to select the most relevant
    context_chunks = _rerank_chunks(query, context_chunks, top_k)

    # 3. Generate answer
    answer, citations = generate(query, context_chunks)

    return {
        "answer": answer,
        "citations": citations,
        "context_chunks": context_chunks,
    }
