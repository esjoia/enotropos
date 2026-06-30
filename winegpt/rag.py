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

import json
import logging
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

from winegpt.config import (
    LLM_MODEL,
    SYSTEM_PROMPT,
    TOP_K_CHUNKS,
)
from winegpt.embed import embed_texts
from winegpt.llm import get_llm_client
from winegpt.schema import StreamResult

logger = logging.getLogger(__name__)

_KEYWORDS_PATH = Path(__file__).parent / "data" / "keywords.json"
_keywords_cache: dict[str, Any] | None = None


def _load_keywords() -> dict[str, Any]:
    """Load and cache the keyword tables (CA→ES map + stop words)."""
    global _keywords_cache
    if _keywords_cache is None:
        _keywords_cache = json.loads(_KEYWORDS_PATH.read_text(encoding="utf-8"))
    return _keywords_cache


_gi_names_cache: dict[str, str] | None = None


def _load_corpus_gi_names() -> dict[str, str]:
    """Load and cache known GI names from the extracted corpus.

    Returns ``{normalized_display_name: underscore_name}`` so we can detect
    queries like *"varietats de Rioja"* (no DOP/IGP prefix).
    """
    global _gi_names_cache
    if _gi_names_cache is not None:
        return _gi_names_cache
    _gi_names_cache = {}
    from winegpt.config import EXTRACTED_DIR
    from winegpt.schema import parse_folder_name
    for country_dir in sorted(EXTRACTED_DIR.iterdir()):
        if not country_dir.is_dir():
            continue
        for gi_dir in country_dir.iterdir():
            if not gi_dir.is_dir():
                continue
            info = parse_folder_name(gi_dir.name)
            if not info.is_gi:
                continue
            norm_lower = info.display_name.lower().replace(" ", "_")
            _gi_names_cache[norm_lower] = info.gi_name
            _gi_names_cache[info.gi_name.lower()] = info.gi_name
    return _gi_names_cache


def _extract_gi_names(query: str) -> list[str]:
    """Extract GI names from a query.

    Example: 'varietats DOP Rioja i DOP Penedès?' → ['Rioja', 'Penedès'].
    Also detects names without explicit DOP/IGP prefix via corpus lookup
    (e.g. 'varietats de Rioja' → ['Rioja']).
    """
    matches = re.finditer(
        r"(?:DOP|dop|IGP|igp)\s+(?:de\s+)?(?:la\s+)?(?:el\s+)?([A-ZÀ-Ü][\w\s\-']+?)(?=\s*(?:[.,?\"']|\s+i\s+|\s+y\s+|\s+o\s+|\s+a\s+|\s+amb\s+|$))",
        query,
    )
    names = []
    for match in matches:
        name = match.group(1).strip()
        parts = name.split()
        result: list[str] = []
        for i, part in enumerate(parts):
            is_upper = part[0].isupper() if part else False
            is_connector = part.lower() in {
                "de", "del", "dels", "los", "las", "la", "el", "y", "i", "d", "d'",
            }
            if i == 0 and not is_upper:
                break
            if is_upper or is_connector:
                result.append(part)
            else:
                break
        if result:
            final_name = " ".join(result).replace(" ", "_")
            if final_name not in names:
                names.append(final_name)

    # Detect GI names without DOP/IGP prefix by matching against the corpus
    corpus_names = _load_corpus_gi_names()
    query_norm = query.lower().replace("-", " ").replace("_", " ")
    for norm_key, underscore_name in corpus_names.items():
        search_name = norm_key.replace("_", " ").lower()
        if len(search_name) >= 4 and search_name in query_norm:
            if underscore_name not in names:
                names.append(underscore_name)
            continue
        # For multi-name GIs (e.g. "Priorat__Priorato"), also match each
        # component individually so "Priorat" alone is detected.
        if "__" in underscore_name:
            for part in underscore_name.split("__"):
                part_norm = part.lower().replace("_", " ").strip()
                if len(part_norm) >= 4 and part_norm in query_norm:
                    if underscore_name not in names:
                        names.append(underscore_name)
                    break

    return names


def _rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = TOP_K_CHUNKS,
    gi_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid reranking: combine embedding similarity with keyword matching.

    For each chunk, compute a keyword score based on how many query words
    appear in the chunk text. Combine with embedding distance for final rank.
    """
    if len(chunks) <= top_k:
        return chunks

    # Extract meaningful keywords from query (words > 3 chars, not stop words)
    kw_data = _load_keywords()
    stop_words = set(kw_data["stop_words"])
    ca_es_map: dict[str, str] = kw_data["ca_es_map"]
    query_words = [w.lower() for w in re.findall(r"[a-zA-Zà-üÀ-Ü]{4,}", query)]
    keywords = [w for w in query_words if w not in stop_words]

    # Exclude GI names from keywords so they don't monopolize keyword score
    gi_lowers = [g.lower().replace("_", " ") for g in (gi_names or [])]
    filtered_keywords = []
    for kw in keywords:
        if not any(kw in g for g in gi_lowers) and not any(g in kw for g in gi_lowers):
            filtered_keywords.append(kw)
    keywords = filtered_keywords

    # Expand keywords: add Spanish equivalents for Catalan words
    expanded_keywords = set(keywords)
    for kw in keywords:
        es = ca_es_map.get(kw)
        if es:
            expanded_keywords.add(es)
        # Also add partial matches
        for ca, es in ca_es_map.items():
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

        # GI match bonus
        stored_name = chunk.get("metadata", {}).get("gi_name", "").lower().replace("_", " ")
        gi_match_bonus = 0.0
        for gi_lower in gi_lowers:
            if gi_lower in stored_name or stored_name in gi_lower:
                gi_match_bonus = 0.4
                break

        # Combined: 30% embedding + 50% keywords + 20% bonus
        combined = 0.3 * emb_score + 0.5 * kw_score + gi_match_bonus
        scored.append((combined, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def build_context(
    query: str,
    context_chunks: list[dict[str, Any]],
    gi_names: list[str] | None = None,
    conversation_history: str = "",
) -> tuple[str, str]:
    """Build context text and full prompt from retrieved chunks.

    Returns (context_text, full_prompt).
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        src = (
            f"[{i}] {meta.get('gi_name', '?')} ({meta.get('gi_type', '?')}) — "
            f"{meta.get('section', '?')} — {meta.get('source_file', '?')}"
        )
        context_parts.append(f"{src}\n{chunk['document']}")

    context_text = "\n\n---\n\n".join(context_parts)

    extra_instructions = ""
    if gi_names and len(gi_names) > 1:
        extra_instructions = (
            "\nS'ha detectat que la consulta involucra múltiples regions geogràfiques. "
            "Si es demana una comparació o llistat combinat, estructura la teva resposta "
            "de forma clara, preferiblement utilitzant taules o llistes estructurades per "
            "cada regió, destacant-ne les similituds i diferències principals."
        )

    history_block = (
        f"{conversation_history}\n\n"
        if conversation_history
        else ""
    )

    prompt = (
        "Basat exclusivament en els documents seguents, respon la pregunta. "
        "Cita les fonts entre claudators (ex: [1]). "
        "Si la informacio no es troba als documents, digues-ho."
        f"{extra_instructions}\n\n"
        f"{history_block}"
        f"## Documents de referencia\n\n{context_text}\n\n"
        f"## Pregunta\n\n{query}\n\n"
        "## Resposta"
    )
    return context_text, prompt


def generate(
    query: str,
    context_chunks: list[dict[str, Any]],
    gi_names: list[str] | None = None,
    conversation_history: str = "",
) -> tuple[str, list[dict[str, str]]]:
    """Generate an answer using the LLM with retrieved context.

    Uses DeepSeek V4 Flash via OpenCode Go.

    Returns (answer, citations).
    """
    context_text, _prompt = build_context(
        query, context_chunks, gi_names, conversation_history,
    )

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

    citations = _build_citations(context_chunks)
    return answer, citations


def generate_stream(
    query: str,
    context_chunks: list[dict[str, Any]],
    gi_names: list[str] | None = None,
    conversation_history: str = "",
    result: StreamResult | None = None,
) -> Generator[str, None, None]:
    """Generate a streaming answer using the LLM.

    Yields token chunks as they arrive. If a ``result`` holder is provided, it
    is populated with the final ``answer`` and ``citations`` once the stream is
    exhausted (preferred over the legacy ``generator.value`` pattern).

    Uses DeepSeek V4 Flash via OpenCode Go with SSE streaming.
    """
    _context_text, prompt = build_context(
        query, context_chunks, gi_names, conversation_history,
    )

    client = get_llm_client()
    full_answer: list[str] = []

    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=16000,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_answer.append(delta.content)
                yield delta.content

    except Exception as e:
        logger.error("LLM streaming error: %s", e)
        msg = f"\n\nError generating answer: {e}"
        full_answer.append(msg)
        yield msg

    if result is not None:
        result.answer = "".join(full_answer)
        result.citations = _build_citations(context_chunks)


def _build_citations(
    context_chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build citation list from context chunks."""
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
    return citations


def query_rag(
    query: str,
    country: str | None = None,
    gi_type: str | None = None,
    top_k: int = TOP_K_CHUNKS,
    conversation_history: str = "",
) -> dict[str, Any]:
    """Full RAG pipeline: extract GI → embed → retrieve → generate.

    Returns a dict with a consistent shape::

        {"answer", "citations", "context_chunks", "gi_names", "ok"}

    On failure ``ok`` is ``False`` and ``answer`` carries the error message.
    """
    context_chunks, gi_names, error = _retrieve(query, country, gi_type, top_k)
    if error:
        error["ok"] = False
        error.setdefault("gi_names", [])
        return error

    answer, citations = generate(
        query, context_chunks, gi_names, conversation_history,
    )

    return {
        "answer": answer,
        "citations": citations,
        "context_chunks": context_chunks,
        "gi_names": gi_names,
        "ok": True,
    }


def query_rag_stream(
    query: str,
    country: str | None = None,
    gi_type: str | None = None,
    top_k: int = TOP_K_CHUNKS,
    conversation_history: str = "",
) -> tuple[Generator[str, None, None], StreamResult]:
    """Streaming version of query_rag.

    Returns ``(generator, result)``. Iterate the generator to render tokens;
    once exhausted, ``result.answer`` / ``result.citations`` are populated. If
    retrieval fails, the generator yields the error message and ``result.answer``
    is set to it.
    """
    result = StreamResult()

    context_chunks, gi_names, error = _retrieve(query, country, gi_type, top_k)
    if error:
        msg = error["answer"]
        result.answer = msg

        def _err_gen() -> Generator[str, None, None]:
            yield msg
        return _err_gen(), result

    def _gen() -> Generator[str, None, None]:
        yield from generate_stream(
            query, context_chunks, gi_names, conversation_history, result,
        )
    return _gen(), result


def _retrieve(
    query: str,
    country: str | None,
    gi_type: str | None,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    """Shared retrieval logic. Returns (chunks, gi_names, error_dict_or_None)."""
    from winegpt.store import query as store_query

    gi_names = _extract_gi_names(query)
    if gi_names:
        logger.info("Detected GI names: %s", gi_names)

    embeddings, embed_err = embed_texts([query], input_type="query")
    if embed_err is not None or not embeddings:
        return [], [], {
            "answer": f"Error generating query embedding: {embed_err or 'empty response'}",
            "citations": [],
            "context_chunks": [],
        }

    multiplier = max(1, len(gi_names))
    retrieve_k = top_k * 8 * multiplier
    final_k = top_k * multiplier

    context_chunks = store_query(
        query_embedding=embeddings[0],
        k=retrieve_k,
        country=country,
        gi_type=gi_type,
        gi_names=gi_names or None,
    )

    if not context_chunks:
        return [], [], {
            "answer": "No relevant documents found for your query.",
            "citations": [],
            "context_chunks": [],
        }

    context_chunks = _rerank_chunks(query, context_chunks, final_k, gi_names)
    return context_chunks, gi_names, None
