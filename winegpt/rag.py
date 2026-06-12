"""enotropos — RAG module.

Retrieval-Augmented Generation chain:
1. User query → embedding
2. Retrieve top-k relevant chunks from ChromaDB
3. Build prompt with context + query
4. Generate answer via DeepSeek V4 Flash (OpenCode Go)
5. Return answer with source citations
"""
import logging
from typing import Any

from openai import OpenAI

from winegpt.config import (
    LLM_BASE_URL,
    LLM_MODEL,
    OPENCODE_GO_API_KEY,
    SYSTEM_PROMPT,
    TOP_K_CHUNKS,
)
from winegpt.embed import embed_texts

logger = logging.getLogger(__name__)


def get_llm_client() -> OpenAI:
    if not OPENCODE_GO_API_KEY:
        raise ValueError("OPENCODE_GO_API_KEY not set in .env")
    return OpenAI(base_url=LLM_BASE_URL, api_key=OPENCODE_GO_API_KEY)


def build_prompt(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """Build a prompt with retrieved context chunks."""
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        src = f"[{i}] {meta.get('gi_name', '?')} ({meta.get('gi_type', '?')}) — {meta.get('section', '?')} — {meta.get('source_file', '?')}"
        context_parts.append(f"{src}\n{chunk['document']}")

    context_text = "\n\n---\n\n".join(context_parts)

    return (
        "Basat exclusivament en els documents seguents, respon la pregunta. "
        "Cita les fonts entre claudators (ex: [1]). "
        "Si la informacio no es troba als documents, digues-ho.\n\n"
        f"## Documents de referencia\n\n{context_text}\n\n"
        f"## Pregunta\n\n{query}\n\n"
        "## Resposta"
    )


def generate(
    query: str,
    context_chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Generate an answer using the LLM with retrieved context.

    Returns (answer, citations).
    """
    client = get_llm_client()
    prompt = build_prompt(query, context_chunks)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
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
    """Full RAG pipeline: embed → retrieve → generate.

    Returns dict with answer, citations, and context_chunks.
    """
    from winegpt.store import query as store_query

    # 1. Embed the query
    embeddings = embed_texts([query])
    if not embeddings:
        return {"answer": "Error generating query embedding.", "citations": [], "context_chunks": []}
    query_embedding = embeddings[0]

    # 2. Retrieve relevant chunks
    context_chunks = store_query(
        query_embedding,
        k=top_k,
        country=country,
        gi_type=gi_type,
    )

    if not context_chunks:
        return {
            "answer": "No relevant documents found for your query.",
            "citations": [],
            "context_chunks": [],
        }

    # 3. Generate answer
    answer, citations = generate(query, context_chunks)

    return {
        "answer": answer,
        "citations": citations,
        "context_chunks": context_chunks,
    }
