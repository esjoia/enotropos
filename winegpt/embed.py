"""enotropos — Embedding module.

Generates vector embeddings for document chunks using
Jina AI embeddings (OpenAI-compatible API).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI

from winegpt.config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    JINA_API_KEY,
)

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY not set in .env")
    return OpenAI(base_url=EMBEDDING_BASE_URL, api_key=JINA_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Generate embeddings for a list of texts in batches.

    Returns list of embedding vectors, or None on error.
    """
    client = get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error("Failed to embed batch %d: %s", i, e)
                    return None
                wait = 2 ** attempt
                logger.warning("Embedding error, retrying in %ds: %s", wait, e)
                time.sleep(wait)
        if i + EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(4.5)

    return all_embeddings


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Embed a list of chunk records. Adds 'embedding' field to each.

    Returns the enriched chunk list, or empty list on error.
    """
    if not chunks:
        return []

    texts = [c["markdown"] for c in chunks]
    logger.info("Embedding %d chunks (model: %s)...", len(texts), EMBEDDING_MODEL)

    embeddings = embed_texts(texts)
    if embeddings is None:
        return []

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector

    logger.info("Embedded %d chunks successfully", len(chunks))
    return chunks
