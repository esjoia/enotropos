"""enotropos — Embedding module.

Generates vector embeddings for document chunks. Supports two providers:

* ``"nvidia"`` — NVIDIA NIM (OpenAI-compatible API, default). Requires
  ``NVIDIA_API_KEY`` in ``.env``.
* ``"local"`` — ``fastembed`` ONNX model (default: ``intfloat/multilingual-e5-large``,
  1024 dimensions). No API key needed; the ONNX model is downloaded on first use
  (~500 MB).

Set ``EMBEDDING_PROVIDER=local`` in ``.env`` to switch. When switching
providers, reset ChromaDB collections with ``--reset`` (dimensions differ:
NVIDIA=2048, local=1024).

If provider is ``"nvidia"`` and the NVIDIA API call fails, the module
automatically falls back to the local model (with a warning).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from winegpt.config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_BATCH_SLEEP,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    NVIDIA_API_KEY,
)

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local model (fastembed) — lazy + thread-safe
# ---------------------------------------------------------------------------

_local_model: Any = None
_local_model_lock = threading.Lock()


def _get_local_model() -> Any:
    """Return the cached fastembed TextEmbedding model."""
    global _local_model
    if _local_model is None:
        with _local_model_lock:
            if _local_model is None:
                from fastembed import TextEmbedding

                logger.info("Loading local embedding model: %s", LOCAL_EMBEDDING_MODEL)
                _local_model = TextEmbedding(model_name=LOCAL_EMBEDDING_MODEL)
    return _local_model


def _embed_local(texts: list[str]) -> tuple[list[list[float]], str | None]:
    """Generate embeddings with the local fastembed (ONNX) model."""
    try:
        model = _get_local_model()
        vectors = list(model.embed(texts))
        return [v.tolist() for v in vectors], None
    except Exception as e:
        logger.error("Local embedding failed: %s", e)
        return [], str(e)


# ---------------------------------------------------------------------------
# NVIDIA NIM provider
# ---------------------------------------------------------------------------


def _get_nvidia_client() -> OpenAI:
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY not set in .env")
    from openai import OpenAI as _OpenAI

    return _OpenAI(base_url=EMBEDDING_BASE_URL, api_key=NVIDIA_API_KEY)


def _embed_nvidia(
    texts: list[str],
    input_type: str,
) -> tuple[list[list[float]], str | None]:
    """Generate embeddings via NVIDIA NIM in batches with retries.

    The NVIDIA Llama-Nemotron embedding models are asymmetric and require an
    ``input_type`` parameter ("query" or "passage").

    Returns ``(embeddings, error)``. On full success ``error`` is ``None`` and
    ``embeddings`` covers every input text. If a batch fails after all retries,
    the function stops and returns the embeddings collected so far (aligned with
    ``texts[:len(embeddings)]``) together with a non-``None`` error message, so
    partial work is never silently discarded.
    """
    client = _get_nvidia_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=EMBEDDING_DIMENSIONS,
                    extra_body={"input_type": input_type},
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    error = (
                        f"Failed to embed batch starting at {i} "
                        f"after {max_retries} retries: {e}"
                    )
                    logger.error("%s", error)
                    return all_embeddings, error
                wait = 2 ** attempt
                logger.warning("Embedding error, retrying in %ds: %s", wait, e)
                time.sleep(wait)
        if i + EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(EMBEDDING_BATCH_SLEEP)

    return all_embeddings, None


# ---------------------------------------------------------------------------
# Public API — dispatches based on EMBEDDING_PROVIDER
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    input_type: str = "passage",
) -> tuple[list[list[float]], str | None]:
    """Generate embeddings for a list of texts.

    Dispatches to the active provider (``EMBEDDING_PROVIDER``).
    When provider is ``"nvidia"`` and NVIDIA fails, automatically
    falls back to the local fastembed model with a warning.

    Args:
        texts: List of text strings to embed.
        input_type: ``"query"`` or ``"passage"`` (NVIDIA only; ignored for local).

    Returns ``(embeddings, error)`` — error is ``None`` on success.
    """
    if EMBEDDING_PROVIDER == "local":
        return _embed_local(texts)

    # NVIDIA provider (or "nvidia" default)
    embeddings, err = _embed_nvidia(texts, input_type)
    if err is None:
        return embeddings, None

    # NVIDIA failed — try local fallback
    logger.warning(
        "NVIDIA embedding failed (%s), falling back to local model (%s). "
        "Results may be poor if ChromaDB was indexed with different dimensions.",
        err, LOCAL_EMBEDDING_MODEL,
    )
    return _embed_local(texts)


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Embed a list of chunk records. Adds 'embedding' field to each.

    Returns the enriched chunk list, or empty list on error.
    """
    if not chunks:
        return []

    texts = [c["markdown"] for c in chunks]
    model_name = (
        LOCAL_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "local" else EMBEDDING_MODEL
    )
    logger.info(
        "Embedding %d chunks (provider=%s, model=%s)...",
        len(texts), EMBEDDING_PROVIDER, model_name,
    )

    embeddings, err = embed_texts(texts, input_type="passage")
    if err is not None:
        logger.error(
            "Embedding failed: %s (collected %d/%d partial vectors, discarded)",
            err,
            len(embeddings),
            len(texts),
        )
        return []
    if len(embeddings) != len(chunks):
        logger.error(
            "Embedding count mismatch: got %d vectors for %d chunks",
            len(embeddings),
            len(chunks),
        )
        return []

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector

    logger.info("Embedded %d chunks successfully", len(chunks))
    return chunks

