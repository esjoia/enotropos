"""enotropos — Shared index-build pipeline.

Centralizes the extract → language → chunk → embed → store orchestration that
was previously inlined in ``scripts/build_index.py``. Moving it into the
library makes it testable and reusable; ``scripts/build_index.py`` is now a
thin CLI wrapper around ``run_gi_index``.
"""
from __future__ import annotations

import logging
from typing import Any

from winegpt.chunk import process_country as chunk_country
from winegpt.config import EXTRACTED_DIR, get_corpus_root
from winegpt.extract import extract_country
from winegpt.language import process_country as language_country

logger = logging.getLogger(__name__)


def run_gi_index(
    country: str,
    force: bool = False,
    fast: bool = False,
    enrich_tables: bool = False,
    reset: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full GI index pipeline for a country.

    Steps: extract → language detection → chunk → embed → store (ChromaDB).

    Returns a dict with ``extract``, ``languages``, ``chunks``, ``stored`` and
    ``ok`` entries. On a dry run, only ``extract`` is populated.
    """
    logger.info("=" * 50)
    logger.info(" enotropos — BUILD INDEX")
    logger.info("=" * 50)
    logger.info("Corpus: %s", get_corpus_root())
    logger.info("Output: %s", EXTRACTED_DIR)
    logger.info("Country: %s", country)
    logger.info("")

    result: dict[str, Any] = {"ok": False}

    # Step 1: Extract PDFs
    logger.info("--- Step 1: Extract ---")
    extract_stats = extract_country(
        country,
        force=force,
        dry_run=dry_run,
        fast=fast,
        enrich_tables=enrich_tables,
    )
    logger.info(
        "Extracted: %d | Skipped: %d | Errors: %d",
        extract_stats["extracted"], extract_stats["skipped"], extract_stats["errors"],
    )
    result["extract"] = extract_stats

    if dry_run:
        logger.info("Dry run complete.")
        result.update({"languages": {}, "chunks": 0, "stored": 0, "ok": True})
        return result

    # Step 2: Detect language
    logger.info("")
    logger.info("--- Step 2: Language Detection ---")
    lang_stats = language_country(country)
    logger.info("Languages: %s", dict(lang_stats))
    result["languages"] = dict(lang_stats)

    # Step 3: Chunk documents
    logger.info("")
    logger.info("--- Step 3: Chunking ---")
    chunks = chunk_country(country)
    logger.info("Total chunks: %d", len(chunks))
    result["chunks"] = len(chunks)

    if not chunks:
        logger.warning("No chunks generated. Check that extraction completed successfully.")
        result.update({"stored": 0, "ok": False})
        return result

    # Step 4: Embed chunks (lazy: pulls openai + NVIDIA client only when needed)
    logger.info("")
    logger.info("--- Step 4: Embedding ---")
    from winegpt.embed import embed_chunks

    embedded = embed_chunks(chunks)
    if not embedded:
        logger.error("Embedding failed. Check NVIDIA_API_KEY.")
        result.update({"stored": 0, "ok": False})
        return result

    # Step 5: Store in ChromaDB (lazy: pulls chromadb only when needed)
    logger.info("")
    logger.info("--- Step 5: Store ---")
    from winegpt.store import (
        add_chunks,
        delete_by_country,
        get_client,
        reset_children_collection,
    )

    client = get_client()
    if reset:
        reset_children_collection(country, client)
    else:
        delete_by_country(country, client)

    count = add_chunks(embedded, country, client)
    logger.info("Stored %d chunks", count)
    result["stored"] = count
    result["ok"] = True

    logger.info("")
    logger.info("=" * 50)
    logger.info(" BUILD COMPLETE")
    logger.info("=" * 50)
    logger.info("Run: streamlit run winegpt/app.py")

    return result
