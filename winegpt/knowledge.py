"""enotropos — Knowledge corpus module.

Manages general wine knowledge PDFs stored under::

    corpus_enotropos/Coneixement/

These documents are not GI product specifications (DOP/IGP) but reference
material such as oenological codes, labelling standards, regulations,
viticulture catalogues and traditional terms. They are extracted, chunked,
embedded and stored in ChromaDB alongside the GI documents so the RAG can
answer broader wine-related questions.
"""

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from winegpt.chunk import MIN_CHUNK_CHARS, chunk_parent_child
from winegpt.config import EXTRACTED_DIR, get_corpus_root
from winegpt.extract import extract_pdf
from winegpt.language import detect_language
from winegpt.parents import save_parents
from winegpt.schema import make_chunk_id

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR_NAME = "Coneixement"


def get_knowledge_root() -> Path:
    """Return the path to the knowledge corpus directory."""
    return get_corpus_root() / KNOWLEDGE_DIR_NAME


def discover_knowledge_pdfs() -> list[Path]:
    """Discover all PDF files under the knowledge corpus directory."""
    knowledge_root = get_knowledge_root()
    if not knowledge_root.exists():
        logger.warning("Knowledge corpus not found: %s", knowledge_root)
        return []
    return sorted(pdf for pdf in knowledge_root.rglob("*.pdf") if pdf.is_file())


def _output_paths(pdf_path: Path) -> tuple[Path, Path]:
    """Compute Markdown and JSON output paths for a knowledge PDF.

    Preserves the folder structure of the knowledge corpus inside
    ``data/extracted/Coneixement/``.
    """
    rel_path = pdf_path.relative_to(get_knowledge_root())
    out_dir = EXTRACTED_DIR / KNOWLEDGE_DIR_NAME / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    return out_dir / f"{stem}.md", out_dir / f"{stem}.json"


def _categorize(rel_path: Path) -> tuple[str, str, str]:
    """Derive category, subcategory and folder metadata from relative path."""
    parent_parts = rel_path.parent.parts
    category = parent_parts[0] if parent_parts else "general"
    subcategory = parent_parts[1] if len(parent_parts) > 1 else ""
    folder = (
        rel_path.parent.as_posix()
        if rel_path.parent != Path(".")
        else category
    )
    return category, subcategory, folder


def extract_knowledge(
    force: bool = False,
    dry_run: bool = False,
    fast: bool = False,
    enrich_tables: bool = False,
) -> dict[str, int]:
    """Extract all knowledge PDFs to Markdown + JSON metadata.

    Returns a stats dict with ``extracted``, ``skipped`` and ``errors`` counts.
    """
    pdfs = discover_knowledge_pdfs()
    if not pdfs:
        logger.warning("No knowledge PDFs found.")
        return {"extracted": 0, "skipped": 0, "errors": 0}

    stats = {"extracted": 0, "skipped": 0, "errors": 0}

    for pdf_path in tqdm(pdfs, desc="Knowledge PDFs", unit="pdf"):
        md_path, json_path = _output_paths(pdf_path)

        if not force and md_path.exists() and json_path.exists():
            stats["skipped"] += 1
            continue

        if dry_run:
            logger.info("[dry-run] Would extract: %s", pdf_path.name)
            stats["extracted"] += 1
            continue

        try:
            markdown, pages, method = extract_pdf(
                pdf_path,
                use_pymupdf4llm=not fast,
                enrich_tables=enrich_tables,
                country="",  # knowledge corpus is not country-bound
            )
            md_path.write_text(markdown, encoding="utf-8")

            rel_path = pdf_path.relative_to(get_knowledge_root())
            category, subcategory, folder = _categorize(rel_path)
            language = detect_language(markdown)

            metadata = {
                "source_file": pdf_path.name,
                "country": KNOWLEDGE_DIR_NAME,
                "type": "knowledge",
                "category": category,
                "subcategory": subcategory,
                "folder": folder,
                "extraction_method": method,
                "char_count": len(markdown),
                "page_count": len(pages),
                "language": language,
            }
            json_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            stats["extracted"] += 1
        except Exception as e:
            logger.error("Error extracting %s: %s", pdf_path.name, e)
            stats["errors"] += 1

    return stats


def chunk_knowledge() -> list[dict[str, Any]]:
    """Chunk all extracted knowledge Markdown files.

    Returns child chunk records ready for embedding and storage. Parent sections
    are persisted to disk via ``save_parents``.
    """
    knowledge_extracted = EXTRACTED_DIR / KNOWLEDGE_DIR_NAME
    if not knowledge_extracted.exists():
        logger.warning("No extracted knowledge data found.")
        return []

    all_chunks: list[dict[str, Any]] = []

    for json_path in sorted(knowledge_extracted.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Cannot read %s: %s", json_path, e)
            continue

        md_path = json_path.with_suffix(".md")
        if not md_path.exists():
            continue

        markdown = md_path.read_text(encoding="utf-8")
        if not markdown:
            continue

        folder = data.get("folder", json_path.parent.name)
        category = data.get("category", folder)
        subcategory = data.get("subcategory", "")
        language = data.get("language", "unknown")
        pdf = json_path.stem

        # Parent-Child chunking for knowledge corpus
        parents, children = chunk_parent_child(
            markdown,
            country=KNOWLEDGE_DIR_NAME,
            folder=folder,
            pdf=pdf,
        )
        if parents:
            save_parents(KNOWLEDGE_DIR_NAME, folder, pdf, parents)

        for i, chunk in enumerate(children):
            content = chunk["markdown"]
            if len(content) < MIN_CHUNK_CHARS:
                continue

            all_chunks.append({
                "chunk_id": make_chunk_id(KNOWLEDGE_DIR_NAME, folder, pdf, i),
                "folder": folder,
                "source_file": pdf,
                "country": KNOWLEDGE_DIR_NAME,
                "gi_type": "knowledge",
                "gi_name": category,
                "subcategory": subcategory,
                "language": language,
                "section": chunk["section"],
                "markdown": content,
                "parent_id": chunk.get("parent_id", ""),
            })

        # Also embed parent texts so the vector store can match queries against
        # full section context.
        for parent in parents:
            if len(parent["markdown"]) < MIN_CHUNK_CHARS:
                continue
            all_chunks.append({
                "chunk_id": parent["parent_id"],
                "folder": folder,
                "source_file": pdf,
                "country": KNOWLEDGE_DIR_NAME,
                "gi_type": "knowledge",
                "gi_name": category,
                "subcategory": subcategory,
                "language": language,
                "section": parent["section"],
                "markdown": parent["markdown"],
                "parent_id": parent["parent_id"],
            })

    logger.info(
        "Knowledge corpus: %d child chunks across %d documents",
        len(all_chunks),
        len(list(knowledge_extracted.rglob("*.json"))),
    )
    return all_chunks


def build_knowledge_index(
    force: bool = False,
    reset: bool = False,
    dry_run: bool = False,
    fast: bool = False,
    enrich_tables: bool = False,
) -> dict[str, Any]:
    """Run the full knowledge index pipeline: extract → chunk → embed → store.

    Returns a dict with extraction stats, chunk count and stored count.
    """
    logger.info("=== enotropos — Knowledge Index ===")
    logger.info("Corpus: %s", get_knowledge_root())
    logger.info("Output: %s", EXTRACTED_DIR / KNOWLEDGE_DIR_NAME)
    logger.info("")

    # Step 1: Extract
    logger.info("--- Step 1: Extract knowledge PDFs ---")
    extract_stats = extract_knowledge(
        force=force, dry_run=dry_run, fast=fast, enrich_tables=enrich_tables
    )
    logger.info(
        "Extracted: %d | Skipped: %d | Errors: %d",
        extract_stats["extracted"],
        extract_stats["skipped"],
        extract_stats["errors"],
    )

    if dry_run:
        logger.info("Dry run complete.")
        return {"extract": extract_stats, "chunks": 0, "stored": 0}

    # Step 2: Chunk
    logger.info("")
    logger.info("--- Step 2: Chunk knowledge ---")
    chunks = chunk_knowledge()
    logger.info("Total chunks: %d", len(chunks))

    if not chunks:
        logger.warning("No chunks generated. Check that extraction completed successfully.")
        return {"extract": extract_stats, "chunks": 0, "stored": 0}

    # Step 3: Embed
    logger.info("")
    logger.info("--- Step 3: Embed knowledge ---")
    # Lazy imports to avoid loading heavy dependencies when only extracting/chunking.
    from winegpt.embed import embed_chunks
    from winegpt.store import (
        add_chunks,
        delete_by_country,
        get_client,
        reset_children_collection,
    )

    embedded = embed_chunks(chunks)
    if not embedded:
        logger.error("Embedding failed. Check NVIDIA_API_KEY.")
        return {"extract": extract_stats, "chunks": len(chunks), "stored": 0}

    # Step 4: Store
    logger.info("")
    logger.info("--- Step 4: Store knowledge ---")
    client = get_client()
    # Knowledge has its own children collection. ``reset`` recreates the
    # collection from scratch; otherwise just clear its contents.
    if reset:
        reset_children_collection(KNOWLEDGE_DIR_NAME, client)
    else:
        delete_by_country(KNOWLEDGE_DIR_NAME, client)

    count = add_chunks(embedded, KNOWLEDGE_DIR_NAME, client)
    logger.info("Stored %d knowledge chunks", count)

    logger.info("")
    logger.info("=== Knowledge index complete ===")

    return {"extract": extract_stats, "chunks": len(chunks), "stored": count}
