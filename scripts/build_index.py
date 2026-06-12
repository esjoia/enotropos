"""enotropos — Build index pipeline.

Orchestrates the full pipeline:
1. Extract — pymupdf4llm → Markdown + JSON
2. Language — fasttext-langdetect per document
3. Chunk — split Markdown by headings
4. Embed — OpenAI text-embedding-3-small
5. Store — ChromaDB persistent

Usage:
    python scripts/build_index.py --country Espanya
    python scripts/build_index.py --country Espanya --force --reset
"""
import argparse
import logging
import sys

from winegpt.chunk import process_country as chunk_country
from winegpt.config import CORPUS_ROOT, EXTRACTED_DIR
from winegpt.embed import embed_chunks
from winegpt.extract import extract_country
from winegpt.language import process_country as language_country
from winegpt.store import add_chunks, delete_by_country, get_client, reset_collection

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG index for enotropos")
    parser.add_argument("--country", type=str, default="Espanya", help="Country to index")
    parser.add_argument("--force", action="store_true", help="Re-extract all PDFs")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collection")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    logger.info("=" * 50)
    logger.info(" enotropos — BUILD INDEX")
    logger.info("=" * 50)
    logger.info("Corpus: %s", CORPUS_ROOT)
    logger.info("Output: %s", EXTRACTED_DIR)
    logger.info("Country: %s", args.country)
    logger.info("")

    # Step 1: Extract PDFs
    logger.info("--- Step 1: Extract ---")
    extract_stats = extract_country(
        args.country,
        force=args.force,
        dry_run=args.dry_run,
    )
    logger.info("Extracted: %d | Skipped: %d | Errors: %d",
                extract_stats["extracted"], extract_stats["skipped"], extract_stats["errors"])

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    # Step 2: Detect language
    logger.info("")
    logger.info("--- Step 2: Language Detection ---")
    lang_stats = language_country(args.country)
    logger.info("Languages: %s", dict(lang_stats))

    # Step 3: Chunk documents
    logger.info("")
    logger.info("--- Step 3: Chunking ---")
    chunks = chunk_country(args.country)
    logger.info("Total chunks: %d", len(chunks))

    if not chunks:
        logger.warning("No chunks generated. Check that extraction completed successfully.")
        return

    # Step 4: Embed chunks
    logger.info("")
    logger.info("--- Step 4: Embedding ---")
    embedded = embed_chunks(chunks)
    if not embedded:
        logger.error("Embedding failed. Check OPENAI_API_KEY.")
        return

    # Step 5: Store in ChromaDB
    logger.info("")
    logger.info("--- Step 5: Store ---")
    client = get_client()
    if args.reset:
        reset_collection(client)
    else:
        delete_by_country(args.country, client)

    count = add_chunks(embedded, client)
    logger.info("Stored %d chunks", count)

    logger.info("")
    logger.info("=" * 50)
    logger.info(" BUILD COMPLETE")
    logger.info("=" * 50)
    logger.info("Run: streamlit run winegpt/app.py")


if __name__ == "__main__":
    main()
