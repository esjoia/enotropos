"""enotropos — Build knowledge index.

Runs the full pipeline for the knowledge corpus located at::

    corpus_enotropos/Coneixement/

Usage:
    python scripts/build_knowledge_index.py
    python scripts/build_knowledge_index.py --force --reset
    python scripts/build_knowledge_index.py --dry-run
"""
import argparse
import logging
import sys

from winegpt.knowledge import build_knowledge_index

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build knowledge index for enotropos")
    parser.add_argument("--force", action="store_true", help="Re-extract all PDFs")
    parser.add_argument("--fast", action="store_true", help="Use fast fitz instead of pymupdf4llm")
    parser.add_argument(
        "--enrich-tables", action="store_true",
        help="Use LLM to extract clean table Markdown",
    )
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collection")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    result = build_knowledge_index(
        force=args.force,
        reset=args.reset,
        dry_run=args.dry_run,
        fast=args.fast,
        enrich_tables=args.enrich_tables,
    )

    logger.info("")
    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
