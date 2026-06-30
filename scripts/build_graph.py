"""Build a NetworkX knowledge graph from wine DOP/IGP documents.

Thin CLI wrapper around ``winegpt.graph_builder.build_graph``. The graph is
built from ``data/tables.db`` (structured data) and enriched markdown files
(via LLM triple extraction) and written to ``data/graph.pkl``.
"""
from __future__ import annotations

import argparse
import logging
import sys

from winegpt.config import SUPPORTED_COUNTRIES
from winegpt.graph_builder import build_graph

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the NetworkX knowledge graph (data/graph.pkl)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the cache looks fresh.",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM triple extraction (fast tables+directory-only build).",
    )
    parser.add_argument(
        "--countries", nargs="+", default=list(SUPPORTED_COUNTRIES),
        help="Countries to scan (default: all supported).",
    )
    parser.add_argument(
        "--gi", type=str, default=None,
        help="Process only one GI (incremental; uses cached graph).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    logger.info("=== enotropos — Knowledge Graph build ===")
    build_graph(
        force=args.force,
        use_llm=not args.no_llm,
        countries=tuple(args.countries),
        gi_filter=args.gi,
    )


if __name__ == "__main__":
    main()
