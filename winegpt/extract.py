"""enotropos — PDF extraction module.

Discovers wine GI folders (DOP_*/IGP_*) in the corpus and extracts
Markdown + JSON from each PDF. By default uses ``pymupdf4llm`` for rich
Markdown output (headings, tables, bold text); ``--fast`` falls back to
plain PyMuPDF (fitz).
"""
import argparse
import json
import logging
import re as _re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from tqdm import tqdm

from winegpt.config import (
    EXTRACTED_DIR,
    get_corpus_root,
)
from winegpt.schema import parse_folder_name
from winegpt.table_extractor import enrich_markdown_with_tables

logger = logging.getLogger(__name__)

# ---- Markdown cleaning patterns ----
# Generic patterns (apply to every country): image placeholders, page markers.
_PICTURE_RE = _re.compile(r"\*\*==> picture .+? intentionally omitted <==\*\*", _re.IGNORECASE)
_PAGE_NUMBER_RE = _re.compile(r"^\s*-\s*\d+\s*-\s*$", _re.MULTILINE)
_PAGE_HEADING_RE = _re.compile(r"^#{2,4}\s+[Ppàaág]g[eií]na\s+\d+\s*$", _re.MULTILINE)
_PAGE_WORD_RE = _re.compile(
    r"^\s*(P[àaá]g[eií]na|Pagina|Page|Seite|Pág\.)\s+\d+\s*$",
    _re.MULTILINE | _re.IGNORECASE,
)

# Country-specific boilerplate. Keyed by country so new countries can be added
# without touching the generic cleaning logic. Use ``""`` to skip country
# cleaning (e.g. for the knowledge corpus, which is not country-bound).


@dataclass(frozen=True)
class _BoilerplateConfig:
    """Country-specific markdown boilerplate to strip."""

    header_regexes: tuple["_re.Pattern[str]", ...]
    literal_headers: tuple[str, ...]


_BOILERPLATE_BY_COUNTRY: dict[str, _BoilerplateConfig] = {
    "Espanya": _BoilerplateConfig(
        header_regexes=(
            _re.compile(r"^BOLET[IÍ]N OFICIAL DEL ESTADO.*$", _re.MULTILINE | _re.IGNORECASE),
        ),
        literal_headers=(
            "**DIRECCIÓN GENERAL DE EMPRESAS AGROALIMENTARIAS Y DESARROLLO RURAL**",
            "**DIRECCIÓN GENERAL DE INDUSTRIAS Y CALIDAD AGROALIMENTARIA**",
        ),
    ),
}


def clean_markdown(md_text: str, country: str = "Espanya") -> str:
    """Remove boilerplate and noise from extracted markdown.

    Generic cleaning (image placeholders, page markers, duplicate lines) always
    runs. Country-specific boilerplate (e.g. BOE headers for Spain) only runs
    when ``country`` matches a configured entry; pass ``""`` to skip it.
    """
    # 1. Remove image placeholder lines
    md_text = _PICTURE_RE.sub("", md_text)

    # 2. Remove page number markers (e.g. "- 5 -")
    md_text = _PAGE_NUMBER_RE.sub("", md_text)

    # 2b. Remove "## Pagina X" page headings
    md_text = _PAGE_HEADING_RE.sub("", md_text)

    # 2c. Remove standalone "Página X" / "Page X" text
    md_text = _PAGE_WORD_RE.sub("", md_text)

    # 2d. Country-specific boilerplate
    cfg = _BOILERPLATE_BY_COUNTRY.get(country)
    if cfg is not None:
        for rx in cfg.header_regexes:
            md_text = rx.sub("", md_text)
        for header in cfg.literal_headers:
            md_text = md_text.replace(header, "")

    # 4. Deduplicate consecutive identical lines (common in page headers)
    lines = md_text.split("\n")
    deduped: list[str] = []
    prev = None
    for line in lines:
        stripped = line.strip()
        if stripped != prev or stripped == "":
            deduped.append(line)
        prev = stripped
    md_text = "\n".join(deduped)

    # 5. Collapse multiple consecutive blank lines (max 2)
    md_text = _re.sub(r"\n{3,}", "\n\n", md_text)

    return md_text.strip()


def discover_gis(country_path: Path) -> list[dict[str, Any]]:
    """Discover all DOP_*/IGP_* folders and their PDFs in a country directory.

    Returns a list of GI dicts with folder_name, display_name, gi_type, path, and pdfs.
    """
    gis: list[dict[str, Any]] = []
    for folder in sorted(country_path.iterdir()):
        if not folder.is_dir():
            continue
        info = parse_folder_name(folder.name)
        if not info.is_gi:
            continue

        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            continue

        gis.append({
            "folder_name": folder.name,
            "display_name": info.gi_name,
            "gi_type": info.gi_type,
            "path": folder,
            "pdfs": [p.name for p in pdfs],
            "pdf_paths": pdfs,
        })
    return gis


def _extract_fitz_text(pdf_path: Path) -> str:
    """Fast fallback extraction using PyMuPDF (fitz)."""
    doc = fitz.open(pdf_path)
    parts: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        parts.append(f"\n\n## Pagina {page_num}\n\n{text}")

    doc.close()
    return "\n".join(parts)


def extract_pdf(
    pdf_path: Path,
    use_pymupdf4llm: bool = True,
    enrich_tables: bool = False,
    country: str = "Espanya",
) -> tuple[str, list[dict[str, Any]], str]:
    """Extract Markdown and page-level data from a single PDF.

    By default uses ``pymupdf4llm`` for rich Markdown output (headings, tables,
    bold text). Falls back to plain PyMuPDF (fitz) if ``use_pymupdf4llm`` is
    False or if the rich extraction fails.

    If ``enrich_tables`` is True, runs LLM-based table detection and clean-up
    on pages that contain table structures. This adds cost and latency.

    ``country`` selects country-specific markdown cleaning (e.g. BOE headers
    for Spain). Pass ``""`` to skip country-specific cleaning (used for the
    knowledge corpus).

    Returns (markdown_str, pages_list, extraction_method).
    """
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    if use_pymupdf4llm:
        try:
            import pymupdf4llm

            md_text = str(pymupdf4llm.to_markdown(str(pdf_path)))
            method = "pymupdf4llm"
        except Exception as e:  # pragma: no cover - defensive fallback
            logger.warning(
                "pymupdf4llm failed for %s, falling back to fitz: %s",
                pdf_path.name,
                e,
            )
            md_text = _extract_fitz_text(pdf_path)
            method = "pymupdf (fitz)"
    else:
        md_text = _extract_fitz_text(pdf_path)
        method = "pymupdf (fitz)"

    # Enrich with LLM-extracted tables if requested
    if enrich_tables and use_pymupdf4llm:
        md_text = enrich_markdown_with_tables(md_text, pdf_path)
        method += " + table_llm"

    pages = [{"page": i} for i in range(1, page_count + 1)]
    markdown = clean_markdown(md_text, country=country)
    return markdown, pages, method


def extract_country(
    country: str,
    force: bool = False,
    dry_run: bool = False,
    gi_filter: str | None = None,
    fast: bool = False,
    enrich_tables: bool = False,
) -> dict[str, Any]:
    """Extract all PDFs for a given country.

    Returns stats dict with counts.
    """
    country_path = get_corpus_root() / country
    if not country_path.exists():
        logger.error("Country directory not found: %s", country_path)
        return {"extracted": 0, "skipped": 0, "errors": 0}

    gis = discover_gis(country_path)
    if gi_filter:
        gis = [gi for gi in gis if gi_filter.lower() in gi["display_name"].lower()]
        if not gis:
            logger.error("No GI found matching: %s", gi_filter)
            return {"extracted": 0, "skipped": 0, "errors": 0}
    total_pdfs = sum(len(gi["pdfs"]) for gi in gis)
    logger.info("Country: %s — %d GIs, %d PDFs", country, len(gis), total_pdfs)

    stats = {"extracted": 0, "skipped": 0, "errors": 0}

    for gi in tqdm(gis, desc="Processing GIs", unit="gi"):
        out_dir = EXTRACTED_DIR / country / gi["folder_name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for pdf_path in gi["pdf_paths"]:
            stem = pdf_path.stem
            md_path = out_dir / f"{stem}.md"
            json_path = out_dir / f"{stem}.json"

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
                    country=country,
                )

                md_path.write_text(markdown, encoding="utf-8")

                metadata = {
                    "source_file": pdf_path.name,
                    "country": country,
                    "type": gi["gi_type"],
                    "name": gi["display_name"],
                    "folder": gi["folder_name"],
                    "extraction_method": method,
                    "char_count": len(markdown),
                    "page_count": len(pages),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PDFs using pymupdf4llm")
    parser.add_argument("--country", type=str, default="Espanya", help="Country to process")
    parser.add_argument("--gi", type=str, default=None, help="Filter by GI name (e.g. 'Priorat')")
    parser.add_argument("--force", action="store_true", help="Re-extract all PDFs")
    parser.add_argument("--fast", action="store_true", help="Use fast fitz instead of pymupdf4llm")
    parser.add_argument(
        "--enrich-tables", action="store_true",
        help="Use LLM to extract clean table Markdown",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    logger.info("=== enotropos — PDF Extraction ===")
    logger.info("Corpus: %s", get_corpus_root())
    logger.info("Output: %s", EXTRACTED_DIR)

    stats = extract_country(
        args.country,
        force=args.force,
        dry_run=args.dry_run,
        gi_filter=args.gi,
        fast=args.fast,
        enrich_tables=args.enrich_tables,
    )

    logger.info("")
    logger.info("Done. Extracted: %d | Skipped: %d | Errors: %d",
                stats["extracted"], stats["skipped"], stats["errors"])


if __name__ == "__main__":
    main()
