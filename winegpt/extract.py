"""enotropos — PDF extraction module.

Discovers wine GI folders (DOP_*/IGP_*) in the corpus and extracts
Markdown + JSON from each PDF using pymupdf4llm (Smart Hybrid OCR).
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pymupdf4llm
from tqdm import tqdm

from winegpt.config import (
    CORPUS_ROOT,
    EXTRACTED_DIR,
    EXTRACTION_PAGE_CHUNKS,
    EXTRACTION_WRITE_IMAGES,
)

logger = logging.getLogger(__name__)

# ---- Markdown cleaning patterns ----

import re as _re

_PICTURE_RE = _re.compile(r"\*\*==> picture .+? intentionally omitted <==\*\*", _re.IGNORECASE)
_PAGE_NUMBER_RE = _re.compile(r"^\s*-\s*\d+\s*-\s*$", _re.MULTILINE)
_PAGE_HEADING_RE = _re.compile(r"^#{2,4}\s+[Pp]agina\s+\d+\s*$", _re.MULTILINE)
_BOILERPLATE_HEADERS = (
    "**DIRECCIÓN GENERAL DE EMPRESAS AGROALIMENTARIAS Y DESARROLLO RURAL**",
    "**DIRECCIÓN GENERAL DE INDUSTRIAS Y CALIDAD AGROALIMENTARIA**",
)

def clean_markdown(md_text: str) -> str:
    """Remove boilerplate and noise from extracted markdown."""
    # 1. Remove image placeholder lines
    md_text = _PICTURE_RE.sub("", md_text)

    # 2. Remove page number markers
    md_text = _PAGE_NUMBER_RE.sub("", md_text)

    # 2b. Remove "## Pagina X" page headings
    md_text = _PAGE_HEADING_RE.sub("", md_text)

    # 3. Remove known repeated boilerplate headers
    for header in _BOILERPLATE_HEADERS:
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
    md_text = _re.sub(r"\n{4,}", "\n\n\n", md_text)

    return md_text.strip()


def discover_gis(country_path: Path) -> list[dict[str, Any]]:
    """Discover all DOP_*/IGP_* folders and their PDFs in a country directory.

    Returns a list of GI dicts with folder_name, display_name, gi_type, path, and pdfs.
    """
    gis: list[dict[str, Any]] = []
    for folder in sorted(country_path.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if name.startswith("DOP_"):
            gi_type = "DOP"
            display = name[4:]
        elif name.startswith("IGP_"):
            gi_type = "IGP"
            display = name[4:]
        else:
            continue

        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            continue

        gis.append({
            "folder_name": name,
            "display_name": display,
            "gi_type": gi_type,
            "path": folder,
            "pdfs": [p.name for p in pdfs],
            "pdf_paths": pdfs,
        })
    return gis


def extract_pdf(pdf_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Extract Markdown and page-level data from a single PDF.

    Returns (markdown_str, pages_list).
    """
    md = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=EXTRACTION_WRITE_IMAGES,
        page_chunks=EXTRACTION_PAGE_CHUNKS,
    )

    pages: list[dict[str, Any]] = []
    if isinstance(md, list):
        # page_chunks=True returns list of page dicts
        full_text = ""
        for page_dict in md:
            metadata = page_dict.get("metadata", {})
            page_num = metadata.get("page_number", 0)
            text = page_dict.get("text", "")
            full_text += f"\n\n## Pagina {page_num}\n\n{text}"
            pages.append({
                "page": page_num,
                "text": text,
            })
        markdown = full_text
    else:
        # Returns plain string
        markdown = str(md)

    markdown = clean_markdown(markdown)
    return markdown, pages


def extract_country(
    country: str,
    force: bool = False,
    dry_run: bool = False,
    gi_filter: str | None = None,
) -> dict[str, Any]:
    """Extract all PDFs for a given country.

    Returns stats dict with counts.
    """
    country_path = CORPUS_ROOT / country
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
                markdown, pages = extract_pdf(pdf_path)

                md_path.write_text(markdown, encoding="utf-8")

                metadata = {
                    "source_file": pdf_path.name,
                    "country": country,
                    "type": gi["gi_type"],
                    "name": gi["display_name"],
                    "folder": gi["folder_name"],
                    "extraction_method": "pymupdf4llm",
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
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    logger.info("=== enotropos — PDF Extraction ===")
    logger.info("Corpus: %s", CORPUS_ROOT)
    logger.info("Output: %s", EXTRACTED_DIR)

    stats = extract_country(args.country, force=args.force, dry_run=args.dry_run, gi_filter=args.gi)

    logger.info("")
    logger.info("Done. Extracted: %d | Skipped: %d | Errors: %d",
                stats["extracted"], stats["skipped"], stats["errors"])


if __name__ == "__main__":
    main()
