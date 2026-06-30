"""enotropos — Table extraction with LLM.

Detects pages containing tables and uses the LLM (meta/llama-3.1-8b-instruct via
NVIDIA NIM) to convert raw extracted text into clean Markdown tables.

Tables from wine product specifications (grape varieties, yields, analytical
parameters) are critical for the RAG but ``pymupdf4llm`` often flattens them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz

from winegpt.config import EMBEDDING_BASE_URL, NVIDIA_API_KEY, TABLE_LLM_MODEL

logger = logging.getLogger(__name__)

# Shared section markers written by this module and read by
# ``scripts/build_tables_db.py``. Keep them here as the single source of truth
# so the producer and consumer never drift.
TABLES_SECTION_HEADER = "## Taules extretes amb IA"
TABLES_SUBSECTION_PREFIX = "### Taula"

TABLE_PROMPT = (
    "Converteix el text extret d'un PDF de plec de condicions de vi en una "
    "taula Markdown neta. Identifica columnes i files. Elimina el soroll "
    "(números de pàgina, encapçalaments repetits, marques d'imatge).\n\n"
    "Retorna NOMÉS la taula en format Markdown, sense text addicional.\n\n"
    "Format esperat:\n"
    "| Columna 1 | Columna 2 | Columna 3 |\n"
    "|-----------|-----------|-----------|\n"
    "| dada 1    | dada 2    | dada 3    |\n\n"
    "Text del PDF:\n"
    "{raw_text}"
)


def detect_table_pages(pdf_path: Path) -> list[int]:
    """Return 1-indexed page numbers that contain tables."""
    table_pages: list[int] = []
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for page_num in range(1, total_pages + 1):
        page = doc[page_num - 1]
        try:
            tables = page.find_tables()
        except Exception:
            doc.close()
            return []

        if tables and len(tables.tables) > 0:
            table_pages.append(page_num)

    doc.close()
    logger.debug("Found tables on %d/%d pages of %s", len(table_pages), total_pages, pdf_path.name)
    return table_pages


def _send_table_prompt(page_text: str) -> str | None:
    """Send one page of raw text to the LLM for table clean-up."""
    from openai import OpenAI

    client = OpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=NVIDIA_API_KEY,
    )

    try:
        response = client.chat.completions.create(
            model=TABLE_LLM_MODEL,
            messages=[
                {"role": "user", "content": TABLE_PROMPT.format(raw_text=page_text[:8000])},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        result: str | None = response.choices[0].message.content
        if result and 10 < len(result) < 8000:
            return result.strip()
    except Exception as e:
        logger.warning("LLM table extraction failed: %s", e)

    return None


def extract_tables_with_llm(
    pdf_path: Path,
    pages_with_tables: list[int],
) -> str:
    """Use the LLM to extract clean Markdown tables from specific pages.

    Returns a Markdown block with one ``### Taula (pàg. N)`` heading per page.
    """
    if not pages_with_tables:
        return ""

    doc = fitz.open(pdf_path)
    blocks: list[str] = []

    for page_num in pages_with_tables:
        page = doc[page_num - 1]
        page_text = page.get_text()

        if len(page_text) < 80:
            continue

        table_md = _send_table_prompt(page_text)
        if table_md:
            blocks.append(
                f"\n\n{TABLES_SUBSECTION_PREFIX} (pàg. {page_num})\n\n{table_md}"
            )

    doc.close()

    if blocks:
        blocks.insert(0, TABLES_SECTION_HEADER + "\n")
        return "\n".join(blocks)
    return ""


def enrich_markdown_with_tables(md_text: str, pdf_path: Path) -> str:
    """Detect table pages and append clean LLM-extracted tables to the markdown.

    The original markdown (usually from ``pymupdf4llm``) is left untouched;
    clean tables are appended so that the chunker can index structured data.
    """
    pages = detect_table_pages(pdf_path)
    if not pages:
        return md_text

    table_section = extract_tables_with_llm(pdf_path, pages)
    if not table_section:
        return md_text

    return md_text + "\n\n" + table_section
