"""enotropos — Chunking module.

Splits extracted Markdown texts into semantic chunks for RAG indexing.
Primary strategy: split by Markdown headings (## Section).
Fallback: paragraph-based with overlap.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

from winegpt.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, EXTRACTED_DIR

logger = logging.getLogger(__name__)

# Approximate: 1 token ≈ 4 chars for Latin-alphabet text
CHARS_PER_TOKEN = 4
HEADING_RE = re.compile(r"^#{2,4}\s+(.+)$", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    """Rough token count estimation."""
    return len(text) // CHARS_PER_TOKEN


def chunk_by_headings(markdown: str) -> list[dict[str, str]]:
    """Split Markdown by ## / ### / #### headings.

    Each chunk is a dict with 'section' (heading text) and 'markdown' (content).
    """
    positions: list[tuple[int, str]] = []
    for m in HEADING_RE.finditer(markdown):
        positions.append((m.start(), m.group(1).strip()))

    if not positions:
        return []

    chunks: list[dict[str, str]] = []
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(markdown)
        content = markdown[start:end].strip()
        if content:
            chunks.append({"section": heading, "markdown": content})

    return chunks


def chunk_by_paragraphs(text: str) -> list[dict[str, str]]:
    """Fallback chunking: split by paragraphs with overlap."""
    max_chars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict[str, str]] = []
    current = ""
    section = "General"

    for para in paragraphs:
        if estimate_tokens(current + para) > CHUNK_SIZE_TOKENS and current:
            chunks.append({"section": section, "markdown": current.strip()})
            # Overlap: keep last part
            words = current.split()
            overlap_words = words[-overlap_chars // 4:] if len(words) > overlap_chars // 4 else []
            current = " ".join(overlap_words) + "\n\n" + para
        else:
            current += ("\n\n" + para) if current else para

    if current.strip():
        chunks.append({"section": section, "markdown": current.strip()})

    return chunks


def chunk_markdown(markdown: str) -> list[dict[str, str]]:
    """Split Markdown text into chunks, preferring heading-based splitting."""
    chunks = chunk_by_headings(markdown)
    if chunks:
        return chunks
    return chunk_by_paragraphs(markdown)


def process_json(json_path: Path, source_info: dict[str, str]) -> list[dict[str, Any]]:
    """Read an extraction JSON, chunk its markdown, and return chunk records.

    Each chunk dict is ready for ChromaDB insertion with metadata.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Cannot read %s: %s", json_path, e)
        return []

    # Prefer markdown from a companion .md file
    md_path = json_path.with_suffix(".md")
    if md_path.exists():
        markdown = md_path.read_text(encoding="utf-8")
    else:
        markdown = data.get("markdown", "")

    if not markdown:
        return []

    chunks = chunk_markdown(markdown)
    records: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        records.append({
            "chunk_id": f"{source_info['folder']}__{i}",
            "folder": source_info["folder"],
            "source_file": source_info["pdf"],
            "country": source_info.get("country", data.get("country", "")),
            "gi_type": source_info.get("gi_type", data.get("type", "")),
            "gi_name": source_info.get("gi_name", data.get("name", "")),
            "language": data.get("language", "unknown"),
            "section": chunk["section"],
            "markdown": chunk["markdown"],
        })

    return records


def process_country(country: str) -> list[dict[str, Any]]:
    """Chunk all extracted documents for a country. Returns list of chunk records."""
    country_extracted = EXTRACTED_DIR / country
    if not country_extracted.exists():
        logger.error("No extracted data for %s", country)
        return []

    all_chunks: list[dict[str, Any]] = []

    for json_file in sorted(country_extracted.rglob("*.json")):
        folder_name = json_file.parent.name
        pdf_name = json_file.stem

        parts = folder_name.split("_", 1)
        gi_type = parts[0] if len(parts) > 1 else ""
        gi_name = parts[1] if len(parts) > 1 else folder_name

        source_info = {
            "folder": folder_name,
            "pdf": pdf_name,
            "country": country,
            "gi_type": gi_type,
            "gi_name": gi_name,
        }

        records = process_json(json_file, source_info)
        all_chunks.extend(records)

    logger.info("Country %s: %d chunks across %d folders",
                country, len(all_chunks),
                len(set(c["folder"] for c in all_chunks)))

    return all_chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=== enotropos — Chunking ===")
    chunks = process_country("Espanya")
    logger.info("Total chunks: %d", len(chunks))


if __name__ == "__main__":
    main()
