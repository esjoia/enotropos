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

from winegpt.config import (
    CHILD_CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EXTRACTED_DIR,
    PARENT_CHUNK_SIZE_TOKENS,
)
from winegpt.parents import save_parents
from winegpt.schema import make_chunk_id, make_parent_id, parse_folder_name

logger = logging.getLogger(__name__)

__all__ = [
    "CHARS_PER_TOKEN",
    "CHUNK_OVERLAP_TOKENS",
    "CHUNK_SIZE_TOKENS",
    "CHILD_CHUNK_SIZE_TOKENS",
    "PARENT_CHUNK_SIZE_TOKENS",
    "chunk_by_headings",
    "chunk_by_paragraphs",
    "chunk_markdown",
    "chunk_parent_child",
    "estimate_tokens",
    "process_country",
    "process_json",
]

# Approximate: 1 token ≈ 4 chars for Latin-alphabet text
CHARS_PER_TOKEN = 4
HEADING_RE = re.compile(r"^#{2,4}\s+(.+)$", re.MULTILINE)

# Minimum meaningful chunk size (in chars) — discard chunks shorter than this
MIN_CHUNK_CHARS = 150

# Heading text to ignore (page markers, not real sections)
_IGNORE_HEADINGS_RE: re.Pattern[str] | None = None


def _ignore_heading(heading: str) -> bool:
    """Check if a heading is just a page marker (should not split on it)."""
    global _IGNORE_HEADINGS_RE
    if _IGNORE_HEADINGS_RE is None:
        _IGNORE_HEADINGS_RE = re.compile(
            r"(?i)^(pagina?|pàgina?|page|página?)\s+\d+", re.MULTILINE
        )
    return bool(_IGNORE_HEADINGS_RE.match(heading.strip()))


def estimate_tokens(text: str) -> int:
    """Rough token count estimation."""
    return len(text) // CHARS_PER_TOKEN


def chunk_by_headings(markdown: str) -> list[dict[str, str]]:
    """Split Markdown by ## / ### / #### headings, skipping page-number headings."""
    positions: list[tuple[int, str]] = []
    for m in HEADING_RE.finditer(markdown):
        heading_text = m.group(1).strip()
        if _ignore_heading(heading_text):
            continue
        positions.append((m.start(), heading_text))

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


def _combine_parts(parts: list[str], section: str, separator: str) -> list[dict[str, str]]:
    """Combine parts into chunks that stay under ``CHUNK_SIZE_TOKENS``."""
    max_chars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
    chunks: list[dict[str, str]] = []
    current = ""

    for part in parts:
        candidate = (current + separator + part).strip() if current else part
        if current and len(candidate) > max_chars:
            chunks.append({"section": section, "markdown": current.strip()})
            current = part
        else:
            current = candidate

    if current.strip():
        chunks.append({"section": section, "markdown": current.strip()})

    return chunks


def _split_text(text: str, section: str) -> list[dict[str, str]]:
    """Split text hierarchically: paragraphs → lines → sentences → words."""
    max_chars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return [{"section": section, "markdown": text}]

    # 1. Try paragraphs
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) > 1:
        return _combine_parts(parts, section, "\n\n")

    # 2. Try lines (useful for tables / bullet lists)
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    if len(parts) > 1:
        return _combine_parts(parts, section, "\n")

    # 3. Try sentences
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(parts) > 1:
        return _combine_parts(parts, section, " ")

    # 4. Last resort: words
    parts = text.split()
    return _combine_parts(parts, section, " ")


def _split_long_chunk(markdown: str, section: str) -> list[dict[str, str]]:
    """Split an oversized heading chunk while keeping its section.

    First groups paragraphs; any paragraph that is still too long is split
    hierarchically by lines, sentences, or words.
    """
    if estimate_tokens(markdown) <= CHUNK_SIZE_TOKENS:
        return [{"section": section, "markdown": markdown}]

    paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
    chunks: list[dict[str, str]] = []
    current = ""

    for para in paragraphs:
        if len(para) > CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN:
            if current:
                chunks.append({"section": section, "markdown": current.strip()})
                current = ""
            chunks.extend(_split_text(para, section))
            continue

        candidate = (current + "\n\n" + para).strip() if current else para
        if current and len(candidate) > CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN:
            chunks.append({"section": section, "markdown": current.strip()})
            current = para
        else:
            current = candidate

    if current.strip():
        chunks.append({"section": section, "markdown": current.strip()})

    return chunks


def chunk_markdown(markdown: str) -> list[dict[str, str]]:
    """Split Markdown text into chunks, preferring heading-based splitting.

    Heading-based chunks that exceed ``CHUNK_SIZE_TOKENS`` are further split
    by paragraphs so that no chunk exceeds the embedding model's input limit.
    """
    heading_chunks = chunk_by_headings(markdown)
    if not heading_chunks:
        return chunk_by_paragraphs(markdown)

    chunks: list[dict[str, str]] = []
    for chunk in heading_chunks:
        if estimate_tokens(chunk["markdown"]) > CHUNK_SIZE_TOKENS:
            chunks.extend(_split_long_chunk(chunk["markdown"], chunk["section"]))
        else:
            chunks.append(chunk)
    return chunks


def chunk_parent_child(
    markdown: str,
    country: str,
    folder: str,
    pdf: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split Markdown into parent-child chunk pairs for retrieval.

    Parents are large sections (up to PARENT_CHUNK_SIZE_TOKENS) that will be
    sent to the LLM as context. Children are small fragments derived from
    each parent (up to CHILD_CHUNK_SIZE_TOKENS) that will be embedded and
    searched via vector similarity.

    Returns ``(parents, children)`` where:
      - parents: list of dicts with ``parent_id``, ``section`` and ``markdown``
      - children: list of dicts with ``section``, ``markdown`` and ``parent_id``
    """
    # Step 1: build parent sections (heading-based, may exceed parent limit)
    heading_chunks = chunk_by_headings(markdown)
    if not heading_chunks:
        heading_chunks = chunk_by_paragraphs(markdown)

    parent_max_chars = PARENT_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
    child_max_chars = CHILD_CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN

    parents: list[dict[str, str]] = []
    for hc in heading_chunks:
        # Split oversized sections into parent-sized blocks
        if len(hc["markdown"]) > parent_max_chars:
            sub_parts = _split_text(hc["markdown"], hc["section"])
            # Merge sub_parts that still fit in a single parent
            current_parent = ""
            for sp in sub_parts:
                candidate = (current_parent + "\n\n" + sp["markdown"]).strip()
                if current_parent and len(candidate) > parent_max_chars:
                    parents.append({"section": hc["section"], "markdown": current_parent.strip()})
                    current_parent = sp["markdown"]
                else:
                    current_parent = candidate
            if current_parent.strip():
                parents.append({"section": hc["section"], "markdown": current_parent.strip()})
        else:
            parents.append(hc)

    # Step 2: derive children from each parent and assign global ids
    children_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    for parent_idx, parent in enumerate(parents):
        parent_id = make_parent_id(country, folder, pdf, parent_idx)
        parent_text = parent["markdown"]
        section = parent["section"]

        parent_records.append({
            "parent_id": parent_id,
            "section": section,
            "markdown": parent_text,
        })

        # Split parent into child-sized fragments
        if len(parent_text) <= child_max_chars:
            children: list[dict[str, str]] = [{"section": section, "markdown": parent_text}]
        else:
            children = _split_text(parent_text, section)
            # Further split any child still over the child limit
            final_children: list[dict[str, str]] = []
            for child in children:
                if len(child["markdown"]) > child_max_chars:
                    final_children.extend(_split_text(child["markdown"], section))
                else:
                    final_children.append(child)
            children = final_children

        for child in children:
            if len(child["markdown"]) < MIN_CHUNK_CHARS:
                continue
            children_records.append({
                "section": section,
                "markdown": child["markdown"],  # small fragment for embedding
                "parent_id": parent_id,
            })

    return parent_records, children_records


def process_json(json_path: Path, source_info: dict[str, str]) -> list[dict[str, Any]]:
    """Read an extraction JSON, chunk its markdown, and return chunk records.

    Uses Parent-Child chunking: children are small fragments stored in ChromaDB
    for precise vector search; parents are persisted to disk and referenced by
    ``parent_id`` so that the LLM receives the full section context at query time.
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

    country = source_info.get("country", data.get("country", ""))
    folder = source_info["folder"]
    pdf = source_info["pdf"]

    # Use Parent-Child chunking: children for embedding, parents for LLM context
    parents, children = chunk_parent_child(markdown, country, folder, pdf)
    if parents:
        save_parents(country, folder, pdf, parents)

    records: list[dict[str, Any]] = []
    for i, chunk in enumerate(children):
        content = chunk["markdown"]

        if len(content) < MIN_CHUNK_CHARS:
            continue

        records.append({
            "chunk_id": make_chunk_id(country, folder, pdf, i),
            "folder": folder,
            "source_file": pdf,
            "country": country,
            "gi_type": source_info.get("gi_type", data.get("type", "")),
            "gi_name": source_info.get("gi_name", data.get("name", "")),
            "subcategory": source_info.get("subcategory", data.get("subcategory", "")),
            "language": data.get("language", "unknown"),
            "section": chunk["section"],
            "markdown": content,
            # Parent-child reference
            "parent_id": chunk.get("parent_id", ""),
        })

    # Also embed parent texts so the vector store can match queries against full
    # section context. Parents share the same parent_id as their children so the
    # store deduplication keeps only one hit per section.
    for parent in parents:
        if len(parent["markdown"]) < MIN_CHUNK_CHARS:
            continue
        records.append({
            "chunk_id": parent["parent_id"],
            "folder": folder,
            "source_file": pdf,
            "country": country,
            "gi_type": source_info.get("gi_type", data.get("type", "")),
            "gi_name": source_info.get("gi_name", data.get("name", "")),
            "subcategory": source_info.get("subcategory", data.get("subcategory", "")),
            "language": data.get("language", "unknown"),
            "section": parent["section"],
            "markdown": parent["markdown"],
            "parent_id": parent["parent_id"],
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

        info = parse_folder_name(folder_name)

        source_info = {
            "folder": folder_name,
            "pdf": pdf_name,
            "country": country,
            "gi_type": info.gi_type,
            "gi_name": info.gi_name,
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
