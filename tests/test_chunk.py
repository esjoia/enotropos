"""Tests for enotropos — chunking logic."""
from winegpt.chunk import (
    CHARS_PER_TOKEN,
    CHUNK_SIZE_TOKENS,
    chunk_by_headings,
    chunk_by_paragraphs,
    chunk_markdown,
    chunk_parent_child,
)


def test_chunk_by_headings_basic() -> None:
    md = "## Zona geografica\n\nContingut de la zona.\n\n## Varietats\n\nTempranillo i Garnacha."
    chunks = chunk_by_headings(md)
    assert len(chunks) == 2
    assert chunks[0]["section"] == "Zona geografica"
    assert chunks[1]["section"] == "Varietats"
    assert "Contingut" in chunks[0]["markdown"]
    assert "Tempranillo" in chunks[1]["markdown"]


def test_chunk_by_headings_subheadings() -> None:
    md = "## Zone\nContent.\n\n### Subzone\n\nMore content."
    chunks = chunk_by_headings(md)
    assert len(chunks) == 2
    assert chunks[0]["section"] == "Zone"
    assert chunks[1]["section"] == "Subzone"


def test_chunk_by_headings_empty() -> None:
    md = "No headings here.\n\nJust text."
    chunks = chunk_by_headings(md)
    assert len(chunks) == 0


def test_chunk_markdown_heading() -> None:
    """Should prefer heading-based chunking when headings exist."""
    md = "## Section 1\n\nText one.\n\n## Section 2\n\nText two."
    chunks = chunk_markdown(md)
    assert len(chunks) == 2
    assert chunks[0]["section"] == "Section 1"
    assert chunks[1]["section"] == "Section 2"


def test_chunk_markdown_fallback_paragraphs() -> None:
    """Should fall back to paragraph chunking when no headings exist."""
    md = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_markdown(md)
    assert len(chunks) >= 1
    assert chunks[0]["section"] == "General"


def test_chunk_by_paragraphs_basic() -> None:
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_by_paragraphs(text)
    assert len(chunks) == 1
    assert "First" in chunks[0]["markdown"]
    assert "Second" in chunks[0]["markdown"]
    assert chunks[0]["section"] == "General"


def test_chunk_markdown_splits_oversized_heading() -> None:
    """Long sections under one heading must be split to fit the token limit."""
    long_para = "word " * (CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN)
    md = f"## Section\n\n{long_para}\n\nAnother paragraph."
    chunks = chunk_markdown(md)
    assert len(chunks) >= 2
    assert all(chunk["section"] == "Section" for chunk in chunks)
    assert all(len(chunk["markdown"]) >= 10 for chunk in chunks)
    # No chunk should exceed the configured token limit (with rough estimate)
    assert all(
        len(chunk["markdown"]) // CHARS_PER_TOKEN <= CHUNK_SIZE_TOKENS * 2
        for chunk in chunks
    )


def test_chunk_parent_child_structure() -> None:
    """Parent-Child chunking should return separate parents and children."""
    md = "## Section A\n\n" + ("Text one. " * 50) + "\n\n## Section B\n\n" + ("Text two. " * 50)
    parents, children = chunk_parent_child(md, "Espanya", "DOP_Rioja", "DOP_Rioja")

    assert len(parents) == 2
    assert len(children) >= 2

    # Parent ids must be globally unique
    parent_ids = {p["parent_id"] for p in parents}
    assert len(parent_ids) == len(parents)
    assert all(pid.startswith("Espanya__DOP_Rioja__DOP_Rioja__parent_") for pid in parent_ids)

    # Every child must reference one of the parents
    assert all(child["parent_id"] in parent_ids for child in children)
    assert "parent_text" not in children[0]

    # Parents carry full section text
    assert "Section A" in parents[0]["markdown"]
    assert "Section B" in parents[1]["markdown"]
