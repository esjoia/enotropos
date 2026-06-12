"""Tests for enotropos — chunking logic."""
from winegpt.chunk import chunk_by_headings, chunk_by_paragraphs, chunk_markdown


def test_chunk_by_headings_basic():
    md = "## Zona geografica\n\nContingut de la zona.\n\n## Varietats\n\nTempranillo i Garnacha."
    chunks = chunk_by_headings(md)
    assert len(chunks) == 2
    assert chunks[0]["section"] == "Zona geografica"
    assert chunks[1]["section"] == "Varietats"
    assert "Contingut" in chunks[0]["markdown"]
    assert "Tempranillo" in chunks[1]["markdown"]


def test_chunk_by_headings_subheadings():
    md = "## Zone\nContent.\n\n### Subzone\n\nMore content."
    chunks = chunk_by_headings(md)
    assert len(chunks) == 2
    assert chunks[0]["section"] == "Zone"
    assert chunks[1]["section"] == "Subzone"


def test_chunk_by_headings_empty():
    md = "No headings here.\n\nJust text."
    chunks = chunk_by_headings(md)
    assert len(chunks) == 0


def test_chunk_markdown_heading():
    """Should prefer heading-based chunking when headings exist."""
    md = "## Section 1\n\nText one.\n\n## Section 2\n\nText two."
    chunks = chunk_markdown(md)
    assert len(chunks) == 2
    assert chunks[0]["section"] == "Section 1"
    assert chunks[1]["section"] == "Section 2"


def test_chunk_markdown_fallback_paragraphs():
    """Should fall back to paragraph chunking when no headings exist."""
    md = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_markdown(md)
    assert len(chunks) >= 1
    assert chunks[0]["section"] == "General"


def test_chunk_by_paragraphs_basic():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_by_paragraphs(text)
    assert len(chunks) == 1
    assert "First" in chunks[0]["markdown"]
    assert "Second" in chunks[0]["markdown"]
    assert chunks[0]["section"] == "General"
