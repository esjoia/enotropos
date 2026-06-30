"""Tests for enotropos — knowledge corpus logic."""
import tempfile
from pathlib import Path
from unittest.mock import patch

from winegpt.knowledge import _categorize, discover_knowledge_pdfs


def test_discover_knowledge_pdfs() -> None:
    """discover_knowledge_pdfs should find all PDFs under Coneixement."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus_root = Path(tmp)
        knowledge_root = corpus_root / "Coneixement"
        (knowledge_root / "enologia").mkdir(parents=True)
        (knowledge_root / "regulacio" / "EU").mkdir(parents=True)
        (knowledge_root / "Info").mkdir()
        (knowledge_root / "Info" / "notes.txt").touch()

        (knowledge_root / "enologia" / "codex.pdf").touch()
        (knowledge_root / "regulacio" / "EU" / "reg.pdf").touch()

        with patch("winegpt.knowledge.get_corpus_root", return_value=corpus_root):
            pdfs = discover_knowledge_pdfs()
            assert len(pdfs) == 2
            assert any("codex.pdf" in str(p) for p in pdfs)
            assert any("reg.pdf" in str(p) for p in pdfs)


def test_categorize_top_level() -> None:
    """_categorize should use category only for top-level files."""
    assert _categorize(Path("enologia/codex.pdf")) == ("enologia", "", "enologia")


def test_categorize_nested() -> None:
    """_categorize should detect subcategory for nested files."""
    assert _categorize(Path("regulacio/EU/reg.pdf")) == ("regulacio", "EU", "regulacio/EU")
