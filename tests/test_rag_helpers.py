"""Tests for winegpt.rag pure helper functions (no network)."""
from typing import Any

from winegpt.rag import _extract_gi_names, _rerank_chunks, build_context


def test_extract_gi_names_single() -> None:
    names = _extract_gi_names("Quines varietats té la DOP Rioja?")
    assert "Rioja" in names


def test_extract_gi_names_multiple() -> None:
    names = _extract_gi_names("Compara la DOP Rioja i la DOP Penedès")
    assert "Rioja" in names
    assert "Penedès" in names


def test_extract_gi_names_none() -> None:
    assert _extract_gi_names("Què és una denominació d'origen?") == []


def _make_chunk(
    doc: str, gi_name: str = "Rioja", distance: float = 0.5, section: str = "Varietats",
) -> dict[str, Any]:
    return {
        "id": "x",
        "document": doc,
        "metadata": {"gi_name": gi_name, "gi_type": "DOP", "section": section, "source_file": "f"},
        "distance": distance,
    }


def test_rerank_keeps_top_k() -> None:
    chunks = [_make_chunk(f"document {i}", distance=0.1 * i) for i in range(10)]
    out = _rerank_chunks("varietats Rioja", chunks, top_k=3)
    assert len(out) == 3
    assert all(isinstance(c, dict) for c in out)


def test_rerank_gi_match_bonus_ranks_higher() -> None:
    # A chunk from a matching GI but worse embedding should benefit from the bonus
    matching = _make_chunk("text random unrelated", gi_name="Rioja", distance=0.9)
    other = _make_chunk("varietats varietats varietats", gi_name="Penedès", distance=0.1)
    out = _rerank_chunks("varietats DOP Rioja", [matching, other], top_k=2)
    # The keyword-heavy chunk should still rank first; both are returned
    assert len(out) == 2


def test_rerank_fewer_than_top_k_returns_all() -> None:
    chunks = [_make_chunk("a"), _make_chunk("b")]
    out = _rerank_chunks("query", chunks, top_k=5)
    assert len(out) == 2


def test_build_context_contains_citations_and_prompt() -> None:
    chunks = [_make_chunk("Some document text", gi_name="Rioja", section="Varietats")]
    ctx, prompt = build_context("Quines varietats?", chunks)
    assert "[1]" in ctx
    assert "Rioja" in ctx
    assert "Some document text" in ctx
    assert "Quines varietats?" in prompt
    assert "## Resposta" in prompt


def test_build_context_multi_gi_instruction() -> None:
    chunks = [_make_chunk("a", gi_name="Rioja"), _make_chunk("b", gi_name="Penedès")]
    _, prompt = build_context(
        "Compara", chunks, gi_names=["Rioja", "Penedès"],
    )
    assert "múltiples regions" in prompt
