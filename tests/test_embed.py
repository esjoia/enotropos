"""Tests for winegpt.embed — NVIDIA provider batch/partial-result behaviour."""
from dataclasses import dataclass
from typing import Any

import winegpt.embed as embed_mod
from winegpt.embed import embed_texts


@dataclass
class _Emb:
    embedding: list[float]


@dataclass
class _Resp:
    data: list[Any]


def _patch_nvidia_client(monkeypatch, batch_size: int, create_fn) -> None:  # type: ignore[no-untyped-def]
    """Patch the NVIDIA provider so tests run against a fake API."""

    class _FakeClient:
        embeddings: Any

        def __init__(self) -> None:
            self.embeddings = type("E", (), {"create": staticmethod(create_fn)})()

    monkeypatch.setattr(embed_mod, "_get_nvidia_client", lambda: _FakeClient())
    monkeypatch.setattr(embed_mod, "EMBEDDING_PROVIDER", "nvidia")
    monkeypatch.setattr(embed_mod, "EMBEDDING_BATCH_SIZE", batch_size)
    monkeypatch.setattr(embed_mod, "EMBEDDING_BATCH_SLEEP", 0)


def test_embed_texts_success(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    state = {"calls": 0}

    def embeddings_create(**kwargs: Any) -> Any:
        state["calls"] += 1
        batch = kwargs["input"]
        return _Resp(data=[_Emb([0.1]) for _ in batch])

    _patch_nvidia_client(monkeypatch, batch_size=8, create_fn=embeddings_create)

    embeddings, err = embed_texts(["a", "b"], input_type="passage")
    assert err is None
    assert len(embeddings) == 2


def test_embed_texts_returns_partial_on_batch_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    state = {"success": 0}

    def embeddings_create(**kwargs: Any) -> Any:
        if state["success"] >= 1:
            raise RuntimeError("simulated downstream failure")
        state["success"] += 1
        batch = kwargs["input"]
        return _Resp(data=[_Emb([1.0, 2.0, 3.0]) for _ in batch])

    _patch_nvidia_client(monkeypatch, batch_size=2, create_fn=embeddings_create)

    embeddings, err = embed_texts(["t0", "t1", "t2", "t3"], input_type="passage")
    assert err is not None
    assert "simulated downstream failure" in err
    assert len(embeddings) == 2  # first batch only
