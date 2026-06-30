"""Shared test fixtures and stubs for enotropos.

Provides lightweight fakes for the OpenAI-compatible clients so that the
embedding, store, RAG and agent layers can be unit-tested without network
access or API keys. ``tmp_path`` and ``monkeypatch`` are built-in pytest
fixtures and are used directly by the tests.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class _FakeDelta:
    content: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChoice:
    delta: _FakeDelta
    finish_reason: str | None = None


@dataclass
class _FakeChunk:
    choices: list[_FakeChoice]


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChatResponse:
    choices: list[Any] = field(default_factory=list)


class _FakeCompletions:
    """Fake ``client.chat.completions`` namespace."""

    def __init__(
        self,
        chat_create: Callable[..., Any] | None = None,
        stream_chunks: list[_FakeChunk] | None = None,
    ) -> None:
        self._chat_create = chat_create
        self._stream_chunks = stream_chunks

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            if self._stream_chunks is None:
                raise AssertionError("No stream chunks configured for this fake client")
            return iter(self._stream_chunks)
        if self._chat_create is None:
            raise AssertionError("No chat_create configured for this fake client")
        return self._chat_create(**kwargs)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeEmbeddings:
    def __init__(self, create_fn: Callable[..., Any]) -> None:
        self.create = create_fn


class FakeOpenAI:
    """A configurable OpenAI-compatible client stub.

    ``embeddings_create`` maps a batch ``input`` to a response with ``.data``
    (each item having ``.embedding``). ``chat_create`` maps kwargs to a
    response with ``.choices[0].message``. ``stream_chunks`` are returned for
    streaming calls.
    """

    def __init__(
        self,
        embeddings_create: Callable[..., Any] | None = None,
        chat_create: Callable[..., Any] | None = None,
        stream_chunks: list[_FakeChunk] | None = None,
    ) -> None:
        self.embeddings = _FakeEmbeddings(embeddings_create or _default_embeddings_create)
        self.chat = _FakeChat(_FakeCompletions(chat_create, stream_chunks))


def _default_embeddings_create(**kwargs: Any) -> Any:
    input_texts = kwargs.get("input", [])
    n = len(input_texts)

    @dataclass
    class _Emb:
        embedding: list[float]

    @dataclass
    class _Resp:
        data: list[Any]

    return _Resp(data=[_Emb([0.0, 0.0, 0.0]) for _ in range(n)])


@pytest.fixture
def fake_openai() -> FakeOpenAI:
    """Return a default FakeOpenAI (override callables per test as needed)."""
    return FakeOpenAI()
