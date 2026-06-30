"""Agent regression tests (3T): UI filters must reach the vector-search tool.

Validates the 0A fix — ``country`` / ``gi_type`` / ``top_k`` from the agent
entry point are injected into ``search_vector_db`` calls (previously dropped
when the agent used LangGraph). Uses a stubbed ``search_vector_db`` so no
network or ChromaDB access is required.
"""
from typing import Any, cast

import winegpt.agent as agent_mod
from winegpt.agent import TOOLS, _call_tool


def _record_tool(calls: list[dict[str, Any]]) -> Any:
    def _stub(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "answer": "stub answer",
            "citations": [{"ref": "1", "gi_name": "Rioja", "section": "s"}],
            "context_chunks": [],
            "ok": True,
        }
    return _stub


def test_call_tool_injects_filters_into_search_vector_db(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(agent_mod, "search_vector_db", _record_tool(calls))

    content, citations = _call_tool(
        "search_vector_db",
        {"query": "varietats Rioja"},
        country="Espanya",
        gi_type="DOP",
        top_k=7,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["query"] == "varietats Rioja"
    assert call["country"] == "Espanya"
    assert call["gi_type"] == "DOP"
    assert call["top_k"] == 7
    # The tool result is serialized to JSON for the LLM message.
    assert "stub answer" in content
    # Citations are surfaced back to the agent loop.
    assert citations and citations[0]["gi_name"] == "Rioja"


def test_call_tool_defaults_to_none_filters_when_not_set(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(agent_mod, "search_vector_db", _record_tool(calls))

    _call_tool(
        "search_vector_db", {"query": "x"},
        country=None, gi_type=None, top_k=5,
    )
    assert calls[0]["country"] is None
    assert calls[0]["gi_type"] is None
    assert calls[0]["top_k"] == 5


def test_search_vector_db_tool_schema_has_no_country_or_gi_type() -> None:
    """The LLM must not choose filters; the loop injects them (0A fix)."""
    tools = cast(list[dict[str, Any]], TOOLS)
    spec = next(t for t in tools if t["function"]["name"] == "search_vector_db")
    props = spec["function"]["parameters"]["properties"]
    assert "query" in props
    assert "country" not in props
    assert "gi_type" not in props
    assert "top_k" not in props


def test_list_dops_tool_enum_has_no_null() -> None:
    """JSON Schema enums must not mix ``null`` with strings (invalid)."""
    tools = cast(list[dict[str, Any]], TOOLS)
    spec = next(t for t in tools if t["function"]["name"] == "list_dops")
    props = spec["function"]["parameters"]["properties"]
    gt = props["gi_type"]
    enum = gt.get("enum")
    assert enum is not None
    assert None not in enum
