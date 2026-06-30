"""enotropos — Agent module.

Agentic RAG with OpenAI tool calling (no external agent framework). A small
supervisor loop asks the LLM which tools to call, executes them while injecting
the UI-selected filters (country / gi_type / top_k) into the vector search, and
streams the final answer in a single pass — no double synthesis call.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any

from winegpt.config import LLM_MODEL, SYSTEM_PROMPT, TOP_K_CHUNKS
from winegpt.schema import StreamResult
from winegpt.tools import (
    get_table_schema,
    list_dops,
    search_graph_db,
    search_table_db,
    search_vector_db,
)

logger = logging.getLogger(__name__)

# Safety bound on supervisor iterations (tool-call rounds).
MAX_TOOL_ROUNDS = 6

# ---- Tools definition for the LLM ----
# Note: search_vector_db intentionally exposes only ``query``; the agent loop
# injects the UI-selected country / gi_type / top_k so sidebar filters always
# apply (previously they were silently dropped in agent mode).

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_vector_db",
            "description": (
                "Search the vector database of wine DOP/IGP documents. "
                "Use for: definitions, descriptions, regulations, traditional terms, "
                "winemaking practices, organoleptic characteristics, or any topic "
                "requiring textual context from official documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to search for",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_table_db",
            "description": (
                "Execute SQL query on structured wine data tables. "
                "Use for: numeric comparisons (alcohol levels, acidity, yields), "
                "listing varieties across DOPs, finding municipality names, "
                "or any question requiring precise numeric or list data.\n\n"
                + get_table_schema()
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQLite SELECT query to execute",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dops",
            "description": (
                "List available DOP and IGP names in the corpus. "
                "Use when the user asks what DOPs/IGPs are available, "
                "or needs to know which denominations exist in a region."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Filter by country (e.g. Espanya). Omit for all.",
                    },
                    "gi_type": {
                        "type": "string",
                        "enum": ["DOP", "IGP"],
                        "description": "Filter by type. Omit for all.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_graph_db",
            "description": (
                "Query the knowledge graph for entity relationships. "
                "Use for: discovering indirect connections between entities "
                "(varieties, DOPs, zones, municipalities, soil types, climate data), "
                "multi-hop traversals (e.g. variety -> DOP -> zone -> climate), "
                "cross-DOP comparisons via shared entities, or any question "
                "about how entities relate to each other beyond direct lookup."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's original question for entity extraction",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ---- Agent prompts ----

SUPERVISOR_SYSTEM = (
    "Ets un assistent especialitzat en denominacions d'origen protegides (DOP) "
    "i indicacions geogràfiques protegides (IGP) de vins. Decideix quines eines "
    "utilitzar per respondre la pregunta de l'usuari.\n\n"
    "Guia de selecció d'eines:\n"
    "- search_vector_db → preguntes sobre definicions, normativa, característiques "
    "organolèptiques, pràctiques enològiques, termes tradicionals, o qualsevol "
    "tema que requereixi context textual de documents oficials.\n"
    "- search_table_db → preguntes que demanen valors numèrics concrets (grau "
    "alcohòlic, acidesa, SO2, rendiments), llistes de varietats, comparacions "
    "numèriques entre DOPs, o cerca de municipis.\n"
    "- list_dops → l'usuari vol saber quines DOP/IGP existeixen o estan "
    "disponibles.\n"
    "- search_graph_db → preguntes sobre relacions indirectes entre entitats "
    "(ex: \"quina relació hi ha entre el clima de Rioja Alta i la varietat "
    "Mazuelo?\"), connexions multi-hop, varietats compartides entre DOPs, "
    "o qualsevol pregunta que impliqui travessar múltiples nivells de relació.\n\n"
    "Regles:\n"
    "- Si la pregunta demana valors numèrics o comparacions, usa search_table_db "
    "I TAMBÉ search_vector_db (per si les dades estan en text).\n"
    "- Si la pregunta és sobre normativa, definicions o context textual, usa "
    "search_vector_db.\n"
    "- Pots cridar múltiples eines si la pregunta ho requereix.\n"
    "- Genera SQL vàlida per SQLite. Usa LIKE per cerques parcials.\n"
    "- Si no necessites cap eina, respon directament."
)


# ---- Tool execution ----

def _call_tool(
    name: str,
    args: dict[str, Any],
    *,
    country: str | None,
    gi_type: str | None,
    top_k: int,
) -> tuple[str, list[dict[str, str]]]:
    """Execute a tool and return ``(content_json, citations)``.

    UI filters (``country`` / ``gi_type`` / ``top_k``) are injected into the
    tools that support them so the sidebar selections always take effect.
    """
    citations: list[dict[str, str]] = []
    try:
        if name == "search_vector_db":
            result = search_vector_db(
                query=args["query"],
                country=country,
                gi_type=gi_type,
                top_k=top_k,
            )
            citations = [
                c for c in result.get("citations", []) if isinstance(c, dict)
            ]
            content = json.dumps({
                "answer": result.get("answer", ""),
                "citations": [
                    f"[{c.get('ref', '')}] {c.get('gi_name', '')} — {c.get('section', '')}"
                    for c in citations
                ],
            }, ensure_ascii=False)
            return content, citations
        if name == "search_table_db":
            table_result = search_table_db(query=args["query"])
            return json.dumps(table_result, ensure_ascii=False, default=str), []
        if name == "list_dops":
            # Let the LLM choose, but coerce nulls to "no filter".
            dops_result = list_dops(
                country=args.get("country"),
                gi_type=args.get("gi_type"),
            )
            return json.dumps(dops_result, ensure_ascii=False), []
        if name == "search_graph_db":
            graph_result = search_graph_db(query=args["query"])
            return json.dumps({"graph_context": graph_result}, ensure_ascii=False), []
        return json.dumps({"error": f"Unknown tool: {name}"}), []
    except Exception as e:
        return json.dumps({"error": str(e)}), []


# ---- Streaming completion helper ----

class _StreamedCompletion:
    """Mutable holder for one streamed LLM completion."""

    kind: str  # "content" or "tool_calls"
    content: str
    tool_calls: list[dict[str, Any]]

    def __init__(self) -> None:
        self.kind = "content"
        self.content = ""
        self.tool_calls = []


def _stream_completion(
    client: Any,
    messages: list[dict[str, Any]],
    holder: _StreamedCompletion,
) -> Generator[str, None, None]:
    """Stream one chat completion, yielding content tokens.

    Populates ``holder`` with either the assembled ``content`` or the assembled
    ``tool_calls`` (by index). Used for every supervisor round so the final
    answer is streamed exactly once (no separate synthesis call).
    """
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
        max_tokens=16000,
        stream=True,
    )

    tool_acc: dict[int, dict[str, Any]] = {}
    content_parts: list[str] = []
    finish_reason: str | None = None

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "tool_calls", None):
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                acc = tool_acc.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""},
                )
                if tc_delta.id:
                    acc["id"] = tc_delta.id
                fn = tc_delta.function
                if fn:
                    if fn.name:
                        acc["name"] = fn.name
                    if fn.arguments:
                        acc["arguments"] += fn.arguments
        if delta.content:
            content_parts.append(delta.content)
            yield delta.content
        finish_reason = chunk.choices[0].finish_reason

    if tool_acc:
        holder.kind = "tool_calls"
        holder.tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    else:
        holder.kind = "content"
        holder.content = "".join(content_parts)
    _ = finish_reason  # available for debugging if needed


def _build_user_messages(
    query: str,
    conversation_history: str,
) -> list[dict[str, Any]]:
    """Build the initial message list (system + optional history + user).

    Two system messages are used: ``SYSTEM_PROMPT`` (grounding/citation rules)
    and ``SUPERVISOR_SYSTEM`` (tool-selection guidance), so the final answer
    inherits the citation discipline of the plain RAG path.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": SUPERVISOR_SYSTEM},
    ]
    if conversation_history:
        messages.append({"role": "system", "content": conversation_history})
    messages.append({"role": "user", "content": query})
    return messages


def _run_agent_loop_stream(
    client: Any,
    messages: list[dict[str, Any]],
    result: StreamResult,
    *,
    country: str | None,
    gi_type: str | None,
    top_k: int,
) -> Generator[str, None, None]:
    """Run the supervisor/tool loop, streaming the final answer.

    Yields tokens of the final answer. Populates ``result`` with the answer,
    citations and tool names.
    """
    for _round in range(MAX_TOOL_ROUNDS):
        holder = _StreamedCompletion()
        yield from _stream_completion(client, messages, holder)

        if holder.kind == "tool_calls":
            # Record the assistant's tool-call message, then execute and append
            # tool results so the next round can use them.
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in holder.tool_calls
                ],
            })
            for tc in holder.tool_calls:
                name = tc["name"]
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                logger.info("Executing tool: %s(%s)", name, args)
                content, cites = _call_tool(
                    name, args, country=country, gi_type=gi_type, top_k=top_k,
                )
                result.tools.append(name)
                if cites:
                    result.citations.extend(cites)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                })
            continue

        # Final answer round: content was already streamed above.
        result.answer = holder.content
        return

    # Exhausted tool rounds without a final answer.
    result.answer = (
        "No s'ha pogut generar una resposta final després de "
        f"{MAX_TOOL_ROUNDS} rondes d'eines."
    )
    yield result.answer


# ---- Public API ----

def query_agent_stream(
    query: str,
    country: str | None = None,
    gi_type: str | None = None,
    top_k: int = TOP_K_CHUNKS,
    conversation_history: str = "",
) -> tuple[Generator[str, None, None], StreamResult]:
    """Streaming agentic RAG.

    Returns ``(generator, result)``. Iterate the generator to render tokens;
    once exhausted, ``result.answer`` / ``result.citations`` / ``result.tools``
    are populated. UI filters are injected into every vector-search tool call.
    """
    from winegpt.llm import get_llm_client

    result = StreamResult()
    client = get_llm_client()
    messages = _build_user_messages(query, conversation_history)

    def _gen() -> Generator[str, None, None]:
        try:
            yield from _run_agent_loop_stream(
                client, messages, result,
                country=country, gi_type=gi_type, top_k=top_k,
            )
        except Exception as e:
            logger.error("Agent streaming error: %s", e)
            msg = f"\n\nError generating answer: {e}"
            result.answer = msg
            yield msg

    return _gen(), result


def query_agent(
    query: str,
    country: str | None = None,
    gi_type: str | None = None,
    top_k: int = TOP_K_CHUNKS,
    conversation_history: str = "",
) -> dict[str, Any]:
    """Non-streaming agentic RAG. Returns a dict with answer, citations, tools."""
    gen, result = query_agent_stream(
        query, country=country, gi_type=gi_type, top_k=top_k,
        conversation_history=conversation_history,
    )
    # Drain the generator (discarding streamed tokens) to populate ``result``.
    for _ in gen:
        pass
    return {
        "answer": result.answer,
        "citations": result.citations,
        "tools": result.tools,
    }
