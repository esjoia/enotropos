"""enotropos — Streamlit web application.

Chat interface for the wine DOP/IGP RAG assistant with:
- Agentic tool calling (OpenAI tool-calling loop)
- Streaming token-by-token responses
- Conversational memory (multi-turn chat)
"""
from collections.abc import Generator
from typing import Any

import streamlit as st

from winegpt.memory import get_memory
from winegpt.schema import StreamResult

st.set_page_config(
    page_title="enotropos — Wine DOP/IGP Assistant",
    page_icon="🍷",
    layout="wide",
)

st.title("🍷 enotropos")
st.caption("AI assistant for Spanish wine denominations (DOP/IGP)")

# ---- Sidebar ----
with st.sidebar:
    st.header("Filters")
    country = st.selectbox("Country", ["Espanya", "Coneixement", "All"], index=0)
    gi_type = st.selectbox("Type", ["All", "DOP", "IGP", "knowledge"], index=0)
    top_k = st.slider("Chunks to retrieve", 3, 10, 5)

    st.divider()

    use_agent = st.toggle("Agent mode (tool calling)", value=True)

    st.divider()

    memory_turns = st.slider("Conversation memory (turns)", 0, 20, 10, help=(
        "Number of past conversation turns to include as context. "
        "0 = no memory."
    ))

    if st.button("Clear memory"):
        get_memory(0).clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    from winegpt.config import EMBEDDING_PROVIDER
    embed_label = "local (bge-m3)" if EMBEDDING_PROVIDER == "local" else "NVIDIA NIM"
    st.caption(
        f"Embed: {embed_label} · pymupdf4llm · ChromaDB · DeepSeek V4 Flash\n\n"
        "Answers are based on official EU product specifications."
    )

# ---- Conversation memory ----
memory = get_memory(max_turns=memory_turns)

# ---- Chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("Sources"):
                for cit in msg["citations"]:
                    ref = cit.get("ref", "")
                    name = cit.get("gi_name", cit.get("source", ""))
                    section = cit.get("section", "")
                    source = cit.get("source_file", "")
                    st.markdown(
                        f"- **[{ref}] {name}** — {section} — `{source}`"
                    )
        if "tools" in msg:
            st.caption(f"🛠️ Tools: {', '.join(msg['tools'])}")


def _render_stream(
    gen: Generator[str, None, None],
    result: StreamResult,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Consume a streaming generator and return (answer, citations, tools)."""
    collected: list[str] = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for token in gen:
            collected.append(token)
            placeholder.markdown("".join(collected) + "▌")
    return "".join(collected), result.citations, result.tools


def _run_rag_stream(
    query_text: str, conv_history: str,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Run RAG streaming and return (answer, citations, tool_names)."""
    from winegpt.rag import query_rag_stream

    gen, result = query_rag_stream(
        query=query_text,
        country=None if country == "All" else country,
        gi_type=None if gi_type == "All" else gi_type,
        top_k=top_k,
        conversation_history=conv_history,
    )
    answer, citations, tools = _render_stream(gen, result)
    # Prefer the holder's final answer (includes any trailing error tail).
    if result.answer:
        answer = result.answer
    return answer, citations, tools


def _run_agent_stream(
    query_text: str, conv_history: str,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Run agent streaming and return (answer, citations, tool_names)."""
    from winegpt.agent import query_agent_stream

    gen, result = query_agent_stream(
        query=query_text,
        country=None if country == "All" else country,
        gi_type=None if gi_type == "All" else gi_type,
        top_k=top_k,
        conversation_history=conv_history,
    )
    answer, citations, tools = _render_stream(gen, result)
    if result.answer:
        answer = result.answer
    return answer, citations, tools


if prompt := st.chat_input("Ask about a wine denomination..."):
    # Add user message to chat and memory
    st.session_state.messages.append({"role": "user", "content": prompt})
    memory.add_user(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history
    conv_history = memory.get_history_text() if memory_turns > 0 else ""

    # Run LLM with streaming
    if use_agent:
        answer, citations, tools_used = _run_agent_stream(prompt, conv_history)
    else:
        answer, citations, tools_used = _run_rag_stream(prompt, conv_history)

    # Show tool indicator
    if tools_used:
        st.caption(f"🛠️ Tools: {', '.join(tools_used)}")

    # Show sources
    if citations:
        with st.expander("Sources"):
            for cit in citations:
                ref = cit.get("ref", "")
                name = cit.get("gi_name", cit.get("source", ""))
                section = cit.get("section", "")
                source_f = cit.get("source_file", "")
                st.markdown(
                    f"- **[{ref}] {name}** — {section} — `{source_f}`"
                )

    # Add to memory and session state
    memory.add_assistant(answer)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "tools": tools_used,
    })
