"""enotropos — Conversation memory module.

Stores recent conversation turns and formats them as context for the LLM,
enabling multi-turn conversations with memory of past exchanges.
"""
from __future__ import annotations

from typing import Any


class ConversationMemory:
    """Sliding-window conversation buffer.

    Stores up to ``max_turns`` user/assistant pairs and formats them
    as a conversation history string for injection into the system prompt.
    """

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._messages: list[dict[str, str]] = []

    def add_user(self, content: str) -> None:
        """Add a user message to the buffer."""
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        """Add an assistant message to the buffer."""
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        """Keep only the last max_turns pairs."""
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def get_history_text(self) -> str:
        """Format the conversation history as a text block for the prompt."""
        if not self._messages:
            return ""

        lines: list[str] = ["## Historial de la conversa"]
        for i, msg in enumerate(self._messages):
            role = "Usuari" if msg["role"] == "user" else "Assistent"
            # Truncate very long messages to keep context manageable
            content = msg["content"]
            if len(content) > 1500:
                content = content[:1500] + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def get_messages(self) -> list[dict[str, str]]:
        """Return the raw message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Reset the conversation buffer."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages) // 2


# Convenience helpers for Streamlit session state

def get_memory(max_turns: int = 10) -> Any:
    """Get or create a ConversationMemory in the current context.

    Returns the memory instance cast to Any to avoid Streamlit type issues.
    """
    import streamlit as st

    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory(max_turns=max_turns)
    return st.session_state.conversation_memory
