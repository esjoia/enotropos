"""Tests for winegpt.memory.ConversationMemory (pure in-memory logic)."""
from winegpt.memory import ConversationMemory


def test_memory_starts_empty() -> None:
    m = ConversationMemory(max_turns=3)
    assert len(m) == 0
    assert m.get_history_text() == ""
    assert m.get_messages() == []


def test_memory_add_pair_and_history() -> None:
    m = ConversationMemory(max_turns=3)
    m.add_user("Hola")
    m.add_assistant("Què vols saber?")
    assert len(m) == 1
    text = m.get_history_text()
    assert "Historial de la conversa" in text
    assert "Usuari: Hola" in text
    assert "Assistent: Què vols saber?" in text


def test_memory_trims_to_max_turns() -> None:
    m = ConversationMemory(max_turns=2)
    for i in range(5):
        m.add_user(f"q{i}")
        m.add_assistant(f"a{i}")
    # max_turns=2 -> at most 4 messages -> 2 pairs
    assert len(m._messages) <= 4
    assert len(m) == 2
    # Most recent pair retained
    msgs = m.get_messages()
    assert msgs[-2]["content"] == "q4"
    assert msgs[-1]["content"] == "a4"


def test_memory_clear() -> None:
    m = ConversationMemory(max_turns=3)
    m.add_user("x")
    m.add_assistant("y")
    m.clear()
    assert len(m) == 0
    assert m.get_messages() == []


def test_memory_truncates_long_message() -> None:
    m = ConversationMemory(max_turns=3)
    long_msg = "a" * 2000
    m.add_user(long_msg)
    text = m.get_history_text()
    assert "..." in text
    # The 2000-char message is cut to 1500 'a's (header contributes a few more).
    assert "a" * 1500 in text
    assert "a" * 1501 not in text


def test_memory_get_messages_returns_copy() -> None:
    m = ConversationMemory(max_turns=3)
    m.add_user("x")
    msgs = m.get_messages()
    msgs.clear()
    # Mutating the returned list must not affect internal state
    assert len(m.get_messages()) == 1
