"""Tests for winegpt.language.detect_language (ftlangdetect mocked)."""
from winegpt.language import detect_language


def test_detect_language_returns_code(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("ftlangdetect.detect", lambda text: {"lang": "es"})
    assert detect_language("El vino tinto de Rioja es excelente.") == "es"


def test_detect_language_handles_non_dict_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("ftlangdetect.detect", lambda text: "es")
    assert detect_language("texto") == "es"


def test_detect_language_returns_unknown_on_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _raise(text: str) -> None:
        raise RuntimeError("model not loaded")
    monkeypatch.setattr("ftlangdetect.detect", _raise)
    assert detect_language("anything") == "unknown"


def test_detect_language_empty_text(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Empty snippet -> detect likely fails or returns something; ensure no crash.
    monkeypatch.setattr("ftlangdetect.detect", lambda text: {"lang": "unknown"})
    assert detect_language("") == "unknown"
