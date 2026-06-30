"""Tests for the SQL safety guard in winegpt.tools._validate_sql."""
from winegpt.tools import _validate_sql


def test_valid_select_passes() -> None:
    assert _validate_sql("SELECT dop, value FROM analytics WHERE parameter='x'") is None


def test_valid_select_with_trailing_semicolon_passes() -> None:
    assert _validate_sql("SELECT * FROM varieties;") is None


def test_valid_with_clause_passes() -> None:
    assert _validate_sql(
        "WITH t AS (SELECT * FROM analytics) SELECT * FROM t"
    ) is None


def test_non_select_rejected() -> None:
    assert _validate_sql("DELETE FROM analytics") is not None
    assert _validate_sql("INSERT INTO analytics VALUES (1)") is not None
    assert _validate_sql("DROP TABLE analytics") is not None


def test_multistatement_rejected() -> None:
    err = _validate_sql("SELECT 1; DELETE FROM analytics")
    assert err is not None
    assert ";" in err


def test_dangerous_keyword_rejected() -> None:
    assert _validate_sql("SELECT * FROM varieties; DROP TABLE analytics") is not None
    # 'drop' as a token (even without ;) is blocked by the keyword guard
    assert _validate_sql("SELECT drop FROM analytics") is not None


def test_empty_rejected() -> None:
    assert _validate_sql("") is not None
    assert _validate_sql("   ") is not None


def test_select_from_table_named_drop_is_safe() -> None:
    """A table/column literally containing 'drop' as a substring is not blocked."""
    # 'drop_table' is a single token -> does not match the 'drop' keyword token
    assert _validate_sql("SELECT * FROM drop_table") is None
