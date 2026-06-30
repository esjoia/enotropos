"""Tests for the pure parsing helpers in scripts.build_tables_db."""
from scripts.build_tables_db import (
    _classify_table,
    _clean_cell,
    _extract_gi_info,
    _extract_value_and_relation,
    _parse_markdown_table,
)


def test_clean_cell_strips_decorations() -> None:
    assert _clean_cell("**Tempranillo**") == "Tempranillo"
    assert _clean_cell("_Garnacha_") == "Garnacha"


def test_parse_markdown_table_basic() -> None:
    md = (
        "| Variedad | Color |\n"
        "|----------|-------|\n"
        "| Tempranillo | tinta |\n"
        "| Garnacha | tinta |\n"
    )
    rows = _parse_markdown_table(md)
    assert len(rows) == 2
    assert rows[0]["Variedad"] == "Tempranillo"
    assert rows[0]["Color"] == "tinta"


def test_parse_markdown_table_skips_separator() -> None:
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    rows = _parse_markdown_table(md)
    assert len(rows) == 1
    assert rows[0]["a"] == "1"


def test_classify_table_varieties() -> None:
    rows = [{"Variedad": "Tempranillo", "Color": "tinta"}]
    assert _classify_table(rows) == "varieties"


def test_classify_table_yields() -> None:
    rows = [{"Rendimiento": "5000", "kg/ha": "5000"}]
    assert _classify_table(rows) == "yields"


def test_classify_table_municipalities() -> None:
    rows = [{"Municipio": "Haro", "Zona": "Rioja Alta"}]
    assert _classify_table(rows) == "municipalities"


def test_classify_table_unknown_returns_none() -> None:
    rows = [{"Foo": "bar", "Baz": "qux"}]
    assert _classify_table(rows) is None


def test_extract_value_and_relation_european_decimal() -> None:
    value, unit, relation = _extract_value_and_relation("11,5 % Vol")
    assert value == 11.5
    assert unit == "% Vol"


def test_extract_value_and_relation_min() -> None:
    value, _unit, relation = _extract_value_and_relation("mínimo 11,5")
    assert relation == "min"
    assert value == 11.5


def test_extract_value_and_relation_max() -> None:
    value, _unit, relation = _extract_value_and_relation("máximo 1,2 g/l")
    assert relation == "max"
    assert value == 1.2


def test_extract_value_and_relation_european_thousands() -> None:
    value, _unit, _relation = _extract_value_and_relation("10.000")
    assert value == 10000.0


def test_extract_gi_info_dop() -> None:
    gt, name = _extract_gi_info("DOP_Rioja")
    assert gt == "DOP"
    assert name == "Rioja"


def test_extract_gi_info_igp_multiword() -> None:
    gt, name = _extract_gi_info("IGP_Vino_de_la_Tierra")
    assert gt == "IGP"
    assert name == "Vino de la Tierra"


def test_extract_gi_info_unknown() -> None:
    gt, name = _extract_gi_info("enologia")
    assert gt == "unknown"
    assert name == "enologia"
