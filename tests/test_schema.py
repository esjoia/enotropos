"""Tests for winegpt.schema — folder parsing, ids, and StreamResult."""
from winegpt.schema import (
    StreamResult,
    make_chunk_id,
    make_parent_id,
    parse_folder_name,
)


def test_parse_folder_name_dop() -> None:
    info = parse_folder_name("DOP_Rioja")
    assert info.gi_type == "DOP"
    assert info.gi_name == "Rioja"
    assert info.display_name == "Rioja"
    assert info.is_gi is True


def test_parse_folder_name_igp_multiword() -> None:
    info = parse_folder_name("IGP_Castilla_y_Leon")
    assert info.gi_type == "IGP"
    assert info.gi_name == "Castilla_y_Leon"
    assert info.display_name == "Castilla y Leon"
    assert info.is_gi is True


def test_parse_folder_name_dop_with_separator() -> None:
    info = parse_folder_name("DOP_Vino_de_Quality__Subzone")
    assert info.gi_type == "DOP"
    # "__" becomes " / " before single "_" -> " "
    assert info.display_name == "Vino de Quality / Subzone"


def test_parse_folder_name_knowledge() -> None:
    info = parse_folder_name("enologia")
    assert info.gi_type == "knowledge"
    assert info.gi_name == "enologia"
    assert info.display_name == "enologia"
    assert info.is_gi is False


def test_make_chunk_id_format() -> None:
    assert make_chunk_id("Espanya", "DOP_Rioja", "DOP_Rioja", 3) == (
        "Espanya__DOP_Rioja__DOP_Rioja__3"
    )


def test_make_parent_id_format() -> None:
    assert (
        make_parent_id("Espanya", "DOP_Rioja", "DOP_Rioja", 0)
        == "Espanya__DOP_Rioja__DOP_Rioja__parent_0"
    )


def test_chunk_and_parent_id_share_country_prefix() -> None:
    country, folder, pdf = "Espanya", "DOP_Rioja", "DOP_Rioja"
    assert make_chunk_id(country, folder, pdf, 1).startswith(f"{country}__")
    assert make_parent_id(country, folder, pdf, 1).startswith(f"{country}__")


def test_stream_result_defaults() -> None:
    r = StreamResult()
    assert r.answer == ""
    assert r.citations == []
    assert r.tools == []


def test_stream_result_mutable() -> None:
    r = StreamResult()
    r.answer = "x"
    r.citations.append({"ref": "1"})
    r.tools.append("search_vector_db")
    assert r.answer == "x"
    assert r.citations == [{"ref": "1"}]
    assert r.tools == ["search_vector_db"]
