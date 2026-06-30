"""Tests for winegpt.graph_rag entity extraction and fuzzy node matching."""
import networkx as nx

from winegpt.graph_rag import _extract_entity_names, _fuzzy_find_nodes


def test_extract_entity_names_finds_capitalized() -> None:
    names = _extract_entity_names("Quina relació hi ha entre Rioja i Tempranillo?")
    assert "Rioja" in names
    assert "Tempranillo" in names


def test_extract_entity_names_empty() -> None:
    assert _extract_entity_names("què és una denominació?") == []


def test_fuzzy_find_nodes_exact_match() -> None:
    g = nx.MultiDiGraph()
    g.add_node("Rioja", type="DOP")
    g.add_node("Tempranillo", type="Variety")
    found = _fuzzy_find_nodes(g, ["Rioja"])
    assert "Rioja" in found


def test_fuzzy_find_nodes_accent_insensitive() -> None:
    g = nx.MultiDiGraph()
    g.add_node("Rioja", type="DOP")
    found = _fuzzy_find_nodes(g, ["Ríoja"])  # accent variant
    assert "Rioja" in found


def test_fuzzy_find_nodes_contains_match() -> None:
    g = nx.MultiDiGraph()
    g.add_node("Rioja Alta", type="Zone")
    found = _fuzzy_find_nodes(g, ["Rioja"])
    assert "Rioja Alta" in found


def test_fuzzy_find_nodes_dedups() -> None:
    g = nx.MultiDiGraph()
    g.add_node("Rioja", type="DOP")
    found = _fuzzy_find_nodes(g, ["Rioja", "Rioja"])
    assert found.count("Rioja") == 1
