"""Tests for winegpt.graph_builder pure helpers."""
import networkx as nx

from winegpt.graph_builder import (
    _add_triples_to_graph,
    _get_markdown_sections,
    _normalize_name,
    _parse_triples_json,
)


def test_normalize_name_titles_and_collapses_spaces() -> None:
    assert _normalize_name("  tempranillo  ") == "Tempranillo"
    assert _normalize_name("rioja   alta") == "Rioja Alta"


def test_parse_triples_json_plain_array() -> None:
    raw = '[{"subject": "Rioja", "predicate": "AUTHORIZES_VARIETY", "object": "Tempranillo"}]'
    triples = _parse_triples_json(raw, "doc")
    assert len(triples) == 1
    assert triples[0]["object"] == "Tempranillo"


def test_parse_triples_json_with_code_fence() -> None:
    raw = '```json\n[{"subject": "A", "predicate": "P", "object": "B"}]\n```'
    triples = _parse_triples_json(raw, "doc")
    assert len(triples) == 1


def test_parse_triples_json_trailing_comma_fixed() -> None:
    raw = '[{"subject": "A", "predicate": "P", "object": "B",},]'
    triples = _parse_triples_json(raw, "doc")
    assert len(triples) == 1


def test_parse_triples_json_no_array_returns_empty() -> None:
    assert _parse_triples_json("no json here", "doc") == []


def test_get_markdown_sections_splits_by_headings() -> None:
    long_a = "Tempranillo és una varietat tinta principal de la zona. " * 3
    long_b = "Rioja Alta és una subzona amb sòls argilosos i climàtic atlàntic. " * 3
    md = f"## Varietats\n\n{long_a}\n\n## Zona\n\n{long_b}"
    sections = _get_markdown_sections(md)
    headings = [h for h, _ in sections]
    assert "Varietats" in headings
    assert "Zona" in headings


def test_get_markdown_sections_skips_short_content() -> None:
    md = "## Curta\n\nxx\n\n## Llarga\n\n" + "a" * 200
    sections = _get_markdown_sections(md)
    # Short section (len <= 80) is filtered out
    headings = [h for h, _ in sections]
    assert "Curta" not in headings
    assert "Llarga" in headings


def test_add_triples_to_graph_basic() -> None:
    graph = nx.MultiDiGraph()
    triples = [
        {
            "subject": "Rioja",
            "subject_type": "DOP",
            "predicate": "AUTHORIZES_VARIETY",
            "object": "Tempranillo",
            "object_type": "Variety",
            "attrs": {"color": "tinta"},
        }
    ]
    count = _add_triples_to_graph(graph, triples, source="test")
    assert count == 1
    assert "Rioja" in graph.nodes()
    assert "Tempranillo" in graph.nodes()
    assert graph.nodes["Rioja"]["type"] == "DOP"
    assert graph.nodes["Tempranillo"]["type"] == "Variety"
    assert graph.has_edge("Rioja", "Tempranillo")
    assert graph["Rioja"]["Tempranillo"][0]["predicate"] == "AUTHORIZES_VARIETY"
    assert graph["Rioja"]["Tempranillo"][0]["color"] == "tinta"


def test_add_triples_to_graph_duplicate_nodes() -> None:
    graph = nx.MultiDiGraph()
    triples = [
        {
            "subject": "Rioja",
            "subject_type": "DOP",
            "predicate": "AUTHORIZES_VARIETY",
            "object": "Tempranillo",
            "object_type": "Variety",
            "attrs": {},
        },
        {
            "subject": "Rioja",
            "subject_type": "DOP",
            "predicate": "AUTHORIZES_VARIETY",
            "object": "Garnacha",
            "object_type": "Variety",
            "attrs": {},
        }
    ]
    count = _add_triples_to_graph(graph, triples, source="test")
    assert count == 2
    assert graph.nodes["Rioja"]["type"] == "DOP"
    assert graph.nodes["Tempranillo"]["type"] == "Variety"
    assert graph.nodes["Garnacha"]["type"] == "Variety"


def test_add_triples_to_graph_empty_triples() -> None:
    graph = nx.MultiDiGraph()
    count = _add_triples_to_graph(graph, [], source="test")
    assert count == 0
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_add_cross_relationships_authorizes_variety() -> None:
    graph = nx.MultiDiGraph()
    # Add two varieties authorized by the same DOP
    graph.add_node("Rioja", type="DOP", source="test")
    graph.add_node("Tempranillo", type="Variety", source="test")
    graph.add_node("Garnacha", type="Variety", source="test")
    graph.add_edge("Rioja", "Tempranillo", predicate="AUTHORIZES_VARIETY", source="test")
    graph.add_edge("Rioja", "Garnacha", predicate="AUTHORIZES_VARIETY", source="test")

    # Add the third variety via triple processing
    triples = [
        {
            "subject": "Rioja",
            "subject_type": "DOP",
            "predicate": "AUTHORIZES_VARIETY",
            "object": "Monastrell",
            "object_type": "Variety",
            "attrs": {},
        }
    ]
    _add_triples_to_graph(graph, triples, source="test")

    # Check that cross-relationships were created between new variety and existing ones
    assert graph.has_edge("Monastrell", "Tempranillo")
    assert graph.has_edge("Tempranillo", "Monastrell")
    assert graph["Monastrell"]["Tempranillo"][0]["predicate"] == "COAUTHORIZED_WITH"
    assert graph["Monastrell"]["Tempranillo"][0]["via"] == "Rioja"
    assert graph["Monastrell"]["Tempranillo"][0]["source"] == "cross_inference"

    assert graph.has_edge("Monastrell", "Garnacha")
    assert graph.has_edge("Garnacha", "Monastrell")
    assert graph["Monastrell"]["Garnacha"][0]["predicate"] == "COAUTHORIZED_WITH"
    assert graph["Monastrell"]["Garnacha"][0]["via"] == "Rioja"
    assert graph["Monastrell"]["Garnacha"][0]["source"] == "cross_inference"


def test_add_cross_relationships_municipality_zone() -> None:
    graph = nx.MultiDiGraph()
    # Add zone with soil and climate
    graph.add_node("Madrid Zona Norte", type="Zone", source="test")
    graph.add_node("Argiloso", type="SoilType", source="test")
    graph.add_node("Mediterráneo", type="ClimateData", source="test")
    graph.add_edge("Madrid Zona Norte", "Argiloso", predicate="HAS_SOIL", source="test")
    graph.add_edge("Madrid Zona Norte", "Mediterráneo", predicate="HAS_CLIMATE", source="test")

    # Add another soil/climate pair
    graph.add_node("Calcáreo", type="SoilType", source="test")
    graph.add_node("Continental", type="ClimateData", source="test")
    graph.add_edge("Madrid Zona Norte", "Calcáreo", predicate="HAS_SOIL", source="test")
    graph.add_edge("Madrid Zona Norte", "Continental", predicate="HAS_CLIMATE", source="test")

    # Process a new triple that adds a municipality to the zone
    triples = [
        {
            "subject": "Alcalá",
            "subject_type": "Municipality",
            "predicate": "MUNICIPALITY_IN_ZONE",
            "object": "Madrid Zona Norte",
            "object_type": "Zone",
            "attrs": {},
        }
    ]
    _add_triples_to_graph(graph, triples, source="test")

    # Check that cross-relationships were created between soils and climates of the zone
    assert graph.has_edge("Argiloso", "Mediterráneo")
    assert graph["Argiloso"]["Mediterráneo"][0]["predicate"] == "SOIL_IN_CLIMATE"
    assert graph["Argiloso"]["Mediterráneo"][0]["via"] == "Madrid Zona Norte"
    assert graph["Argiloso"]["Mediterráneo"][0]["source"] == "cross_inference"

    assert graph.has_edge("Calcáreo", "Continental")
    assert graph["Calcáreo"]["Continental"][0]["predicate"] == "SOIL_IN_CLIMATE"
    assert graph["Calcáreo"]["Continental"][0]["via"] == "Madrid Zona Norte"
    assert graph["Calcáreo"]["Continental"][0]["source"] == "cross_inference"
