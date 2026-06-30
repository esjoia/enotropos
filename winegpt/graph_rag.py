"""enotropos — Graph RAG module.

Loads the knowledge graph and extracts relevant subgraphs for RAG context.
Provides subgraph traversal and natural language formatting.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

_graph: nx.MultiDiGraph | None = None


def _get_graph() -> nx.MultiDiGraph:
    """Lazy-load the knowledge graph with auto-rebuild if stale."""
    global _graph
    if _graph is not None:
        return _graph

    from winegpt.graph_builder import build_graph, needs_rebuild

    if needs_rebuild():
        logger.info("Graph is stale, rebuilding...")
        _graph = build_graph(force=True)
    else:
        from winegpt.graph_builder import _load_cached

        cached = _load_cached()
        if cached is None:
            # Missing or incompatible schema version -> rebuild.
            logger.info("Graph cache unusable, rebuilding...")
            _graph = build_graph(force=True)
        else:
            _graph = cached
    return _graph


def _extract_entity_names(query: str) -> list[str]:
    """Extract potential entity names from a user query."""
    names: list[str] = []

    # Stop words that shouldn't be treated as entities
    _entity_stop = {
        "que", "quines", "quins", "quin", "quina", "com", "per", "amb",
        "dop", "igp", "what", "the", "how", "why", "which",
        "quin", "quina", "quines", "quins", "quina",
        "relacio", "relació", "entre", "sobre", "dels", "deles",
        "segons", "segons", "qualsevol", "quina", "algun",
    }

    # Match multi-word capitalized phrases
    matches = re.findall(r"\b([A-ZÀ-Ü][a-zà-ü]+(?:\s+[A-ZÀ-Ü][a-zà-ü]+)*)\b", query)
    for m in matches:
        name = m.strip()
        if name.lower() not in _entity_stop and len(name) > 2:
            names.append(name)

    # Also match single capitalized words
    single = re.findall(r"\b([A-ZÀ-Ü][a-zà-ü]{2,})\b", query)
    for s in single:
        if s.lower() not in _entity_stop and s not in names:
            names.append(s)

    return names


def _fuzzy_find_nodes(graph: nx.MultiDiGraph, query_entities: list[str]) -> list[str]:
    """Find graph nodes that match query entities (fuzzy matching)."""
    found: list[str] = []
    all_nodes = list(graph.nodes())

    # Normalize accents helper
    def _normalize(s: str) -> str:
        return s.lower().replace("è", "e").replace("é", "e").replace("à", "a")\
                         .replace("ò", "o").replace("í", "i").replace("ú", "u")\
                         .replace("ü", "u").replace("ç", "c")

    for entity in query_entities:
        entity_norm = _normalize(entity)

        # 1. Try exact match
        for node in all_nodes:
            if node == entity:
                found.append(node)
                break
        else:
            # 2. Try case+accent-insensitive match
            for node in all_nodes:
                if _normalize(node) == entity_norm:
                    found.append(node)
                    break
            else:
                # 3. Try contains (strict: word boundary aware)
                for node in all_nodes:
                    node_norm = _normalize(node)
                    if entity_norm in node_norm or node_norm in entity_norm:
                        if len(node) > 2 and entity_norm not in {
                            "es", "en", "de", "el", "la", "los", "del", "dop",
                        }:
                            found.append(node)
                            break

    # Deduplicate while preserving order
    seen = set()
    result = []
    for f in found:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def query_graph(query: str, max_hops: int = 2, max_edges: int = 30) -> str:
    """Extract a relevant subgraph from the knowledge graph.

    Args:
        query: User's natural language question.
        max_hops: Maximum hops from seed nodes.
        max_edges: Maximum edges to include in the output.

    Returns:
        Natural language description of the subgraph.
    """
    graph = _get_graph()
    if graph.number_of_nodes() == 0:
        return ""

    # Extract entity names from query
    entities = _extract_entity_names(query)
    if not entities:
        return ""

    # Find matching nodes
    seed_nodes = _fuzzy_find_nodes(graph, entities)
    if not seed_nodes:
        return ""

    # Extract subgraph around seed nodes
    subgraph_nodes = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(max_hops):
        new_frontier: set[str] = set()
        for node in frontier:
            for neighbor in graph.neighbors(node):
                if neighbor not in subgraph_nodes:
                    new_frontier.add(neighbor)
            for _, neighbor in graph.in_edges(node):
                if neighbor not in subgraph_nodes:
                    new_frontier.add(str(neighbor))
        subgraph_nodes.update(new_frontier)
        frontier = new_frontier
        if not frontier:
            break

    # Extract subgraph
    subgraph = graph.subgraph(subgraph_nodes)

    # Format as natural language
    return _format_subgraph(subgraph, seed_nodes, max_edges)


def _format_subgraph(
    subgraph: nx.MultiDiGraph,
    seed_nodes: list[str],
    max_edges: int = 30,
) -> str:
    """Format a subgraph as natural language text for the LLM prompt."""
    if subgraph.number_of_edges() == 0:
        return ""

    lines: list[str] = ["## Coneixements del graf de relacions"]

    # Collect all edges with predicates
    edges_info: list[tuple[str, str, str, str, str]] = []
    for u, v, data in subgraph.edges(data=True):
        pred = data.get("predicate", "RELATED_TO")
        u_type = subgraph.nodes[u].get("type", "Entity")
        v_type = subgraph.nodes[v].get("type", "Entity")
        edges_info.append((u, u_type, pred, v, v_type))

    # Sort: prioritize meaningful relationships first
    seed_set = set(seed_nodes)
    _predicate_priority = {
        "AUTHORIZES_VARIETY": 0,
        "VARIETY_HAS_COLOR": 1,
        "COAUTHORIZED_WITH": 2,
        "SHARES_VARIETY_WITH": 3,
        "MUNICIPALITY_IN_ZONE": 4,
        "MUNICIPALITY_IN_PROVINCE": 5,
        "CONTAINS_MUNICIPALITY": 6,
        "ZONE_IN_DOP": 7,
        "PROVINCE_CONTAINS_DOP": 8,
        "REQUIRES_PARAMETER": 9,
        "REGULATES_YIELD": 10,
        "DEFINES_AGING": 11,
        "HAS_SOIL": 12,
        "HAS_CLIMATE": 13,
        "PRODUCES_WINE_TYPE": 14,
        "SOIL_IN_CLIMATE": 15,
    }
    edges_info.sort(key=lambda x: (
        0 if x[0] in seed_set else 1,
        _predicate_priority.get(x[2], 50),
    ))

    # Format edges, limiting total
    included = 0
    for u, u_type, pred, v, v_type in edges_info[:max_edges]:
        pred_label = PREDICATE_LABELS.get(pred, pred)
        line = f"- {u} ({u_type}) {pred_label} {v} ({v_type})"
        lines.append(line)
        included += 1

    if subgraph.number_of_edges() > max_edges:
        lines.append(f"- ... i {subgraph.number_of_edges() - max_edges} relacions més")

    return "\n".join(lines)


PREDICATE_LABELS: dict[str, str] = {
    "AUTHORIZES_VARIETY": "→ autoritza la varietat →",
    "CONTAINS_MUNICIPALITY": "→ conté el municipi →",
    "MUNICIPALITY_IN_ZONE": "→ pertany a la zona →",
    "MUNICIPALITY_IN_PROVINCE": "→ està a la província →",
    "REQUIRES_PARAMETER": "→ requereix el paràmetre →",
    "REGULATES_YIELD": "→ regula el rendiment →",
    "DEFINES_AGING": "→ defineix l'envelliment →",
    "HAS_SOIL": "→ té sòl →",
    "HAS_CLIMATE": "→ té clima →",
    "PRODUCES_WINE_TYPE": "→ produeix el tipus →",
    "VARIETY_HAS_COLOR": "→ és de color →",
    "AGING_APPLIES_TO": "→ aplica a →",
    "PARAMETER_MEASURED_IN": "→ es mesura en →",
    "COAUTHORIZED_WITH": "→ coautoritzada amb →",
    "SOIL_IN_CLIMATE": "→ es troba al clima →",
    "SHARES_VARIETY_WITH": "→ comparteix varietat amb →",
    "ZONE_IN_DOP": "→ està dins de →",
    "PROVINCE_CONTAINS_DOP": "→ conté la DOP →",
}


def get_graph_stats() -> dict[str, Any]:
    """Return statistics about the knowledge graph."""
    graph = _get_graph()
    if graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "node_types": {}, "predicates": {}}

    node_types: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    predicates: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        p = data.get("predicate", "RELATED_TO")
        predicates[p] = predicates.get(p, 0) + 1

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": dict(sorted(node_types.items(), key=lambda x: -x[1])),
        "predicates": dict(sorted(predicates.items(), key=lambda x: -x[1])),
    }
