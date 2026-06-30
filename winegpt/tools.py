"""enotropos — Tools module for the agent.

Provides tools that the Agent Supervisor can call:
- search_vector_db: semantic search via ChromaDB (existing RAG)
- search_table_db: SQL queries against the structured tables database
- list_dops: list available DOP/IGP names
"""
from __future__ import annotations

import sqlite3
from typing import Any

from winegpt.config import DATA_DIR, TABLES_DB_PATH
from winegpt.schema import parse_folder_name

# ---- Tool 1: search_vector_db ----

def search_vector_db(
    query: str,
    country: str | None = None,
    gi_type: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Semantic search over the vector database of wine documents.

    Use this for questions about wine characteristics, definitions,
    regulations, traditional terms, or any topic requiring textual context.
    Not suitable for numeric comparisons across DOPs.

    Args:
        query: The search query in natural language.
        country: Filter by country (Espanya, Coneixement, or None for all).
        gi_type: Filter by type (DOP, IGP, knowledge, or None for all).
        top_k: Number of results to return.

    Returns:
        Dict with answer, citations, and context_chunks.
    """
    from winegpt.rag import query_rag

    return query_rag(
        query=query,
        country=country,
        gi_type=gi_type,
        top_k=top_k,
    )


# ---- Tool 2: search_table_db ----

# Defense-in-depth keyword blocklist (in addition to PRAGMA query_only).
_DANGEROUS_KEYWORDS = frozenset({
    "drop", "delete", "insert", "update", "alter", "create", "attach",
    "detach", "pragma", "replace", "merge", "vacuum", "reindex",
})


def _validate_sql(query: str) -> str | None:
    """Validate that ``query`` is a safe read-only SELECT.

    Returns an error message string if the query is rejected, or ``None`` if it
    is acceptable. Pure function (no I/O) so it can be unit-tested directly.
    """
    if not query or not query.strip():
        return "Empty query."

    stripped = query.strip()

    # Reject multi-statement queries: a lone ';' (outside a trailing one) is a
    # strong injection signal. Legitimate SELECT tools never need ';'.
    body = stripped.rstrip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    if ";" in body:
        return "Multi-statement queries (';') are not allowed."

    clean = body.lower()
    # Must be a SELECT (optionally prefixed with optional WITH clause).
    if not (clean.startswith("select") or clean.startswith("with")):
        return "Only SELECT queries are allowed."

    # Token-level dangerous keyword check (defense in depth; PRAGMA query_only
    # is the real enforcement). Split on non-alphanumeric so "drop_table" does
    # not falsely match "drop".
    import re as _re

    tokens = {t for t in _re.split(r"[^a-z_]+", clean) if t}
    blocked = tokens & _DANGEROUS_KEYWORDS
    if blocked:
        return f"Keyword(s) not allowed: {', '.join(sorted(blocked))}."

    return None


def search_table_db(query: str) -> list[dict[str, Any]]:
    """Execute an SQL query against the structured wine tables database.

    The database contains analytical parameters (alcohol, acidity, SO2),
    authorized grape varieties, yield limits, and municipalities for
    Spanish wine DOPs and IGPs.

    Available tables:
    - analytics(dop, gi_type, wine_type, parameter, value, unit, relation)
      Parameters: grado_alcoholico, acidez_volatil, acidez_total,
                  so2_total, azucares_residuales, azucares_totales
      wine_type: blanco, rosado, tinto, espumoso, generoso, general
      relation: min, max, target
    - varieties(dop, gi_type, variety, color, role)
      color: blanca, tinta, unknown
      role: principal, secundaria, autorizada
    - yields(dop, gi_type, wine_type, variety, kg_ha, hl_ha)
    - municipalities(dop, gi_type, municipality, zone, province)

    Example queries:
    - \"SELECT dop, value, unit FROM analytics WHERE parameter='grado_alcoholico'
       AND wine_type='tinto' AND relation='min' ORDER BY value DESC\"
    - \"SELECT dop, variety FROM varieties WHERE variety LIKE '%Garnacha%'\"
    - \"SELECT dop, kg_ha FROM yields WHERE wine_type='tinto' ORDER BY kg_ha DESC\"

    Args:
        query: A valid SQLite SELECT query.

    Returns:
        List of dicts with query results.
    """
    if not TABLES_DB_PATH.exists():
        return [{"error": "Tables database not found. Run scripts/build_tables_db.py first."}]

    err = _validate_sql(query)
    if err is not None:
        return [{"error": err}]

    # Execute the validated body (without a trailing semicolon) in read-only
    # mode so that even a bypassed guard cannot mutate the database.
    body = query.strip().rstrip()
    if body.endswith(";"):
        body = body[:-1]

    conn = sqlite3.connect(str(TABLES_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = 1")
        cursor = conn.execute(body)
        rows = [dict(r) for r in cursor.fetchall()]
        return rows[:50]  # Limit results
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        conn.close()


# ---- Tool 3: list_dops ----

def list_dops(
    country: str | None = None,
    gi_type: str | None = None,
) -> list[dict[str, str]]:
    """List available DOP / IGP names in the corpus.

    Args:
        country: Filter by country (Espanya, Coneixement). None for all.
        gi_type: Filter by type (DOP, IGP). None for all.

    Returns:
        List of dicts with 'name' and 'type' keys.
    """
    extracted_dir = DATA_DIR / "extracted"
    results: list[dict[str, str]] = []

    for country_dir in sorted(extracted_dir.iterdir()):
        if not country_dir.is_dir():
            continue
        country_name = country_dir.name
        if country and country_name != country:
            continue

        for gi_dir in sorted(country_dir.iterdir()):
            if not gi_dir.is_dir():
                continue
            info = parse_folder_name(gi_dir.name)

            if gi_type and info.gi_type != gi_type:
                continue

            results.append({"name": info.display_name, "type": info.gi_type})

    return results


# ---- Tool 4: get_table_schema ----

def get_table_schema() -> str:
    """Return the schema of the tables database (for the agent's SQL generation)."""
    return (
        "Tables in the database:\n\n"
        "1. analytics(dop TEXT, gi_type TEXT, wine_type TEXT, parameter TEXT, "
        "value REAL, unit TEXT, relation TEXT)\n"
        "   Parameters: grado_alcoholico, acidez_volatil, acidez_total, "
        "so2_total, azucares_residuales, azucares_totales\n"
        "   wine_type values: blanco, rosado, tinto, espumoso, generoso, general\n"
        "   relation values: min, max, target\n\n"
        "2. varieties(dop TEXT, gi_type TEXT, variety TEXT, color TEXT, role TEXT)\n"
        "   color values: blanca, tinta\n"
        "   role values: principal, secundaria, autorizada\n\n"
        "3. yields(dop TEXT, gi_type TEXT, wine_type TEXT, variety TEXT, "
        "kg_ha REAL, hl_ha REAL)\n\n"
        "4. municipalities(dop TEXT, gi_type TEXT, municipality TEXT, "
        "zone TEXT, province TEXT)\n"
    )


# ---- Tool 5: search_graph_db ----

def search_graph_db(query: str) -> str:
    """Query the knowledge graph for entity relationships.

    The graph contains nodes (DOPs, varieties, municipalities, zones,
    soil types, climate data, parameters) connected by edges representing
    relationships (AUTHORIZES_VARIETY, CONTAINS_MUNICIPALITY, HAS_SOIL,
    HAS_CLIMATE, DEFINES_AGING, etc.).

    Use this for questions about:
    - Indirect relationships (e.g., which varieties grow in a climate zone)
    - Multi-hop connections (e.g., municipalities → zone → soil → wine style)
    - Cross-DOP comparisons via shared entities
    - Discovering connections between varieties, zones, and DOPs

    Args:
        query: The user's original question (used to extract entity names).

    Returns:
        Natural language description of the relevant subgraph.
    """
    from winegpt.graph_rag import query_graph

    result = query_graph(query)
    if not result:
        return "No relevant relationships found in the knowledge graph."
    return result
