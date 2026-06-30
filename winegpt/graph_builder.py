"""enotropos — Knowledge graph builder.

Extracts entities and relationships from:

1. ``data/tables.db`` (structured data — direct import)
2. Enriched markdown files (via LLM triple extraction)

Outputs ``data/graph.pkl`` (pickled NetworkX graph). Auto-rebuilds when stale
(older than ``tables.db`` or markdown sources) or when the on-disk schema
version does not match ``GRAPH_SCHEMA_VERSION``.

This module lives in the library package so that ``winegpt.graph_rag`` can
import it without depending on ``scripts`` (which would invert the layering).
``scripts/build_graph.py`` is a thin CLI wrapper around ``build_graph``.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import sqlite3
import time
from typing import Any

import networkx as nx

from winegpt.config import (
    EXTRACTED_DIR,
    GRAPH_PATH,
    GRAPH_SCHEMA_VERSION,
    LLM_MODEL,
    SUPPORTED_COUNTRIES,
    TABLES_DB_PATH,
)
from winegpt.schema import parse_folder_name

logger = logging.getLogger(__name__)

SECTION_MAX_CHARS = 4000  # Max chars per markdown section sent to LLM

# Entity type vocabulary used in the triple-extraction prompt.
_TRIPLE_TYPES = (
    "DOP|IGP|Variety|Municipality|Zone|SoilType|ClimateData|"
    "WineType|AgingCategory|Parameter|ProductionMethod"
)


# ---- Triple extraction prompt ----

TRIPLE_EXTRACTION_PROMPT = """Ets un extractor de coneixement per a DOP/IGP de vins europeus.
Extreu TOTES les relacions entre entitats del text següent.

Retorna EXCLUSIVAMENT un array JSON de tripletes.
No afegeixis explicacions, markdown ni text addicional.
Cada tripleta ha de tenir aquest format:
{{
  "subject": "Nom Entitat",
  "subject_type": "{types}",
  "predicate": "RELACIÓ",
  "object": "Nom Entitat",
  "object_type": "{types}",
  "attrs": {{"clau": "valor"}}
}}

EXEMPLE d'array vàlid:
[
  {{
    "subject": "Rioja", "subject_type": "DOP",
    "predicate": "AUTHORIZES_VARIETY",
    "object": "Tempranillo", "object_type": "Variety",
    "attrs": {{"color": "tinta", "role": "principal"}}
  }},
  {{
    "subject": "Rioja", "subject_type": "DOP",
    "predicate": "CONTAINS_MUNICIPALITY",
    "object": "Haro", "object_type": "Municipality",
    "attrs": {{}}
  }},
  {{
    "subject": "Haro", "subject_type": "Municipality",
    "predicate": "MUNICIPALITY_IN_ZONE",
    "object": "Rioja Alta", "object_type": "Zone",
    "attrs": {{}}
  }},
  {{
    "subject": "Haro", "subject_type": "Municipality",
    "predicate": "MUNICIPALITY_IN_PROVINCE",
    "object": "La Rioja", "object_type": "Province",
    "attrs": {{}}
  }}
]

RELACIONS a extreure (usa exactament aquests valors de "predicate"):

Varietats i raïm:
- AUTHORIZES_VARIETY: Una DOP o IGP autoritza una varietat de raïm.
  Incloure color ("tinta"/"blanca") i role ("principal"/"secundaria"/"autorizada") a attrs.
- VARIETY_HAS_COLOR: Una varietat té un color (tinta o blanca).
- COAUTHORIZED_WITH: (opcional) Dues varietats autoritzades a la mateixa DOP.

Municipis i geografia:
- CONTAINS_MUNICIPALITY: Una DOP conté un municipi dins la seva zona de producció.
- MUNICIPALITY_IN_ZONE: Un municipi pertany a una subzona (ex: "Rioja Alta", "Rioja Alavesa").
- MUNICIPALITY_IN_PROVINCE: Un municipi està en una província (ex: "La Rioja", "Burgos").

Paràmetres analítics:
- REQUIRES_PARAMETER: Una DOP requereix un paràmetre analític
  (grau alcohòlic, acidesa, SO2, sucres, etc.).
  Incloure value (número), unit, wine_type,
  relation ("min"/"max"/"range") a attrs.

Rendiments:
- REGULATES_YIELD: Una DOP regula el rendiment per a una varietat o tipus de vi.
  Incloure kg_ha i/o hl_ha a attrs. Ex: {{"kg_ha": 6500, "hl_ha": 45.5}}.

Envelliment:
- DEFINES_AGING: Una DOP defineix una categoria d'envelliment
  (Crianza, Reserva, Gran Reserva, etc.).
  Incloure months_barrica, months_total a attrs si estan disponibles.
- AGING_APPLIES_TO: Una categoria d'envelliment s'aplica a un tipus de vi.

Sòl i clima:
- HAS_SOIL: Una DOP o zona té un tipus de sòl (argilós, calcari, etc.).
  Incloure description a attrs.
- HAS_CLIMATE: Una DOP o zona té característiques climàtiques.
  Incloure altitude (m), rainfall_mm, temperature_c a attrs si estan disponibles.

Tipus de vi:
- PRODUCES_WINE_TYPE: Una DOP produeix un tipus de vi
  (tinto, blanco, rosado, espumoso, generoso, licor, etc.).
- AGING_APPLIES_TO: Una categoria d'envelliment s'aplica a un tipus de vi.
- PARAMETER_MEASURED_IN: Un paràmetre es mesura en una unitat concreta.

NORMES IMPORTANTS:
- Normalitza els noms: "Tempranillo" (no "TEMPRANILLO"), "La Rioja" (no "LA RIOJA").
- Si una entitat ja apareix al text o a altres tripletes, usa el mateix nom exacte.
- Extreu TOTES les varietats, municipis, paràmetres, zones i categories d'envelliment esmentades.
- No inventis dades: només extreu el que està explícit al text.
- L'array JSON ha de ser vàlid i parsejable. No afegeixis comes finals ni camps undefined.
- Si no hi ha cap relació al text, retorna EXACTAMENT [].
- Cada tripleta ha de tenir subject_type i object_type vàlids de la llista proporcionada.
- Si no saps el tipus d'una entitat, usa "Entity".

Text del document (secció):
```
{text}
```

Array JSON de tripletes:"""


# ---- Graph builder ----

def _load_tables_db(graph: nx.MultiDiGraph) -> int:
    """Import structured data from tables.db into the graph. Returns count of edges added."""
    if not TABLES_DB_PATH.exists():
        logger.warning("tables.db not found, skipping")
        return 0

    conn = sqlite3.connect(str(TABLES_DB_PATH))
    conn.row_factory = sqlite3.Row
    count = 0

    # Analytics: DOP → REQUIRES_PARAMETER
    for row in conn.execute(
        "SELECT DISTINCT dop, gi_type, wine_type, parameter, "
        "value, unit, relation FROM analytics WHERE value IS NOT NULL"
    ):
        dop_node = row["dop"]
        param_name = f"{row['parameter']} ({row['wine_type']})"
        graph.add_node(dop_node, type=row["gi_type"], source="tables_db")
        graph.add_node(param_name, type="Parameter", source="tables_db")
        graph.add_edge(dop_node, param_name,
                       predicate="REQUIRES_PARAMETER",
                       value=row["value"], unit=row["unit"],
                       relation=row["relation"], wine_type=row["wine_type"])
        count += 1

    # Varieties: DOP → AUTHORIZES_VARIETY
    for row in conn.execute("SELECT DISTINCT dop, gi_type, variety, color, role FROM varieties"):
        dop_node = row["dop"]
        var_node = row["variety"]
        graph.add_node(dop_node, type=row["gi_type"], source="tables_db")
        graph.add_node(var_node, type="Variety", source="tables_db",
                       color=row["color"], role=row["role"])
        graph.add_edge(dop_node, var_node,
                       predicate="AUTHORIZES_VARIETY",
                       color=row["color"], role=row["role"])
        if row["color"] and row["color"] != "unknown":
            color_node = f"color_{row['color']}"
            graph.add_node(color_node, type="Color", source="tables_db")
            graph.add_edge(var_node, color_node, predicate="VARIETY_HAS_COLOR")
        count += 1

    # Municipalities: DOP → CONTAINS_MUNICIPALITY
    for row in conn.execute(
        "SELECT DISTINCT dop, gi_type, municipality, zone, province "
        "FROM municipalities"
    ):
        dop_node = row["dop"]
        muni_node = row["municipality"]
        graph.add_node(dop_node, type=row["gi_type"], source="tables_db")
        graph.add_node(muni_node, type="Municipality", source="tables_db")
        graph.add_edge(dop_node, muni_node, predicate="CONTAINS_MUNICIPALITY")
        if row["zone"]:
            zone_node = row["zone"]
            graph.add_node(zone_node, type="Zone", source="tables_db")
            graph.add_edge(muni_node, zone_node, predicate="MUNICIPALITY_IN_ZONE")
        if row["province"]:
            prov_node = row["province"]
            graph.add_node(prov_node, type="Province", source="tables_db")
            graph.add_edge(muni_node, prov_node, predicate="MUNICIPALITY_IN_PROVINCE")
        count += 1

    # Yields: DOP → REGULATES_YIELD
    for row in conn.execute(
        "SELECT DISTINCT dop, gi_type, wine_type, variety, kg_ha, hl_ha "
        "FROM yields WHERE kg_ha IS NOT NULL OR hl_ha IS NOT NULL"
    ):
        dop_node = row["dop"]
        yield_node = f"yield_{row['dop']}_{row['variety'] or 'general'}"
        graph.add_node(dop_node, type=row["gi_type"], source="tables_db")
        graph.add_node(yield_node, type="YieldNorm", source="tables_db",
                       wine_type=row["wine_type"])
        graph.add_edge(dop_node, yield_node,
                       predicate="REGULATES_YIELD",
                       kg_ha=row["kg_ha"], hl_ha=row["hl_ha"],
                       wine_type=row["wine_type"])
        count += 1

    # Wine types: DOP → PRODUCES_WINE_TYPE (from analytics)
    for row in conn.execute(
        "SELECT DISTINCT dop, gi_type, wine_type FROM analytics "
        "WHERE wine_type IS NOT NULL AND wine_type != 'general'"
    ):
        dop_node = row["dop"]
        wine_node = f"wine_{row['wine_type']}"
        graph.add_node(dop_node, type=row["gi_type"], source="tables_db")
        graph.add_node(wine_node, type="WineType", source="tables_db")
        if not graph.has_edge(dop_node, wine_node):
            graph.add_edge(dop_node, wine_node,
                           predicate="PRODUCES_WINE_TYPE",
                           wine_type=row["wine_type"])
            count += 1

    # Province nodes: ensure they exist for municipality links
    for row in conn.execute(
        "SELECT DISTINCT province FROM municipalities WHERE province IS NOT NULL"
    ):
        prov_node = row["province"]
        if prov_node not in graph:
            graph.add_node(prov_node, type="Province", source="tables_db")

    conn.close()
    logger.info("Imported %d edges from tables.db", count)
    return count


def _extract_triples_with_llm(text: str, doc_name: str) -> list[dict[str, Any]]:
    """Use LLM to extract triples from a text section."""
    from winegpt.llm import get_llm_client

    if not text or len(text) < 100:
        return []

    client = get_llm_client()
    truncated = text[:SECTION_MAX_CHARS]

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": TRIPLE_EXTRACTION_PROMPT.format(
                    text=truncated,
                    types=_TRIPLE_TYPES,
                )},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        result = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("LLM extraction failed for %s: %s", doc_name, e)
        return []

    triples = _parse_triples_json(result, doc_name)
    if triples:
        logger.info("    → %d triples from %s (%d chars)", len(triples), doc_name, len(truncated))
    return triples


def _parse_triples_json(raw: str, doc_name: str) -> list[dict[str, Any]]:
    """Parse triples JSON from LLM response, handling common formatting issues."""
    if not raw or not raw.strip():
        return []

    # Strip markdown code fences that some models wrap around the JSON
    cleaned_raw = raw.strip()
    cleaned_raw = re.sub(r"^```(?:json)?\s*", "", cleaned_raw)
    cleaned_raw = re.sub(r"\s*```$", "", cleaned_raw)

    # Find JSON array in the response (greedy match across lines)
    json_match = re.search(r"\[.*\]", cleaned_raw, re.DOTALL)
    if not json_match:
        logger.debug("No JSON array found in LLM response for %s (len=%d)", doc_name, len(raw))
        return []

    json_str = json_match.group(0)

    # Attempt 1: direct parse
    try:
        triples = json.loads(json_str)
        if isinstance(triples, list):
            return triples
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix common JSON issues
    cleaned = json_str
    # Remove trailing commas before ] or }
    cleaned = re.sub(r",\s*(\]|\})", r"\1", cleaned)
    # Fix single quotes → double quotes (but only outside string values)
    # Fix missing quotes around keys
    cleaned = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', cleaned)
    # Remove JavaScript-style undefined / NaN / Infinity
    cleaned = re.sub(r":\s*(?:undefined|NaN|Infinity|null)\b", ': ""', cleaned)
    # Fix unescaped control characters
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    try:
        triples = json.loads(cleaned)
        if isinstance(triples, list):
            return triples
    except json.JSONDecodeError:
        pass

    # Attempt 3: try to parse each line as a separate JSON object and collect into array
    try:
        objects: list[dict[str, Any]] = []
        for line in cleaned_raw.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                line_fixed = re.sub(r",\s*}", "}", line)
                line_fixed = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
                                    r'\1"\2":', line_fixed)
                line_fixed = re.sub(r":\s*(?:undefined|NaN|Infinity|null)\b", ': ""', line_fixed)
                try:
                    obj = json.loads(line_fixed)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    continue
        if objects:
            return objects
    except Exception:
        pass

    logger.debug("Could not parse triples JSON for %s (len=%d)", doc_name, len(raw))
    return []


def _normalize_name(name: str) -> str:
    """Normalize entity names for consistent node identity."""
    name = name.strip().title()
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    return name


def _add_triples_to_graph(
    graph: nx.MultiDiGraph, triples: list[dict[str, Any]], source: str,
) -> int:
    """Add extracted triples to the graph. Returns count of edges added."""
    count = 0
    for t in triples:
        try:
            subj = _normalize_name(str(t.get("subject", "")))
            obj = _normalize_name(str(t.get("object", "")))
            pred = t.get("predicate", "")
            subj_type = t.get("subject_type", "Entity")
            obj_type = t.get("object_type", "Entity")
            attrs = t.get("attrs", {}) or {}

            if not subj or not obj or not pred:
                continue

            # Add nodes
            if subj not in graph:
                graph.add_node(subj, type=subj_type, source=source)
            if obj not in graph:
                graph.add_node(obj, type=obj_type, source=source)

            # Add edge with attributes
            edge_attrs: dict[str, Any] = {"predicate": pred, "source": source}
            if isinstance(attrs, dict):
                edge_attrs.update({k: v for k, v in attrs.items() if v is not None})

            graph.add_edge(subj, obj, **edge_attrs)
            count += 1

            # Cross-relationships: infer reverse connections
            _add_cross_relationships(graph, subj, subj_type, obj, obj_type, pred, attrs)

        except Exception as e:
            logger.debug("Error adding triple: %s", e)

    return count


def _add_cross_relationships(
    graph: nx.MultiDiGraph,
    subj: str, subj_type: str,
    obj: str, obj_type: str,
    pred: str, attrs: dict[str, Any],
) -> None:
    """Infer cross-relationships from direct ones.

    Examples:
    - If Variety X is authorized in DOP A and DOP B, connect Variety X → DOP B
    - If DOP A has climate C and DOP B has climate C, connect DOP A <-> DOP B via climate
    - Connect varieties that share the same DOP
    """
    # If two varieties are authorized in the same DOP, connect them
    if pred == "AUTHORIZES_VARIETY":
        for _, target, edge_data in graph.out_edges(subj, data=True):
            if edge_data.get("predicate") == "AUTHORIZES_VARIETY" and target != obj:
                if not graph.has_edge(obj, target):
                    graph.add_edge(obj, target,
                                   predicate="COAUTHORIZED_WITH",
                                   via=subj, source="cross_inference")
                if not graph.has_edge(target, obj):
                    graph.add_edge(target, obj,
                                   predicate="COAUTHORIZED_WITH",
                                   via=subj, source="cross_inference")

    # If a zone has a soil type and a climate, link soil → climate
    if pred == "MUNICIPALITY_IN_ZONE":
        zone_node = obj
        # Find soils and climates for this zone
        soils = []
        climates = []
        for _, neighbor, edge_data in graph.edges(zone_node, data=True):
            if edge_data.get("predicate") == "HAS_SOIL":
                soils.append(neighbor)
            elif edge_data.get("predicate") == "HAS_CLIMATE":
                climates.append(neighbor)
        for s in soils:
            for c in climates:
                if not graph.has_edge(s, c):
                    graph.add_edge(s, c, predicate="SOIL_IN_CLIMATE",
                                   via=zone_node, source="cross_inference")


def _infer_shared_entity_relationships(graph: nx.MultiDiGraph) -> int:
    """Infer high-level relationships from already-loaded entities.

    Runs after both tables.db import and LLM extraction so the complete entity
    pool is available. Adds:

    * SHARES_VARIETY_WITH — connect DOPs that authorize the same variety
    * ZONE_IN_DOP — reverse chain from MUNICIPALITY_IN_ZONE
    * PROVINCE_IN_DOP — reverse chain from MUNICIPALITY_IN_PROVINCE
    """
    count = 0

    # Collect variety → DOPs mapping
    variety_dops: dict[str, set[str]] = {}
    for u, v, data in graph.edges(data=True):
        if data.get("predicate") == "AUTHORIZES_VARIETY":
            variety_dops.setdefault(v, set()).add(u)

    for variety, dops in variety_dops.items():
        if len(dops) < 2:
            continue
        for dop_a in dops:
            for dop_b in dops:
                if dop_a >= dop_b:
                    continue
                if not graph.has_edge(dop_a, dop_b):
                    graph.add_edge(dop_a, dop_b,
                                   predicate="SHARES_VARIETY_WITH",
                                   variety=variety, source="cross_inference")
                    count += 1

    # Collect zone → DOPs (via municipality chain)
    zone_dops: dict[str, set[str]] = {}
    for u, v, data in graph.edges(data=True):
        if data.get("predicate") == "MUNICIPALITY_IN_ZONE":
            zone = v
            for _, dop_node, e_data in graph.in_edges(u, data=True):
                if e_data.get("predicate") == "CONTAINS_MUNICIPALITY":
                    zone_dops.setdefault(zone, set()).add(dop_node)

    for zone, dops in zone_dops.items():
        for dop_node in dops:
            if not graph.has_edge(zone, dop_node):
                graph.add_edge(zone, dop_node,
                               predicate="ZONE_IN_DOP",
                               source="cross_inference")
                count += 1

    # Collect province → DOPs (via municipality chain)
    province_dops: dict[str, set[str]] = {}
    for u, v, data in graph.edges(data=True):
        if data.get("predicate") == "MUNICIPALITY_IN_PROVINCE":
            province = v
            for _, dop_node, e_data in graph.in_edges(u, data=True):
                if e_data.get("predicate") == "CONTAINS_MUNICIPALITY":
                    province_dops.setdefault(province, set()).add(dop_node)

    for province, dops in province_dops.items():
        for dop_node in dops:
            if not graph.has_edge(province, dop_node):
                graph.add_edge(province, dop_node,
                               predicate="PROVINCE_CONTAINS_DOP",
                               source="cross_inference")
                count += 1

    if count:
        logger.info("Inferred %d shared-entity relationships", count)
    return count


def _get_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into sections by headings. Returns [(heading, content), ...].

    A leading newline is prepended so a heading at the very start of the document
    is also split out (otherwise its content would be lumped into the heading
    part and dropped).
    """
    sections: list[tuple[str, str]] = []
    # Split by ## or ### headings
    parts = re.split(r"\n(#{2,3}\s+.+?)\n", "\n" + text)
    current_heading = "Document"
    for part in parts:
        if re.match(r"^#{2,3}\s+", part):
            current_heading = part.strip("#").strip()
        else:
            content = part.strip()
            if content and len(content) > 80:
                sections.append((current_heading, content))
    return sections


# Headings worth sending to the LLM for triple extraction.
# Expanded to cover most section types found in EU wine product specifications
# (both Spanish and Catalan). Covers: varieties, geography, analytics, yields,
# aging, soil/climate, wine types, production, quality, regulations, governance,
# viticulture, enology, and structural sections that contain wine data.
_SECTION_KEYWORDS = (
    # Varieties
    "variedad", "varietat", "variedades", "varietats", "cepa", "uva", "raïm",
    # Municipalities & geography
    "municipio", "municipi", "municipios", "municipis",
    "zona", "subzona", "zonas", "subzonas", "comarca",
    "geográf", "geograf", "localidad", "localitat",
    # Analytics & parameters
    "grado", "alcohol", "graduación", "graduació",
    "acidez", "acidesa", "ácido", "acid",
    "azúcar", "azucar", "azúcares", "azucares",
    "sulfuroso", "anhídrido", "anhidrido", "so2",
    "ph", "parámetro", "parametre",
    # Yield & production
    "rendimiento", "rendiment", "producción", "producció",
    "kg/ha", "hl/ha",
    # Aging
    "crianza", "reserva", "envejecimiento", "envelliment",
    "añada", "anyada",
    # Soil & climate
    "suelo", "sòl", "clima", "climat", "edafolog",
    "altitud", "precipitación", "pluviom", "temperatura",
    # Wine types
    "tipo", "tipus", "categoria", "clase",
    "blanco", "blanc", "tinto", "negre", "rosado", "rosat",
    "espumoso", "generoso", "licor",
    # Production & viticulture
    "elabor", "producc", "práctica", "practica",
    "vitic", "cultiu", "cultivo", "factores", "factors",
    "vendimia", "verema",
    # Quality & characteristics
    "calidad", "qualitat", "característ", "caracterist",
    "organol", "sensorial", "analíti", "analiti",
    "descripci", "descript",
    # Regulations
    "norma", "reglament", "plec", "pliego", "condicion", "condició",
    "denominació", "denominación", "ampar",
    "nombre", "protegi", "proteg",  # "nombre protegido", "protected name"
    # Labeling & marketing
    "etiquet", "presentació", "presentación",
    # Governance & board
    "consejo", "consell",
    # Enology & wine-making
    "vinific", "ferment", "macera",
    "crian", "envejec", "barrica", "botella",
    "prens", "prensa", "filtra",
    # General wine / product indicators (catch-all for sections with data)
    "vino", "vinos", "vi", "vins", "producto", "producte",
    # Yield / planting density
    "densidad", "densitat", "plantación", "plantacio",
    "marco", "marc", "riego", "regadiu", "reg",
    "vendim", "verem",
)


def needs_rebuild() -> bool:
    """Check if graph.pkl is older than sources and needs rebuilding."""
    if not GRAPH_PATH.exists():
        return True

    graph_mtime = GRAPH_PATH.stat().st_mtime

    # Check tables.db
    if TABLES_DB_PATH.exists() and TABLES_DB_PATH.stat().st_mtime > graph_mtime:
        logger.info("tables.db is newer than graph.pkl, rebuild needed")
        return True

    # Check extracted markdowns
    if EXTRACTED_DIR.exists():
        for md_path in EXTRACTED_DIR.rglob("*.md"):
            if md_path.stat().st_mtime > graph_mtime:
                logger.info("Markdown %s is newer than graph.pkl, rebuild needed", md_path.name)
                return True

    return False


def _load_cached() -> nx.MultiDiGraph | None:
    """Load and validate the cached graph.

    Returns ``None`` if the file is missing, unreadable, or stamped with an
    incompatible schema version (so the caller rebuilds from scratch).
    """
    if not GRAPH_PATH.exists():
        return None
    try:
        with open(GRAPH_PATH, "rb") as f:
            graph = pickle.load(f)  # noqa: S301
    except Exception as e:
        logger.warning("Failed to load cached graph: %s", e)
        return None
    cached_version = graph.graph.get("schema_version")
    if cached_version != GRAPH_SCHEMA_VERSION:
        logger.info(
            "Graph schema version mismatch (cached=%r, current=%d), rebuilding",
            cached_version,
            GRAPH_SCHEMA_VERSION,
        )
        return None
    return graph


def build_graph(
    force: bool = False,
    use_llm: bool = True,
    countries: tuple[str, ...] | None = None,
    gi_filter: str | None = None,
) -> nx.MultiDiGraph:
    """Build the knowledge graph. Rebuilds if stale or forced.

    Args:
        force: Rebuild even if the cache looks fresh.
        use_llm: Run LLM triple extraction over markdown sections (expensive:
            ~600 API calls, ~30 min). Disable for a fast tables+directory-only
            build.
        countries: Override the country list (defaults to
            ``config.SUPPORTED_COUNTRIES``).
        gi_filter: When set, only extract triples from the specified GI
            (e.g. ``"Rioja"``). The existing graph is loaded and enriched
            incrementally — no full rebuild. Forces ``use_llm=True``.

    Returns the NetworkX graph.
    """
    if countries is None:
        countries = SUPPORTED_COUNTRIES

    # Incremental mode: start from cached graph
    if gi_filter and not force:
        cached = _load_cached()
        if cached is not None:
            graph = cached
            logger.info("Loading graph from cache (%d nodes, %d edges) for incremental build",
                        graph.number_of_nodes(), graph.number_of_edges())
        else:
            logger.info("No cached graph found, building fresh for gi_filter=%s", gi_filter)
            graph = nx.MultiDiGraph()
            graph.graph["schema_version"] = GRAPH_SCHEMA_VERSION
            _load_tables_db(graph)
    elif not force and not needs_rebuild():
        cached = _load_cached()
        if cached is not None:
            logger.info("Loading graph from cache (%d bytes)", GRAPH_PATH.stat().st_size)
            return cached
        graph = nx.MultiDiGraph()
        graph.graph["schema_version"] = GRAPH_SCHEMA_VERSION
        graph.graph["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        db_edges = _load_tables_db(graph)
    else:
        logger.info("Building knowledge graph...")
        graph = nx.MultiDiGraph()
        graph.graph["schema_version"] = GRAPH_SCHEMA_VERSION
        graph.graph["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        db_edges = _load_tables_db(graph)

    # Step 2: Add DOP/IGP nodes from directory structure (without LLM)
    for country in countries:
        country_dir = EXTRACTED_DIR / country
        if not country_dir.exists():
            continue

        for gi_dir in sorted(country_dir.iterdir()):
            if not gi_dir.is_dir():
                continue
            info = parse_folder_name(gi_dir.name)
            display = info.display_name

            if display not in graph:
                graph.add_node(display, type=info.gi_type, source="directory")
            elif graph.nodes[display].get("type", "") != info.gi_type:
                graph.nodes[display]["type"] = info.gi_type

    # Step 3: LLM extraction — extract triples from markdown sections
    llm_edges = 0
    if use_llm:
        for country in countries:
            country_dir = EXTRACTED_DIR / country
            if not country_dir.exists():
                continue
            md_files = sorted(country_dir.rglob("*.md"))
            if gi_filter:
                md_files = [
                    p for p in md_files
                    if parse_folder_name(p.parent.name).gi_name.lower() == gi_filter.lower()
                ]
                if not md_files:
                    logger.warning("No markdown files found for gi_filter=%s", gi_filter)
            logger.info("Processing %d markdown files in %s", len(md_files), country)
            for i, md_path in enumerate(md_files):
                doc_name = md_path.parent.name
                logger.info("  [%d/%d] %s", i + 1, len(md_files), doc_name)
                try:
                    text = md_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    logger.warning("Could not read %s: %s", md_path, e)
                    continue
                for heading, content in _get_markdown_sections(text):
                    # Match if heading OR first 500 chars of content contain keywords
                    if not (
                        any(w in heading.lower() for w in _SECTION_KEYWORDS)
                        or any(w in content[:500].lower() for w in _SECTION_KEYWORDS)
                    ):
                        continue
                    doc_label = f"{doc_name}/{heading[:40]}"
                    triples = _extract_triples_with_llm(content, doc_label)
                    added = _add_triples_to_graph(graph, triples, source=f"llm:{doc_label}")
                    llm_edges += added

    # Step 4: Infer cross-entity relationships from the complete graph
    cross_edges = _infer_shared_entity_relationships(graph)

    # db_edges is set by the full-build branches; default to 0 for incremental/cached.
    try:
        _ = db_edges
    except NameError:
        db_edges = 0

    logger.info(
        "Graph built: %d nodes, %d edges (tables_db=%d, llm=%d, cross_inference=%d)",
        graph.number_of_nodes(), graph.number_of_edges(), db_edges, llm_edges, cross_edges,
    )

    # Save
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)

    logger.info("Graph saved to %s", GRAPH_PATH)
    return graph


def load_graph() -> nx.MultiDiGraph:
    """Load the knowledge graph, rebuilding if needed."""
    return build_graph(force=False)
