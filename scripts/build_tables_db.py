"""Build a SQLite database from enriched Markdown tables.

Parses ``## Taules extretes amb IA`` sections in extracted Markdown files
and populates ``data/tables.db`` with structured wine data (analytical
parameters, varieties, yields, municipalities).
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from typing import Any

from winegpt.config import EXTRACTED_DIR, SUPPORTED_COUNTRIES, TABLES_DB_PATH
from winegpt.schema import parse_folder_name
from winegpt.table_extractor import TABLES_SECTION_HEADER, TABLES_SUBSECTION_PREFIX

logger = logging.getLogger(__name__)

# ``TABLES_DB_PATH`` / ``EXTRACTED_DIR`` / ``SUPPORTED_COUNTRIES`` come from
# config so they no longer drift from the rest of the project.

# ---- Table classifiers ----

ANALYTICS_PARAM_MAP: dict[str, tuple[str, str]] = {
    # (parameter_key, unit) — lowercased fuzzy match
    "grado alcohólico": ("grado_alcoholico", "% Vol"),
    "alcohol adquirido": ("grado_alcoholico", "% Vol"),
    "acidez volátil": ("acidez_volatil", "g/l"),
    "acidez total": ("acidez_total", "g/l"),
    "anhídrido sulfuroso": ("so2_total", "mg/l"),
    "sulfuroso total": ("so2_total", "mg/l"),
    "azúcares residuales": ("azucares_residuales", "g/l"),
    "azúcares totales": ("azucares_totales", "g/l"),
}

# Columns that indicate a varieties table (must be header-level, not row values)
VARIETY_COLUMN_HINTS = {
    "variedad", "variedade", "varietat", "cepa", "uva",
    "variedades", "varietats", "raïm", "raim",
    "varietats blanques", "varietats negres", "varietats tintes",
}

# Words that should NOT be treated as variety names
VARIETY_REJECT_WORDS = {
    "blancos", "blancas", "blancs", "blanques", "blanco", "blanca",
    "tintos", "tintas", "tints", "tintes", "tinto", "tinta", "negres",
    "rosados", "rosadas", "rosats", "rosat",
    "grado", "alcohol", "acidez", "azucares", "azúcares", "sulfuroso",
    "anhídrido", "dióxido", "azufre", "máximo", "mínimo", "màxim", "mínim",
    "total", "volátil", "mín", "máx", "min", "max",
    "categoría", "categoria", "parámetro", "parametre", "característica",
    "tipo", "tipus", "clase",
    # Organoleptic table row keywords
    "aspecto", "aroma", "sabor", "limpio", "brillante", "turbio",
    "afrutado", "frutal", "frutas", "ácido", "ligeramente", "suave", "fresco",
    "color amarillo", "color rojo", "color granate", "sin malos",
    "sin presentar", "notas", "envejecimiento", "olfativa", "gustativa",
    "fase visual", "fase olfativa", "fase gustativa", "visual",
    "intensidad", "calidad", "equilibrio", "estructura",
    "defectos", "barrica",
    # Non-variety table rows
    "conservación", "temperatura", "humedad", "capacidad",
}

# Organoleptic / sensory keywords that mean the whole table should be skipped
ORGANOLEPTIC_TABLE_KEYWORDS = {
    "aspecto", "aroma", "sabor", "fase visual", "fase olfativa",
    "fase gustativa", "limpio", "brillante", "turbio", "afrutado",
    "frutal", "ligeramente ácido", "tacto suave", "sin malos olores",
    "sin presentar aspecto", "notas de envejecimiento",
}

# Wine type keywords for column/row matching
WINE_TYPE_KEYWORDS = {
    "blanco": ["blanco", "blanc", "blanco", "blanca"],
    "rosado": ["rosado", "rosat"],
    "tinto": ["tinto", "negre", "tinta"],
    "espumoso": ["espumoso", "escumós", "cava"],
    "generoso": ["generoso", "generós", "licor", "dulce"],
}

# Columns that indicate a yield table
YIELD_COLUMN_HINTS = {
    "rendimiento", "rendiment", "kg/ha", "hl/ha", "producción",
    "producció", "kg_ha", "hl_ha",
}

# Columns that indicate a municipality table
MUNI_COLUMN_HINTS = {
    "municipio", "municipi", "provincia", "província", "zona",
    "parroquia", "pedanía", "pedania",
}


def _clean_cell(cell: str) -> str:
    return cell.strip().strip("*").strip("_").strip()


def _parse_markdown_table(table_text: str) -> list[dict[str, str]]:
    """Parse a Markdown table into list of {header: value} dicts."""
    lines = table_text.strip().split("\n")
    rows: list[list[str]] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [_clean_cell(c) for c in line.split("|")]
        # First and last elements are empty strings from leading/trailing |
        cells = [c for c in cells if c]
        if not cells:
            continue
        # Skip separator rows like |---|---|
        if all(re.fullmatch(r"[-:]+", c) for c in cells):
            continue
        rows.append(cells)

    if not rows:
        return []

    headers = rows[0]
    result: list[dict[str, str]] = []
    for row in rows[1:]:
        if len(row) >= len(headers):
            row_dict = {headers[i]: _clean_cell(row[i]) for i in range(len(headers))}
        else:
            row_dict = {}
            for i, val in enumerate(row):
                key = headers[i] if i < len(headers) else f"col_{i}"
                row_dict[key] = _clean_cell(val)
        result.append(row_dict)
    return result


def _classify_table(
    rows: list[dict[str, str]],
) -> str | None:
    """Classify a parsed table as 'analytics', 'varieties', 'yields', or None."""
    if not rows:
        return None

    # Use headers for classification (more reliable than row values)
    headers = list(rows[0].keys())
    headers_text = " ".join(h.lower() for h in headers)

    # Check all header+row text for deeper matching
    all_keys = {k.lower() for row in rows for k in row}
    all_text = headers_text + " " + " ".join(all_keys)

    # Reject organoleptic tables (sensory descriptions, not data)
    organo_count = sum(1 for kw in ORGANOLEPTIC_TABLE_KEYWORDS if kw in all_text)
    if organo_count >= 3:
        return None  # Skip organoleptic tables

    # Check yield indicators first (most specific)
    if any(h in headers_text for h in YIELD_COLUMN_HINTS):
        return "yields"

    # Check variety indicators — must have a header matching, not just row values
    if any(h in headers_text for h in VARIETY_COLUMN_HINTS):
        return "varieties"

    # Check municipality indicators
    if any(h in headers_text for h in MUNI_COLUMN_HINTS):
        return "municipalities"

    # Check analytics indicators (in headers or values)
    for hint in ANALYTICS_PARAM_MAP:
        if hint in all_text:
            return "analytics"

    # Fallback: if rows mention params, classify as analytics
    param_count = sum(1 for hint in ANALYTICS_PARAM_MAP if hint in all_text)
    if param_count >= 2:
        return "analytics"

    return None


def _extract_value_and_relation(raw: str) -> tuple[float | None, str | None, str | None]:
    """Extract numeric value, unit, and min/max relation from a raw cell.

    Returns (value, unit, relation). Handles European number format (10.000 → 10000).
    """
    value: float | None = None
    unit: str | None = None
    relation: str | None = None

    lower = raw.lower().strip()

    # Detect relation
    if "mínim" in lower or "minim" in lower or "mínimo" in lower or "mín" in lower:
        relation = "min"
    elif "màxim" in lower or "maxim" in lower or "máximo" in lower or "máx" in lower:
        relation = "max"

    # Try to extract number — handle European thousands separator (10.000 = 10000)
    # First try decimal number with comma: 11,5 → 11.5
    num_match = re.search(r"(\d+)[,](\d+)", raw)
    if num_match:
        value = float(f"{num_match.group(1)}.{num_match.group(2)}")
    else:
        # Then try integer — but be careful with 10.000 which is 10000 in EU notation
        # Strategy: if there are multiple dots, it's EU thousands separator
        dots = [m.start() for m in re.finditer(r"\d+\.\d+", raw)]
        if len(dots) >= 1:
            # EU thousands: 10.000 → 10000;  1.200 → 1200
            # But 10.5 could be either decimal or thousands...
            # Use heuristic: 3 digits after dot = thousands separator
            num_match = re.search(r"(\d+)\.(\d{3})(?!\d)", raw)
            if num_match:
                value = float(num_match.group(1) + num_match.group(2))
            else:
                # Try simple decimal: 10.5
                num_match = re.search(r"(\d+)\.(\d+)", raw)
                if num_match:
                    value = float(f"{num_match.group(1)}.{num_match.group(2)}")
        else:
            num_match = re.search(r"(\d+)", raw)
            if num_match:
                value = float(num_match.group(1))

    # Detect unit
    if "%" in raw or "vol" in lower:
        unit = "% Vol"
    elif "g/l" in lower or "gr/l" in lower or "gramos" in lower:
        unit = "g/l"
    elif "mg/l" in lower:
        unit = "mg/l"
    elif "kg/ha" in lower:
        unit = "kg/ha"
    elif "hl/ha" in lower:
        unit = "hl/ha"
    elif "kg" in lower:
        unit = "kg/ha" if "ha" in lower or "hect" in lower else "kg"
    elif "hl" in lower:
        unit = "hl/ha" if "ha" in lower or "hect" in lower else "hl"
    elif any(w in raw.lower() for w in ("g/", "gram", "gr.",)):
        unit = "g/l"

    return value, unit, relation


def _find_param_key(text: str) -> str | None:
    """Match text against ANALYTICS_PARAM_MAP to find parameter key."""
    lower = text.lower()
    for hint, (key, default_unit) in ANALYTICS_PARAM_MAP.items():
        if hint in lower:
            return key
    return None


def _extract_analytics_rows(
    dop: str,
    gi_type: str,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Extract analytics rows from a parsed table."""
    results: list[dict[str, Any]] = []

    # Strategy: look at headers and rows to find parameter-value pairs
    all_headers = list(rows[0].keys()) if rows else []

    for row in rows:
        # Try to find parameter column
        param_col = None
        for h in all_headers:
            hl = h.lower()
            if any(w in hl for w in ("parámetro", "parametre", "característica",
                                      "caracteristica", "paràmetre", "categoria")):
                param_col = h
                break
        # Fallback: first column is often the parameter
        if param_col is None and all_headers:
            param_col = all_headers[0]

        # Try to find value column
        value_cols = []
        for h in all_headers:
            hl = h.lower()
            if any(w in hl for w in ("valor", "value", "unidad", "%", "g/l", "mg/l")):
                value_cols.append(h)

        # If only 2 columns and one looks like value
        if len(all_headers) == 2:
            param_col = all_headers[0]
            value_cols = [all_headers[1]]

        # For structure D (Característica | Valor | Unidad)
        if param_col and value_cols:
            param_cell = row.get(param_col, "")
            for vc in value_cols:
                val_cell = row.get(vc, "")
                param_key = _find_param_key(param_cell)
                if not param_key:
                    param_key = _find_param_key(vc)  # Try the header too
                if param_key:
                    value, unit, relation = _extract_value_and_relation(val_cell)
                    # Default unit from ANALYTICS_PARAM_MAP if none detected
                    if unit is None:
                        for hint_key, (pk, default_u) in ANALYTICS_PARAM_MAP.items():
                            if pk == param_key:
                                unit = default_u
                                break
                    wine_type = _infer_wine_type(row, all_headers)
                    results.append({
                        "dop": dop,
                        "gi_type": gi_type,
                        "wine_type": wine_type,
                        "parameter": param_key,
                        "value": value,
                        "unit": unit or "",
                        "relation": relation or "target",
                    })
        # For structure like | Vino Blanco | Vino Tinto |
        else:
            for h in all_headers:
                param_key = _find_param_key(row.get(h, ""))
                if param_key:
                    value, unit, relation = _extract_value_and_relation(row.get(h, ""))
                    if unit is None:
                        for hint_key, (pk, default_u) in ANALYTICS_PARAM_MAP.items():
                            if pk == param_key:
                                unit = default_u
                                break
                    results.append({
                        "dop": dop,
                        "gi_type": gi_type,
                        "wine_type": h,
                        "parameter": param_key,
                        "value": value,
                        "unit": unit or default_u,
                        "relation": relation or "target",
                    })

    return results


def _infer_wine_type(row: dict[str, str], headers: list[str]) -> str:
    """Infer wine type from row/header context."""
    all_text = " ".join(row.values()).lower() + " " + " ".join(headers).lower()
    types = []
    for wine_type, keywords in WINE_TYPE_KEYWORDS.items():
        if any(kw in all_text for kw in keywords):
            types.append(wine_type)
    return "/".join(types) if types else "general"


def _infer_wine_type_from_text(text: str) -> str:
    """Infer wine type from a free text string."""
    lower = text.lower()
    types = []
    for wine_type, keywords in WINE_TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            types.append(wine_type)
    return "/".join(types) if types else "general"


def _extract_varieties_rows(
    dop: str,
    gi_type: str,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Extract variety rows from a parsed table."""
    results: list[dict[str, Any]] = []
    all_headers = list(rows[0].keys()) if rows else []

    # Find variety name column
    variety_col = None
    for h in all_headers:
        hl = h.lower()
        if any(w in hl for w in VARIETY_COLUMN_HINTS):
            variety_col = h
            break

    # Also look for multi-column variety tables (Blancas | Tintas)
    blanc_col = None
    tinto_col = None
    for h in all_headers:
        hl = h.lower()
        if "blanc" in hl:
            blanc_col = h
        if "tint" in hl or "negre" in hl:
            tinto_col = h

    for row in rows:
        # Case 1: Multi-column (Blancas | Tintas)
        if blanc_col and tinto_col:
            for col, color in [(blanc_col, "blanca"), (tinto_col, "tinta")]:
                names = [n.strip() for n in row.get(col, "").split(",")]
                for name in names:
                    name = _clean_variety_name(name)
                    if name and _is_valid_variety_name(name):
                        results.append({
                            "dop": dop, "gi_type": gi_type,
                            "variety": name, "color": color, "role": "autorizada",
                        })
            continue

        # Case 2: Single variety column
        name = row.get(variety_col or "", "").strip()
        if not name:
            continue

        name = _clean_variety_name(name)
        if not _is_valid_variety_name(name):
            continue

        color = "unknown"
        all_row_text = " ".join(v.lower() for v in row.values())
        if any(w in all_row_text for w in ("blanca", "blanco", "blanc")):
            color = "blanca"
        elif any(w in all_row_text for w in ("tinta", "tinto", "negre")):
            color = "tinta"

        role = "autorizada"
        if "principal" in all_row_text:
            role = "principal"
        elif "secundar" in all_row_text or "complementar" in all_row_text:
            role = "secundaria"

        results.append({
            "dop": dop, "gi_type": gi_type,
            "variety": name, "color": color, "role": role,
        })

    return results


def _clean_variety_name(name: str) -> str:
    """Clean and normalize a variety name."""
    name = name.strip().strip("*").strip("_").strip("·").strip("-")
    # Remove parenthetical explanations
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = name.strip()
    return name.title() if name else ""


def _is_valid_variety_name(name: str) -> bool:
    """Check if a string looks like a valid variety name."""
    if not name or len(name) < 2:
        return False
    lower = name.lower().strip()
    # Reject purely numeric values
    if re.match(r"^[\d\s.,]+$", lower):
        return False
    # Reject parameter names and organoleptic terms
    for rw in VARIETY_REJECT_WORDS:
        if rw in lower:
            return False
    # Reject things like "≥ 3,5 G/L"
    if re.search(r"[≥≤<>/%]", name):
        return False
    # Reject strings dominated by special characters
    alpha_count = sum(1 for c in lower if c.isalpha())
    if alpha_count < 3:
        return False
    return True


def _extract_yield_rows(
    dop: str,
    gi_type: str,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Extract yield rows from a parsed table."""
    results: list[dict[str, Any]] = []
    all_headers = list(rows[0].keys()) if rows else []

    for row in rows:
        kg_ha: float | None = None
        hl_ha: float | None = None
        wine_type = "general"
        variety = ""

        for h in all_headers:
            val = row.get(h, "")
            v, u, _ = _extract_value_and_relation(val)
            if v is None:
                continue
            hl = h.lower()
            raw_lower = val.lower()
            if "kg/ha" in hl or "kg/ha" in raw_lower:
                kg_ha = v
            elif "hl/ha" in hl or "hl/ha" in raw_lower:
                hl_ha = v
            elif "kg" in hl or "kg" in raw_lower:
                kg_ha = v

        # Infer variety / wine type
        wt = _infer_wine_type(row, all_headers)
        if wt != "general":
            wine_type = wt
        variety = row.get(all_headers[0], "") if all_headers else ""

        if kg_ha or hl_ha:
            results.append({
                "dop": dop,
                "gi_type": gi_type,
                "wine_type": wine_type,
                "variety": variety if len(variety) > 2 else wine_type,
                "kg_ha": kg_ha,
                "hl_ha": hl_ha,
            })

    return results


def _extract_municipality_rows(
    dop: str,
    gi_type: str,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Extract municipality rows from a parsed table."""
    results: list[dict[str, Any]] = []
    all_headers = list(rows[0].keys()) if rows else []

    muni_col = None
    zone_col = None
    prov_col = None
    for h in all_headers:
        hl = h.lower()
        if "municipi" in hl:
            muni_col = h
        elif "zona" in hl:
            zone_col = h
        elif "provincia" in hl or "província" in hl:
            prov_col = h

    if muni_col is None and all_headers:
        muni_col = all_headers[0]

    for row in rows:
        name = row.get(muni_col or "", "").strip()
        if not name or len(name) < 2:
            continue

        results.append({
            "dop": dop,
            "gi_type": gi_type,
            "municipality": name,
            "zone": row.get(zone_col or "", "").strip() or None,
            "province": row.get(prov_col or "", "").strip() or None,
        })

    return results


def _extract_gi_info(folder_name: str) -> tuple[str, str]:
    """Extract GI type and display name from a folder name.

    Thin wrapper over ``winegpt.schema.parse_folder_name``; returns
    ``(gi_type, display_name)`` where ``gi_type`` is ``"unknown"`` for
    non-GI folders (kept for the existing ``"unknown"`` sentinel behavior).
    """
    info = parse_folder_name(folder_name)
    return (info.gi_type if info.is_gi else "unknown", info.display_name)


def build_tables_db(countries: list[str] | None = None) -> dict[str, int]:
    """Build tables database from enriched markdown files.

    Returns counts per table type.
    """
    if countries is None:
        countries = list(SUPPORTED_COUNTRIES)

    extracted_dir = EXTRACTED_DIR
    TABLES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(TABLES_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dop TEXT NOT NULL,
            gi_type TEXT NOT NULL,
            wine_type TEXT DEFAULT 'general',
            parameter TEXT NOT NULL,
            value REAL,
            unit TEXT,
            relation TEXT DEFAULT 'target',
            source_file TEXT,
            UNIQUE(dop, gi_type, wine_type, parameter, relation)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS varieties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dop TEXT NOT NULL,
            gi_type TEXT NOT NULL,
            variety TEXT NOT NULL,
            color TEXT DEFAULT 'unknown',
            role TEXT DEFAULT 'autorizada',
            source_file TEXT,
            UNIQUE(dop, variety, color)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS yields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dop TEXT NOT NULL,
            gi_type TEXT NOT NULL,
            wine_type TEXT DEFAULT 'general',
            variety TEXT DEFAULT '',
            kg_ha REAL,
            hl_ha REAL,
            source_file TEXT,
            UNIQUE(dop, variety, wine_type)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS municipalities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dop TEXT NOT NULL,
            gi_type TEXT NOT NULL,
            municipality TEXT NOT NULL,
            zone TEXT,
            province TEXT,
            source_file TEXT,
            UNIQUE(dop, municipality)
        )
    """)

    # Clear existing data
    for table in ("analytics", "varieties", "yields", "municipalities"):
        conn.execute(f"DELETE FROM {table}")

    counts: dict[str, int] = {"analytics": 0, "varieties": 0, "yields": 0, "municipalities": 0}

    for country in countries:
        country_dir = extracted_dir / country
        if not country_dir.exists():
            logger.warning("Country dir not found: %s", country_dir)
            continue

        for md_path in sorted(country_dir.rglob("*.md")):
            gi_name = md_path.parent.name
            content = md_path.read_text(encoding="utf-8", errors="replace")

            # Find table sections (shared constant with table_extractor.py)
            sections = content.split(TABLES_SECTION_HEADER)
            if len(sections) < 2:
                continue

            table_section = sections[1]
            # Split by ### Taula headers (shared prefix with table_extractor.py)
            tables = re.split(
                re.escape(TABLES_SUBSECTION_PREFIX) + r"\s*\([^)]*\)",
                table_section,
            )
            if len(tables) < 2:
                continue

            gi_type, dop_name = _extract_gi_info(gi_name)

            for table_text in tables[1:]:
                table_text = table_text.strip()
                if not table_text:
                    continue

                parsed_rows = _parse_markdown_table(table_text)
                if not parsed_rows:
                    continue

                table_type = _classify_table(parsed_rows)
                if not table_type:
                    continue

                if table_type == "analytics":
                    rows = _extract_analytics_rows(dop_name, gi_type, parsed_rows)
                elif table_type == "varieties":
                    rows = _extract_varieties_rows(dop_name, gi_type, parsed_rows)
                elif table_type == "yields":
                    rows = _extract_yield_rows(dop_name, gi_type, parsed_rows)
                elif table_type == "municipalities":
                    rows = _extract_municipality_rows(dop_name, gi_type, parsed_rows)
                else:
                    continue

                for row_data in rows:
                    row_data["source_file"] = md_path.name
                    try:
                        _insert_row(conn, table_type, row_data)
                        counts[table_type] += 1
                    except sqlite3.IntegrityError:
                        pass  # Duplicate, skip

    conn.commit()
    conn.close()

    logger.info(
        "Tables DB built: analytics=%d varieties=%d yields=%d municipalities=%d",
        counts["analytics"], counts["varieties"], counts["yields"], counts["municipalities"],
    )
    return counts


def _insert_row(conn: sqlite3.Connection, table: str, data: dict[str, Any]) -> None:
    """Insert a row into the given table."""
    keys = [k for k in data if k != "source_file"]
    placeholders = ", ".join("?" for _ in keys)
    cols = ", ".join(keys)
    values = [data[k] for k in keys]
    conn.execute(
        f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})",
        values,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the structured tables DB (data/tables.db)",
    )
    parser.add_argument(
        "--countries", nargs="+", default=list(SUPPORTED_COUNTRIES),
        help="Countries to scan (default: all supported).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    logger.info("=== enotropos — Tables DB build ===")
    build_tables_db(countries=args.countries)


if __name__ == "__main__":
    main()
