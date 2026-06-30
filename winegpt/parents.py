"""enotropos — Parent chunk persistence.

Stores large parent sections as JSON files on disk so that child chunks in
ChromaDB only need to carry a lightweight ``parent_id`` reference. At query
time the parent text is loaded and returned to the LLM as rich context.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from winegpt.config import DATA_DIR

logger = logging.getLogger(__name__)

PARENTS_DIR = DATA_DIR / "parents"


def _parents_file(country: str, folder: str, pdf: str) -> Path:
    """Return the JSON file path for a document's parents."""
    return PARENTS_DIR / country / folder / f"{pdf}.json"


def save_parents(
    country: str,
    folder: str,
    pdf: str,
    parents: list[dict[str, Any]],
) -> None:
    """Persist parent chunks for a single source document.

    Parents are stored as a dict keyed by ``parent_id`` so that ``load_parent``
    can retrieve them in O(1) per file.
    """
    if not parents:
        return

    path = _parents_file(country, folder, pdf)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, dict[str, Any]] = {}
    for parent in parents:
        pid = parent["parent_id"]
        data[pid] = {
            "section": parent.get("section", ""),
            "markdown": parent.get("markdown", ""),
        }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("Saved %d parents to %s", len(parents), path)


def _parse_parent_id(parent_id: str) -> tuple[str, str, str] | None:
    """Parse a parent_id of the form 'country__folder__pdf__parent_idx'."""
    parts = parent_id.split("__", 3)
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2]


def load_parent(parent_id: str) -> dict[str, Any] | None:
    """Load a single parent by its global id."""
    parsed = _parse_parent_id(parent_id)
    if parsed is None:
        return None
    country, folder, pdf = parsed
    path = _parents_file(country, folder, pdf)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    parent = data.get(parent_id)
    if isinstance(parent, dict):
        return parent
    return None


def load_parents(parent_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load many parents at once, grouping by file to minimize disk reads."""
    result: dict[str, dict[str, Any]] = {}
    groups: dict[Path, set[str]] = {}

    for pid in parent_ids:
        parsed = _parse_parent_id(pid)
        if parsed is None:
            continue
        country, folder, pdf = parsed
        path = _parents_file(country, folder, pdf)
        groups.setdefault(path, set()).add(pid)

    for path, ids in groups.items():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for pid in ids:
            if pid in data:
                result[pid] = data[pid]

    return result


def delete_parents(country: str) -> None:
    """Delete all persisted parents for a country."""
    country_dir = PARENTS_DIR / country
    if country_dir.exists():
        for _attempt in range(3):
            try:
                shutil.rmtree(country_dir)
                logger.info("Deleted parents for %s", country)
                return
            except PermissionError:
                logger.warning("Permission error deleting %s, retrying...", country_dir)
                import time
                time.sleep(1)
        shutil.rmtree(str(country_dir), ignore_errors=True)
        logger.warning("Force-deleted parents for %s (some files may remain)", country)
