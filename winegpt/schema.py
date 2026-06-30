"""enotropos — Shared data schema and naming conventions.

Single source of truth for:

* The chunk metadata fields stored in ChromaDB (``METADATA_FIELDS`` /
  ``ChunkMetadata``).
* Parsing of ``DOP_*`` / ``IGP_*`` / knowledge folder names
  (``parse_folder_name``).
* Generation of stable, parseable chunk and parent ids
  (``make_chunk_id`` / ``make_parent_id``).

Centralizing these here removes the duplicated folder-name parsing that used to
live in ``extract``, ``chunk``, ``tools``, ``build_graph`` and
``build_tables_db``, and makes the ``store.add_chunks`` metadata contract
enforceable instead of implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

# Separator used in chunk/parent ids. Folder and pdf names must not contain it.
_ID_SEP = "__"

# Canonical metadata fields written to ChromaDB for every child chunk.
# ``store.add_chunks`` builds exactly these fields; producers must populate them
# (missing values default to the empty string at storage time).
METADATA_FIELDS: tuple[str, ...] = (
    "folder",
    "source_file",
    "country",
    "gi_type",
    "gi_name",
    "subcategory",
    "language",
    "section",
    "parent_id",
)


class ChunkMetadata(TypedDict):
    """Metadata stored alongside each child chunk in ChromaDB."""

    folder: str
    source_file: str
    country: str
    gi_type: str
    gi_name: str
    subcategory: str
    language: str
    section: str
    parent_id: str


class ChunkRecord(TypedDict):
    """Full child chunk record flowing from producers to embed/store."""

    chunk_id: str
    folder: str
    source_file: str
    country: str
    gi_type: str
    gi_name: str
    subcategory: str
    language: str
    section: str
    markdown: str
    parent_id: str


@dataclass(frozen=True)
class FolderInfo:
    """Parsed components of a corpus folder name."""

    gi_type: str
    """``"DOP"`` / ``"IGP"`` / ``"knowledge"`` (``""`` if unrecognized)."""

    gi_name: str
    """Underscore form stored in metadata (e.g. ``"Rioja_Oriental"``)."""

    display_name: str
    """Human-readable form for UIs (e.g. ``"Rioja Oriental"``)."""

    is_gi: bool
    """``True`` for ``DOP_*`` / ``IGP_*`` folders."""


def parse_folder_name(name: str) -> FolderInfo:
    """Parse a corpus folder name into its components.

    * ``DOP_Rioja`` → ``FolderInfo("DOP", "Rioja", "Rioja", True)``
    * ``IGP_Castilla` → ``FolderInfo("IGP", "Castilla", "Castilla", True)``
    * ``enologia`` (knowledge) → ``FolderInfo("knowledge", "enologia", "enologia", False)``

    Display names replace ``__`` with ``" / "`` and ``_`` with spaces, matching
    the previous behavior of ``tools.list_dops``.
    """
    if name.startswith("DOP_"):
        rest = name[4:]
        return FolderInfo(
            gi_type="DOP",
            gi_name=rest,
            display_name=rest.replace("__", " / ").replace("_", " "),
            is_gi=True,
        )
    if name.startswith("IGP_"):
        rest = name[4:]
        return FolderInfo(
            gi_type="IGP",
            gi_name=rest,
            display_name=rest.replace("__", " / ").replace("_", " "),
            is_gi=True,
        )
    return FolderInfo(
        gi_type="knowledge",
        gi_name=name,
        display_name=name.replace("_", " "),
        is_gi=False,
    )


def make_chunk_id(country: str, folder: str, pdf: str, index: int) -> str:
    """Build a stable child chunk id: ``country__folder__pdf__index``.

    The country prefix makes ids globally unique across collections and keeps
    the format symmetric with ``make_parent_id`` (4 ``__``-separated parts).
    """
    return f"{country}{_ID_SEP}{folder}{_ID_SEP}{pdf}{_ID_SEP}{index}"


def make_parent_id(country: str, folder: str, pdf: str, parent_index: int) -> str:
    """Build a stable parent id: ``country__folder__pdf__parent_index``.

    Parsed by ``winegpt.parents._parse_parent_id`` to locate the parent JSON.
    """
    return f"{country}{_ID_SEP}{folder}{_ID_SEP}{pdf}{_ID_SEP}parent_{parent_index}"


@dataclass
class StreamResult:
    """Mutable holder populated by streaming RAG / agent generators.

    The streaming functions return ``(generator, StreamResult)``. The caller
    iterates the generator to render tokens; once exhausted, the holder is
    populated with the final ``answer``, ``citations`` and ``tools``. This
    replaces the fragile ``generator.value`` pattern (which is only exposed via
    ``StopIteration.value`` and silently dropped on attribute-access failure).
    """

    answer: str = ""
    citations: list[dict[str, str]] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
