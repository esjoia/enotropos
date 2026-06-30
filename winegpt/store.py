"""enotropos — Vector store module.

Manages ChromaDB for storing and querying child chunk embeddings. Parent
sections are persisted separately in JSON files (see winegpt.parents) and
loaded at query time to provide the LLM with rich, complete context.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from winegpt.config import (
    CHROMA_PATH,
    SUPPORTED_COUNTRIES,
    TOP_K_CHUNKS,
)
from winegpt.parents import delete_parents, load_parents
from winegpt.schema import METADATA_FIELDS

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger(__name__)


def _children_collection_name(country: str) -> str:
    return f"{country}_children"


_client: chromadb.PersistentClient | None = None
_client_lock = threading.Lock()


def get_client() -> chromadb.PersistentClient:
    """Get a persistent ChromaDB client (thread-safe singleton)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import chromadb

                CHROMA_PATH.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def get_or_create_children_collection(
    country: str,
    client: chromadb.PersistentClient | None = None,
) -> Any:
    """Get or create the children collection for a country."""
    if client is None:
        client = get_client()

    name = _children_collection_name(country)
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return collection


def reset_children_collection(
    country: str,
    client: chromadb.PersistentClient | None = None,
) -> None:
    """Delete and recreate the children collection for a country."""
    if client is None:
        client = get_client()

    name = _children_collection_name(country)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Collection '%s' reset", name)


def add_chunks(
    chunks: list[dict[str, Any]],
    country: str,
    client: chromadb.PersistentClient | None = None,
) -> int:
    """Add embedded child chunks to a country's ChromaDB collection.

    Returns number of chunks added.
    """
    if not chunks:
        return 0

    if client is None:
        client = get_client()
    collection = get_or_create_children_collection(country, client)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        if "embedding" not in chunk:
            continue
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        # Index the child (small) fragment for vector search
        documents.append(chunk["markdown"])
        # Build metadata from the canonical field list so the Chroma schema
        # stays aligned with the producers (single source: METADATA_FIELDS).
        metadatas.append({
            field: chunk.get(field, "") for field in METADATA_FIELDS
        })

    if not ids:
        return 0

    # Add in batches to avoid large requests
    batch_size = int(os.getenv("CHROMA_BATCH_SIZE", "500"))
    total = 0
    for i in range(0, len(ids), batch_size):
        batch_slice = slice(i, i + batch_size)
        collection.add(
            ids=ids[batch_slice],
            embeddings=embeddings[batch_slice],
            documents=documents[batch_slice],
            metadatas=metadatas[batch_slice],
        )
        total += len(ids[batch_slice])

    logger.info("Added %d chunks to %s", total, _children_collection_name(country))
    return total


def _query_single_country(
    query_embedding: list[float],
    k: int,
    country: str,
    gi_type: str | None,
    gi_names: list[str] | None,
    client: chromadb.PersistentClient,
) -> list[dict[str, Any]]:
    """Query the children collection for one country."""
    collection = get_or_create_children_collection(country, client)

    where: dict[str, Any] | None = None
    conditions: list[dict[str, Any]] = []
    if gi_type:
        conditions.append({"gi_type": gi_type})
    if gi_names:
        if len(gi_names) == 1:
            conditions.append({"gi_name": gi_names[0]})
        else:
            conditions.append({"gi_name": {"$in": gi_names}})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return output

    for i, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        output.append({
            "id": chunk_id,
            "document": results["documents"][0][i],
            "metadata": metadata,
            "distance": results["distances"][0][i],
        })

    return output


def query(
    query_embedding: list[float],
    k: int = TOP_K_CHUNKS,
    country: str | None = None,
    gi_type: str | None = None,
    gi_names: list[str] | None = None,
    client: chromadb.PersistentClient | None = None,
) -> list[dict[str, Any]]:
    """Query ChromaDB for relevant chunks.

    If ``country`` is None (All), queries both supported country collections
    and merges results. Returns chunks with parent sections loaded from disk.
    """
    if client is None:
        client = get_client()

    countries = [country] if country else list(SUPPORTED_COUNTRIES)

    # Retrieve up to k candidates per country; final reranking will trim.
    candidates: list[dict[str, Any]] = []
    for c in countries:
        if c not in SUPPORTED_COUNTRIES:
            logger.warning("Unknown country '%s', skipping", c)
            continue
        candidates.extend(
            _query_single_country(
                query_embedding,
                k=k,
                country=c,
                gi_type=gi_type,
                gi_names=gi_names,
                client=client,
            )
        )

    # Sort candidates by embedding distance (ascending)
    candidates.sort(key=lambda x: x["distance"])

    # Deduplicate by parent and load parent texts
    output: list[dict[str, Any]] = []
    seen_parents: set[str] = set()
    parent_ids: set[str] = set()

    for chunk in candidates:
        metadata = chunk["metadata"]
        parent_id = metadata.get("parent_id", "")
        dedup_key = parent_id if parent_id else chunk["id"]

        if dedup_key in seen_parents:
            continue
        seen_parents.add(dedup_key)

        if parent_id:
            parent_ids.add(parent_id)

        output.append(chunk)
        if len(output) >= k:
            break

    # Load all needed parents at once
    parent_map = load_parents(parent_ids)

    # Replace child document with parent text when available
    for chunk in output:
        metadata = chunk["metadata"]
        parent_id = metadata.get("parent_id", "")
        if parent_id and parent_id in parent_map:
            chunk["document"] = parent_map[parent_id]["markdown"]
            metadata["parent_section"] = parent_map[parent_id].get("section", "")

    return output


def delete_by_country(country: str, client: chromadb.PersistentClient | None = None) -> int:
    """Delete all children and parents for a country."""
    if client is None:
        client = get_client()

    collection = get_or_create_children_collection(country, client)
    existing = collection.get()
    count = len(existing["ids"]) if existing["ids"] else 0
    if count:
        collection.delete(ids=existing["ids"])
        logger.info("Deleted %d chunks from %s", count, _children_collection_name(country))

    delete_parents(country)
    return count


def reset_all_collections(client: chromadb.PersistentClient | None = None) -> None:
    """Reset all children collections and delete all parents."""
    if client is None:
        client = get_client()

    for country in SUPPORTED_COUNTRIES:
        reset_children_collection(country, client)
        delete_parents(country)
