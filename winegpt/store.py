"""enotropos — Vector store module.

Manages ChromaDB for storing and querying document chunk embeddings.
"""
import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from winegpt.config import CHROMA_PATH, TOP_K_CHUNKS

logger = logging.getLogger(__name__)

COLLECTION_NAME = "winegpt_es"


def get_client() -> chromadb.PersistentClient:
    """Get a persistent ChromaDB client."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient | None = None,
) -> Any:
    """Get or create the winegpt collection."""
    if client is None:
        client = get_client()

    # Delete and recreate if called with reset
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return collection


def reset_collection(client: chromadb.PersistentClient | None = None) -> None:
    """Delete and recreate the collection."""
    if client is None:
        client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Collection '%s' reset", COLLECTION_NAME)


def add_chunks(
    chunks: list[dict[str, Any]],
    client: chromadb.PersistentClient | None = None,
) -> int:
    """Add embedded chunks to ChromaDB. Returns number of chunks added."""
    if not chunks:
        return 0

    if client is None:
        client = get_client()
    collection = get_or_create_collection(client)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        if "embedding" not in chunk:
            continue
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["markdown"])
        metadatas.append({
            "folder": chunk.get("folder", ""),
            "source_file": chunk.get("source_file", ""),
            "country": chunk.get("country", ""),
            "gi_type": chunk.get("gi_type", ""),
            "gi_name": chunk.get("gi_name", ""),
            "language": chunk.get("language", ""),
            "section": chunk.get("section", ""),
        })

    if not ids:
        return 0

    # Add in batches to avoid large requests
    batch_size = 100
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

    logger.info("Added %d chunks to ChromaDB", total)
    return total


def query(
    query_embedding: list[float],
    k: int = TOP_K_CHUNKS,
    country: str | None = None,
    gi_type: str | None = None,
    client: chromadb.PersistentClient | None = None,
) -> list[dict[str, Any]]:
    """Query ChromaDB for relevant chunks.

    Returns list of dicts with id, document, metadata, and distance.
    """
    if client is None:
        client = get_client()
    collection = get_or_create_collection(client)

    where: dict[str, Any] | None = None
    conditions: list[dict[str, Any]] = []
    if country:
        conditions.append({"country": country})
    if gi_type:
        conditions.append({"gi_type": gi_type})

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
    if results["ids"] and results["ids"][0]:
        for i, chunk_id in enumerate(results["ids"][0]):
            output.append({
                "id": chunk_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

    return output


def delete_by_country(country: str, client: chromadb.PersistentClient | None = None) -> int:
    """Delete all chunks for a country. Returns count of deleted items."""
    if client is None:
        client = get_client()
    collection = get_or_create_collection(client)

    # Get all ids matching the country filter
    existing = collection.get(where={"country": country})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("Deleted %d chunks for %s", len(existing["ids"]), country)
        return len(existing["ids"])
    return 0
