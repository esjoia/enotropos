"""Check ChromaDB collection counts, sample ids and metadata schema."""
from collections import Counter

import chromadb

c = chromadb.PersistentClient(path="data/chroma_db")
for name in ("Espanya_children", "Coneixement_children"):
    try:
        col = c.get_collection(name)
        all_data = col.get(include=["metadatas"])
        ids = all_data["ids"]
        print(f"\n=== {name}: {len(ids)} chunks ===")
        # Prefix = first '__' part of the id (the country token).
        prefixes = Counter(i.split("__", 1)[0] for i in ids)
        print(f"  id prefixes: {dict(prefixes)}")
        parent_prefixes = Counter(
            (m.get("parent_id", "") or "").split("__", 1)[0]
            for m in all_data["metadatas"]
        )
        print(f"  parent_id prefixes: {dict(parent_prefixes)}")
        # Sample one id + metadata
        print(f"  sample id: {ids[0]}")
        print(f"  sample meta: {all_data['metadatas'][0]}")
    except Exception as e:
        print(f"{name}: err/missing -> {e}")


