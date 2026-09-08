"""
retrieval_vector.py - Dense Vector Search Engine

Embeds input queries using sentence-transformers ('all-MiniLM-L6-v2') and retrieves
top-k nearest neighbor chunks from persistent ChromaDB collection.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CHROMA_DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "ml_docs_rag"
MODEL_NAME = "all-MiniLM-L6-v2"

# Global lazy-loaded singletons to avoid re-loading weights on each call
_MODEL: Optional[SentenceTransformer] = None
_COLLECTION = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        if not CHROMA_DB_DIR.exists():
            raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DB_DIR}. Run embed.py first!")
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR), settings=Settings(anonymized_telemetry=False))
        _COLLECTION = client.get_collection(name=COLLECTION_NAME)
    return _COLLECTION


def retrieve_vector(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Encodes query and retrieves top-k chunks from ChromaDB.
    Returns list of dicts:
    [
      {
        "rank": 1,
        "chunk_id": "...",
        "source_library": "...",
        "page_or_section": "...",
        "doc_version": "...",
        "score": 0.85, # Cosine similarity
        "text": "..."
      },
      ...
    ]
    """
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    ranked_results = []
    if results and "ids" in results and results["ids"]:
        ids = results["ids"][0]
        docs = results["documents"][0] if "documents" in results and results["documents"] else [""] * len(ids)
        metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)

        for rank, (doc_id, doc_text, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            ranked_results.append({
                "rank": rank,
                "chunk_id": doc_id,
                "source_library": meta.get("source_library", "unknown"),
                "page_or_section": meta.get("page_or_section", "unknown"),
                "doc_version": meta.get("doc_version", "unknown"),
                "score": 1.0 - dist,  # Cosine similarity
                "text": doc_text
            })

    return ranked_results


def main():
    test_query = sys.argv[1] if len(sys.argv) > 1 else "What's the difference between .loc and .iloc for indexing?"
    print(f"Query: {test_query}\n")
    results = retrieve_vector(test_query, top_k=5)
    for r in results:
        snippet = r['text'][:140].replace('\n', ' ')
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Lib: {r['source_library']} | Section: {r['page_or_section']}")
        print(f"  Snippet: {snippet}...\n")


if __name__ == "__main__":
    main()
