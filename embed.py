"""
embed.py - Vector Embedding Generation and Persistent ChromaDB Indexing

Loads processed chunk records from data/processed/chunks.jsonl, computes
dense vector representations using a local sentence-transformers model,
and persists them into a local ChromaDB collection.

Key Features:
- Local SentenceTransformer ('all-MiniLM-L6-v2') - 384 dimensional embeddings
- ChromaDB PersistentClient targeting data/chroma_db/
- Batch encoding and idempotent collection upserts
- Metadata preservation for granular source tracing and citation
- Built-in verification queries showcasing semantic search across all three libraries
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DB_DIR = REPO_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "ml_docs_rag"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128


def load_chunks(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Loads chunk records from chunks.jsonl."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Processed chunks file not found at {jsonl_path}. Run ingest.py first!")

    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping corrupted JSON on line {line_no}: {e}")
    return chunks


def generate_and_store_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = MODEL_NAME,
    chroma_dir: Path = CHROMA_DB_DIR,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
):
    """
    Computes vector embeddings in batches and indexes them into ChromaDB.
    """
    print("=" * 60)
    print("RAG Ingestion Pipeline: Vector Indexing & Embedding Engine")
    print("=" * 60)
    print(f"Loading embedding model: '{model_name}' (local, sentence-transformers)...")
    model = SentenceTransformer(model_name)
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    print(f"  [OK] Model loaded. Vector dimensionality: {dim}")

    # Initialize persistent ChromaDB client
    chroma_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nInitializing ChromaDB persistent storage at {chroma_dir} ...")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "description": "Documentation embeddings for Pandas, XGBoost, and Scikit-Learn"}
    )

    total_chunks = len(chunks)
    existing_count = collection.count()
    if existing_count >= total_chunks and "--force" not in sys.argv:
        print(f"  [OK] Collection '{collection_name}' already contains {existing_count} items in ChromaDB.")
        print("  Skipping re-embedding (pass --force to re-embed from scratch).\n")
        return

    print(f"Indexing {total_chunks} chunks in batches of {batch_size}...\n")

    start_time = time.time()

    for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding & Storing"):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "source_library": c.get("source_library", "unknown"),
                "source_format": c.get("source_format", "unknown"),
                "page_or_section": str(c.get("page_or_section", "unknown")),
                "doc_version": str(c.get("doc_version", "unknown")),
                "token_count": int(c.get("token_count", 0)),
                "char_count": int(c.get("char_count", 0)),
            }
            for c in batch
        ]

        # Generate dense embeddings
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Idempotent upsert into ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )

    elapsed = time.time() - start_time
    print(f"\n[OK] Successfully indexed {collection.count()} items in ChromaDB in {elapsed:.2f}s ({total_chunks / elapsed:.1f} chunks/sec).")


def run_demo_queries(collection_name: str = COLLECTION_NAME, chroma_dir: Path = CHROMA_DB_DIR):
    """
    Executes sample test queries against ChromaDB to demonstrate semantic retrieval.
    """
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=collection_name)
    model = SentenceTransformer(MODEL_NAME)

    sample_queries = [
        ("Pandas Query", "How do you fill missing or null values in a DataFrame with fillna?"),
        ("XGBoost Query", "How does XGBoost handle missing values during tree split construction?"),
        ("Scikit-Learn Query", "What hyperparameters control tree depth and regularization in RandomForestClassifier?")
    ]

    print("\n" + "=" * 70)
    print("VERIFICATION: SEMANTIC SEARCH RETRIEVAL DEMO")
    print("=" * 70)

    for category, query in sample_queries:
        print(f"\n[Query: {category}] '{query}'")
        query_embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )

        for rank in range(len(results["ids"][0])):
            doc_id = results["ids"][0][rank]
            meta = results["metadatas"][0][rank]
            distance = results["distances"][0][rank] if "distances" in results and results["distances"] else 0.0
            doc_snippet = results["documents"][0][rank][:250].replace("\n", " ") + "..."

            print(f"  Rank #{rank+1} (Score/Cosine Sim: {1 - distance:.4f}) | Chunk ID: {doc_id}")
            print(f"    Source: {meta.get('source_library').upper()} ({meta.get('source_format')}) | Section/Page: {meta.get('page_or_section')}")
            print(f"    Snippet: {doc_snippet}\n")

    print("=" * 70 + "\n")


def main():
    chunks = load_chunks(CHUNKS_PATH)
    generate_and_store_embeddings(chunks)
    run_demo_queries()


if __name__ == "__main__":
    main()
