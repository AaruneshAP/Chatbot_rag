"""
retrieval_hybrid.py - Hybrid Lexical & Dense Retrieval with Reciprocal Rank Fusion

Combines BM25 lexical keyword matching (via rank_bm25.BM25Okapi) with dense vector
search (via ChromaDB), fusing top-10 candidates using Reciprocal Rank Fusion (RRF):
  RRF_Score(d) = sum( 1 / (60 + rank_in_list) )
"""

import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from retrieval_vector import retrieve_vector

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
RRF_K = 60 # Standard Reciprocal Rank Fusion smoothing constant

_BM25_INDEX: Optional[BM25Okapi] = None
_CORPUS_CHUNKS: Optional[List[Dict[str, Any]]] = None
_CHUNK_MAP: Optional[Dict[str, Dict[str, Any]]] = None


def simple_tokenize(text: str) -> List[str]:
    """
    Simple, fast tokenization: lowercases and extracts alphanumeric and underscore words.
    Preserves programming identifiers like 'drop_duplicates', 'scale_pos_weight', etc.
    """
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def get_bm25_index():
    """Lazy-loads the chunks.jsonl corpus and builds the BM25 index."""
    global _BM25_INDEX, _CORPUS_CHUNKS, _CHUNK_MAP
    if _BM25_INDEX is None:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}. Run ingest.py first!")

        _CORPUS_CHUNKS = []
        _CHUNK_MAP = {}
        tokenized_corpus = []

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    _CORPUS_CHUNKS.append(chunk)
                    _CHUNK_MAP[chunk["chunk_id"]] = chunk
                    tokenized_corpus.append(simple_tokenize(chunk["text"]))

        _BM25_INDEX = BM25Okapi(tokenized_corpus)

    return _BM25_INDEX, _CORPUS_CHUNKS, _CHUNK_MAP


def retrieve_bm25(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Retrieves top-k chunks using BM25Okapi."""
    bm25, corpus, _ = get_bm25_index()
    query_tokens = simple_tokenize(query)
    if not query_tokens:
        return []

    doc_scores = bm25.get_scores(query_tokens)
    # Get top_k indices sorted descending by score
    top_indices = sorted(range(len(doc_scores)), key=lambda idx: doc_scores[idx], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = corpus[idx]
        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "source_library": chunk.get("source_library", "unknown"),
            "page_or_section": chunk.get("page_or_section", "unknown"),
            "doc_version": chunk.get("doc_version", "unknown"),
            "bm25_score": float(doc_scores[idx]),
            "text": chunk["text"]
        })
    return results


def retrieve_hybrid(query: str, top_k: int = 5, candidate_k: int = 10) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval pipeline:
    1. Fetches candidate_k=10 from BM25.
    2. Fetches candidate_k=10 from ChromaDB vector search.
    3. Merges via Reciprocal Rank Fusion (RRF).
    4. Returns top_k=5 final fused chunks.
    """
    _, _, chunk_map = get_bm25_index()

    bm25_results = retrieve_bm25(query, top_k=candidate_k)
    vector_results = retrieve_vector(query, top_k=candidate_k)

    rrf_scores: Dict[str, float] = {}
    doc_meta_map: Dict[str, Dict[str, Any]] = {}
    doc_ranks_bm25: Dict[str, int] = {}
    doc_ranks_vector: Dict[str, int] = {}

    for item in bm25_results:
        cid = item["chunk_id"]
        doc_ranks_bm25[cid] = item["rank"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + item["rank"])
        doc_meta_map[cid] = item

    for item in vector_results:
        cid = item["chunk_id"]
        doc_ranks_vector[cid] = item["rank"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + item["rank"])
        if cid not in doc_meta_map:
            doc_meta_map[cid] = item

    # Sort candidates by combined RRF score descending
    sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    fused_results = []
    for rank, cid in enumerate(sorted_cids, start=1):
        base_item = doc_meta_map[cid]
        fused_results.append({
            "rank": rank,
            "chunk_id": cid,
            "source_library": base_item["source_library"],
            "page_or_section": base_item["page_or_section"],
            "doc_version": base_item.get("doc_version", "unknown"),
            "rrf_score": rrf_scores[cid],
            "bm25_rank": doc_ranks_bm25.get(cid, None),
            "vector_rank": doc_ranks_vector.get(cid, None),
            "text": base_item["text"]
        })

    return fused_results


def main():
    test_query = sys.argv[1] if len(sys.argv) > 1 else "What does the scale_pos_weight parameter do for imbalanced classification?"
    print(f"Hybrid Query: {test_query}\n")
    results = retrieve_hybrid(test_query, top_k=5)
    for r in results:
        snippet = r['text'][:140].replace('\n', ' ')
        print(f"Rank {r['rank']} | RRF: {r['rrf_score']:.5f} (BM25 #{r['bm25_rank']}, Vec #{r['vector_rank']}) | Lib: {r['source_library']} | Section: {r['page_or_section']}")
        print(f"  Snippet: {snippet}...\n")


if __name__ == "__main__":
    main()
