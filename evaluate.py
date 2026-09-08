"""
evaluate.py - Benchmark Harness Comparing Vector-Only vs. Hybrid (BM25 + Vector) Retrieval

Evaluates 24 benchmark questions across Pandas, Scikit-Learn, and XGBoost.
Computes:
- Recall@1, Recall@3, Recall@5 (library-level recall proxy)
- Mean Reciprocal Rank (MRR) of the first correct library hit
- Topic-level Recall@5 breakdown
- Persists per-query details into results/benchmark_results.json
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from retrieval_vector import retrieve_vector
from retrieval_hybrid import retrieve_hybrid

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EVAL_QUESTIONS_PATH = Path("eval_questions.json")
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"


def load_eval_questions() -> List[Dict[str, Any]]:
    if not EVAL_QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Evaluation questions not found at {EVAL_QUESTIONS_PATH}")
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(results: List[Dict[str, Any]], expected_lib: str, max_k: int = 5) -> Dict[str, Any]:
    """
    Computes hit@1, hit@3, hit@5 and Reciprocal Rank for a single query.
    """
    first_hit_rank = None
    for idx, doc in enumerate(results[:max_k], start=1):
        if doc.get("source_library", "").lower() == expected_lib.lower():
            if first_hit_rank is None:
                first_hit_rank = idx
                break

    reciprocal_rank = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
    hit_at_1 = 1 if (first_hit_rank is not None and first_hit_rank <= 1) else 0
    hit_at_3 = 1 if (first_hit_rank is not None and first_hit_rank <= 3) else 0
    hit_at_5 = 1 if (first_hit_rank is not None and first_hit_rank <= 5) else 0

    return {
        "first_hit_rank": first_hit_rank,
        "rr": reciprocal_rank,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5
    }


def run_benchmark():
    print("=" * 70)
    print("PHASE 2 RETRIEVAL BENCHMARK: VECTOR-ONLY VS. HYBRID (BM25 + VECTOR)")
    print("=" * 70)

    questions = load_eval_questions()
    print(f"Loaded {len(questions)} evaluation questions across Pandas, Scikit-Learn, and XGBoost.\n")

    per_question_results = []

    vector_hits = {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0, "mrr_sum": 0.0}
    hybrid_hits = {"hit_at_1": 0, "hit_at_3": 0, "hit_at_5": 0, "mrr_sum": 0.0}

    topic_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    for q in questions:
        qid = q["id"]
        query = q["question"]
        expected_lib = q["expected_library"]
        topic = q["topic"]

        if topic not in topic_stats:
            topic_stats[topic] = {
                "vector": {"total": 0, "hit_5": 0},
                "hybrid": {"total": 0, "hit_5": 0}
            }

        # 1. Vector-Only Retrieval
        v_results = retrieve_vector(query, top_k=5)
        v_metrics = calculate_metrics(v_results, expected_lib, max_k=5)

        # 2. Hybrid Retrieval (BM25 + Vector + RRF)
        h_results = retrieve_hybrid(query, top_k=5, candidate_k=10)
        h_metrics = calculate_metrics(h_results, expected_lib, max_k=5)

        # Accumulate metrics
        for k in ["hit_at_1", "hit_at_3", "hit_at_5"]:
            vector_hits[k] += v_metrics[k]
            hybrid_hits[k] += h_metrics[k]
        vector_hits["mrr_sum"] += v_metrics["rr"]
        hybrid_hits["mrr_sum"] += h_metrics["rr"]

        # Topic stats
        topic_stats[topic]["vector"]["total"] += 1
        topic_stats[topic]["vector"]["hit_5"] += v_metrics["hit_at_5"]
        topic_stats[topic]["hybrid"]["total"] += 1
        topic_stats[topic]["hybrid"]["hit_5"] += h_metrics["hit_at_5"]

        per_question_results.append({
            "id": qid,
            "question": query,
            "expected_library": expected_lib,
            "topic": topic,
            "vector_retrieval": {
                "metrics": v_metrics,
                "top_chunks": [
                    {
                        "rank": r["rank"],
                        "chunk_id": r["chunk_id"],
                        "source_library": r["source_library"],
                        "page_or_section": r["page_or_section"],
                        "score": r["score"],
                        "hit": r["source_library"].lower() == expected_lib.lower()
                    }
                    for r in v_results
                ]
            },
            "hybrid_retrieval": {
                "metrics": h_metrics,
                "top_chunks": [
                    {
                        "rank": r["rank"],
                        "chunk_id": r["chunk_id"],
                        "source_library": r["source_library"],
                        "page_or_section": r["page_or_section"],
                        "rrf_score": r["rrf_score"],
                        "bm25_rank": r["bm25_rank"],
                        "vector_rank": r["vector_rank"],
                        "hit": r["source_library"].lower() == expected_lib.lower()
                    }
                    for r in h_results
                ]
            }
        })

    num_q = len(questions)
    v_rec1 = vector_hits["hit_at_1"] / num_q
    v_rec3 = vector_hits["hit_at_3"] / num_q
    v_rec5 = vector_hits["hit_at_5"] / num_q
    v_mrr = vector_hits["mrr_sum"] / num_q

    h_rec1 = hybrid_hits["hit_at_1"] / num_q
    h_rec3 = hybrid_hits["hit_at_3"] / num_q
    h_rec5 = hybrid_hits["hit_at_5"] / num_q
    h_mrr = hybrid_hits["mrr_sum"] / num_q

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "num_questions": num_q,
        "evaluation_limitation_note": (
            "Recall@k uses a library-level proxy: a hit is recorded if a chunk from the "
            "expected_library is present within the top-k retrieved set. Exact golden chunk-level "
            "relevance is not evaluated because ground truth chunk IDs were not human-annotated."
        ),
        "overall_metrics": {
            "vector_only": {
                "recall_at_1": round(v_rec1, 4),
                "recall_at_3": round(v_rec3, 4),
                "recall_at_5": round(v_rec5, 4),
                "mrr": round(v_mrr, 4)
            },
            "hybrid_bm25_vector": {
                "recall_at_1": round(h_rec1, 4),
                "recall_at_3": round(h_rec3, 4),
                "recall_at_5": round(h_rec5, 4),
                "mrr": round(h_mrr, 4)
            }
        },
        "per_question": per_question_results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Print Main Comparison Table
    print("\n" + "=" * 70)
    print("OVERALL RETRIEVAL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Method':<24} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<10}")
    print("-" * 70)
    print(f"{'Vector-Only (MiniLM-L6)':<24} | {v_rec1:<10.4f} | {v_rec3:<10.4f} | {v_rec5:<10.4f} | {v_mrr:<10.4f}")
    print(f"{'Hybrid (BM25 + Vector)':<24} | {h_rec1:<10.4f} | {h_rec3:<10.4f} | {h_rec5:<10.4f} | {h_mrr:<10.4f}")
    print("=" * 70)

    print("\n* NOTE ON EVALUATION LIMITATION:")
    print("  Recall@k measures whether the top-k results contain at least one chunk from the")
    print("  expected target library. This serves as a library-level routing proxy.")

    # Print Topic Breakdown
    print("\n" + "=" * 70)
    print("TOPIC-LEVEL RECALL@5 BREAKDOWN")
    print("=" * 70)
    print(f"{'Topic Tag':<30} | {'Count':<6} | {'Vector Recall@5':<16} | {'Hybrid Recall@5':<16}")
    print("-" * 70)
    for topic, stats in sorted(topic_stats.items()):
        total = stats["vector"]["total"]
        v_r5 = stats["vector"]["hit_5"] / total if total else 0.0
        h_r5 = stats["hybrid"]["hit_5"] / total if total else 0.0
        print(f"{topic:<30} | {total:<6} | {v_r5:<16.2%} | {h_r5:<16.2%}")
    print("=" * 70)

    print(f"\n[OK] Full benchmark results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_benchmark()
