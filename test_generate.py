"""
test_generate.py - Verification Harness for Generation Layer & Faithfulness Evaluation

Runs 6 representative questions across Pandas, Scikit-Learn, and XGBoost through:
1. Retrieval (Hybrid BM25 + Vector)
2. Groq Generation (llama-3.3-70b-versatile with citation enforcement)
3. Lexical Faithfulness Verification (keyword overlap against cited chunks)

Saves detailed sample outputs to results/generation_samples.json.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from generate import generate_answer
from faithfulness import evaluate_faithfulness

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EVAL_QUESTIONS_PATH = Path("eval_questions.json")
RESULTS_DIR = Path("results")
OUTPUT_SAMPLES_PATH = RESULTS_DIR / "generation_samples.json"

# Selected 6 diverse questions across all 3 libraries:
# - Pandas (q1: indexing, q2: duplicates)
# - Scikit-Learn (q9: preprocessing, q10: cross-validation)
# - XGBoost (q17: scale_pos_weight, q19: missing values in tree split)
SELECTED_QUESTION_IDS = ["q1", "q2", "q9", "q10", "q17", "q19"]


def load_selected_questions() -> List[Dict[str, Any]]:
    """Loads eval_questions.json and extracts the selected test questions."""
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    id_map = {q["id"]: q for q in all_questions}
    selected = [id_map[qid] for qid in SELECTED_QUESTION_IDS if qid in id_map]
    return selected


def run_benchmark_tests():
    """Executes the test suite and displays formatted reports."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_selected_questions()

    print("=" * 80)
    print("PHASE 3 GENERATION HARNESS: GROQ LLM (llama-3.3-70b-versatile)")
    print(f"Testing {len(questions)} questions with mandatory citations & faithfulness checks")
    print("=" * 80 + "\n")

    results = []

    for idx, q_item in enumerate(questions, start=1):
        qid = q_item["id"]
        question = q_item["question"]
        expected_lib = q_item["expected_library"]
        topic = q_item["topic"]

        print(f"\n[{idx}/{len(questions)}] Question {qid} ({expected_lib} - {topic}):")
        print(f"Q: \"{question}\"")
        print("-" * 80)

        # 1. Run generation pipeline
        gen_result = generate_answer(question, retrieval_method="hybrid", top_k=5)
        answer = gen_result["answer"]
        cited_sources = gen_result["cited_sources"]

        # 2. Run faithfulness evaluation
        faith_result = evaluate_faithfulness(answer, cited_sources)

        print("\n📝 GENERATED ANSWER:", flush=True)
        print(answer, flush=True)

        print("\n📚 CITED SOURCES:", flush=True)
        for src in cited_sources:
            print(f"  {src['marker']} [{src['source_library']}] {src['page_or_section']}", flush=True)
            print(f"      Snippet: {src['snippet']}", flush=True)

        print("\n🔍 FAITHFULNESS AUDIT:", flush=True)
        print(f"  Total Sentences:        {faith_result['total_sentences']}", flush=True)
        print(f"  Cited Sentences:        {faith_result['cited_sentences']}", flush=True)
        print(f"  Uncited Sentences:      {faith_result['uncited_sentences']}", flush=True)
        print(f"  Low Confidence (<0.30): {faith_result['low_confidence_sentences']}", flush=True)
        print(f"  Mean Overlap Score:     {faith_result['mean_overlap_score']:.1%}", flush=True)
        print(f"  Faithfulness Rate:      {faith_result['overall_faithfulness_rate']:.1%}", flush=True)

        if faith_result["low_confidence_sentences"] > 0:
            print("  ⚠️ Low confidence sentence details:", flush=True)
            for s in faith_result["sentence_diagnostics"]:
                if s["flag"] == "low_confidence":
                    print(f"    - Overlap: {s['overlap_score']:.1%} | Citations: {s['citations']}", flush=True)
                    print(f"      Sentence: {s['sentence']}", flush=True)

        print("=" * 80, flush=True)

        results.append({
            "id": qid,
            "question": question,
            "expected_library": expected_lib,
            "topic": topic,
            "answer": answer,
            "cited_sources": cited_sources,
            "faithfulness": faith_result
        })

    # Save results to JSON
    with open(OUTPUT_SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nAll sample results saved successfully to {OUTPUT_SAMPLES_PATH}")


if __name__ == "__main__":
    run_benchmark_tests()
