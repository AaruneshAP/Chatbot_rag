"""
faithfulness.py - Explainable Lexical Faithfulness Verification Heuristic

Evaluates groundedness of RAG-generated answers by checking keyword overlap
between each generated claim/sentence and its cited source chunks.

Design Note:
This is an interpretable, zero-dependency proxy heuristic intended for clear
white-box inspection in portfolio reviews and technical interviews.
It is NOT a full NLI (Natural Language Inference) entailment model.
Production systems would typically use cross-encoder NLI (e.g. RoBERTa-MNLI)
or LLM-as-a-judge (Ragas / TruLens) for fine-grained semantic entailment.
"""

import re
import sys
from typing import List, Dict, Any, Set, Tuple

# Standard English stop words to filter out before computing keyword overlap
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Threshold below which a cited sentence is flagged as low confidence
LOW_CONFIDENCE_THRESHOLD = 0.30


def extract_keywords(text: str) -> Set[str]:
    """
    Extracts informative keywords from text:
    - Lowercases text
    - Identifies alphanumeric tokens and programming identifiers (e.g. 'drop_duplicates')
    - Discards stop words and short tokens (< 3 chars)
    """
    # Matches words, including programming symbols with underscores
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    keywords = {
        token for token in tokens
        if len(token) >= 3 and token not in STOP_WORDS and not token.isdigit()
    }
    return keywords


def split_into_sentences(text: str) -> List[str]:
    """
    Splits answer text into sentences or coherent claim units,
    handling bullet points, newlines, and sentence terminators.
    """
    cleaned = text.strip()
    # Replace markdown bullet headers with clean spacing
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    sentences = []

    for line in lines:
        # Strip leading bullet indicators (e.g. "- ", "* ", "1. ")
        line = re.sub(r"^[-*•]\s+|\d+\.\s+", "", line)
        # Skip code blocks, citation headers, and trailing bibliography lines
        if (
            not line
            or line.startswith("```")
            or line.lower().startswith("citations:")
            or line.lower().startswith("sources:")
            or re.match(r"^\[?\d+\]?\s*\(?Library:", line, re.IGNORECASE)
        ):
            continue
        # Split sentences on:
        # 1. Punctuation followed by whitespace, EXCEPT when followed by citation brackets
        # 2. After a citation bracket closing (']') followed by whitespace and a new sentence start
        parts = re.split(r"(?<=[.!?])\s+(?![\[\d])|(?<=\])\s+(?=[A-Z0-9`\"'])", line)

        # Merge any isolated citation fragments (e.g. "[1][5]") back to the preceding sentence
        merged_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if re.fullmatch(r"(?:\[\d+\])+", p) and merged_parts:
                merged_parts[-1] = merged_parts[-1] + " " + p
            else:
                merged_parts.append(p)

        for part in merged_parts:
            if len(part) > 5:
                sentences.append(part)

    return sentences


def parse_citations(sentence: str) -> List[int]:
    """
    Finds all citation markers in the format [n] within a sentence.
    Returns list of unique integer citation IDs.
    """
    matches = re.findall(r"\[(\d+)\]", sentence)
    return sorted(list(set(int(m) for m in matches)))


def check_sentence_faithfulness(
    sentence: str,
    cited_sources_map: Dict[int, str]
) -> Dict[str, Any]:
    """
    Checks a single sentence against its cited sources:
    1. Extracts [n] citations.
    2. Gathers keywords from sentence and cited chunk(s).
    3. Computes intersection ratio: |sentence_kw ∩ source_kw| / |sentence_kw|.
    4. Flags sentence status: 'ok', 'low_confidence', or 'uncited'.
    """
    citations = parse_citations(sentence)
    sentence_kw = extract_keywords(sentence)

    if not sentence_kw:
        return {
            "sentence": sentence,
            "citations": citations,
            "overlap_score": 1.0,
            "flag": "ok",
            "matched_keywords": [],
            "missing_keywords": []
        }

    if not citations:
        # Sentence contains claims without any bracketed citation
        return {
            "sentence": sentence,
            "citations": [],
            "overlap_score": 0.0,
            "flag": "uncited",
            "matched_keywords": [],
            "missing_keywords": sorted(list(sentence_kw))
        }

    # Combine text from all cited chunks for this sentence
    source_texts = [cited_sources_map[c] for c in citations if c in cited_sources_map]
    combined_source_text = " ".join(source_texts)
    source_kw = extract_keywords(combined_source_text)

    matched = sentence_kw.intersection(source_kw)
    missing = sentence_kw - source_kw

    overlap_score = len(matched) / len(sentence_kw) if sentence_kw else 1.0
    flag = "low_confidence" if overlap_score < LOW_CONFIDENCE_THRESHOLD else "ok"

    return {
        "sentence": sentence,
        "citations": citations,
        "overlap_score": round(overlap_score, 4),
        "flag": flag,
        "matched_keywords": sorted(list(matched)),
        "missing_keywords": sorted(list(missing))
    }


def evaluate_faithfulness(
    answer: str,
    cited_sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates the full generated answer for grounded faithfulness.
    Returns structured metrics and per-sentence diagnostics.
    """
    # Map citation_id (1..k) to full chunk text
    cited_sources_map = {
        src["citation_id"]: src.get("full_text", "")
        for src in cited_sources
    }

    # Defensive parsing: if zero citations exist in the entire text, flag as unrecognized format
    if not parse_citations(answer):
        sentences = split_into_sentences(answer)
        return {
            "total_sentences": len(sentences),
            "cited_sentences": 0,
            "uncited_sentences": len(sentences),
            "low_confidence_sentences": 0,
            "mean_overlap_score": 0.0,
            "overall_faithfulness_rate": 0.0,
            "status": "citation_format_not_recognized",
            "citation_format_recognized": False,
            "sentence_diagnostics": [
                {
                    "sentence": sent,
                    "citations": [],
                    "overlap_score": 0.0,
                    "flag": "citation_format_not_recognized",
                    "matched_keywords": [],
                    "missing_keywords": sorted(list(extract_keywords(sent)))
                }
                for sent in sentences
            ]
        }

    sentences = split_into_sentences(answer)
    sentence_reports = []
    cited_scores = []
    low_confidence_count = 0
    uncited_count = 0

    for sent in sentences:
        report = check_sentence_faithfulness(sent, cited_sources_map)
        sentence_reports.append(report)

        if report["flag"] == "low_confidence":
            low_confidence_count += 1
            cited_scores.append(report["overlap_score"])
        elif report["flag"] == "ok":
            if report["citations"]:
                cited_scores.append(report["overlap_score"])
        elif report["flag"] == "uncited":
            uncited_count += 1

    total_sentences = len(sentences)
    total_cited = len(cited_scores)
    avg_overlap = (sum(cited_scores) / total_cited) if total_cited > 0 else 0.0

    return {
        "total_sentences": total_sentences,
        "cited_sentences": total_cited,
        "uncited_sentences": uncited_count,
        "low_confidence_sentences": low_confidence_count,
        "mean_overlap_score": round(avg_overlap, 4),
        "overall_faithfulness_rate": round(
            (total_cited - low_confidence_count) / total_sentences, 4
        ) if total_sentences > 0 else 1.0,
        "status": "ok",
        "citation_format_recognized": True,
        "sentence_diagnostics": sentence_reports
    }
