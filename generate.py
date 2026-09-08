"""
generate.py - Direct LLM Generation Layer with Mandatory Citation Enforcement

Connects top-k retrieval (hybrid BM25+Vector or Vector-only) directly to Groq's
API (llama-3.3-70b-versatile) without framework overhead (no LangChain, no LlamaIndex).
Enforces grounded answers with bracketed source citations ([1], [2]) linking every
factual assertion to its retrieved documentation chunk.
"""

import os
import sys
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from groq import Groq

from retrieval_hybrid import retrieve_hybrid
from retrieval_vector import retrieve_vector

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables from local .env
load_dotenv()

# Plain instruct-completion models on Groq (no compound agentic systems with web search/code execution)
DEFAULT_GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
GROQ_MODEL = DEFAULT_GROQ_MODELS[0]


def get_groq_client() -> Groq:
    """Initializes and returns the Groq client, validating the API key."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                api_key = str(st.secrets["GROQ_API_KEY"]).strip()
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set or empty. "
            "For local runs, add GROQ_API_KEY to your .env file. "
            "For Streamlit Cloud, configure GROQ_API_KEY in App Settings > Secrets."
        )
    return Groq(api_key=api_key)


def build_prompt_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into numbered context blocks with metadata markers.
    Example:
      [1] (Library: pandas | Source: reference/api/pandas.DataFrame.loc.html)
      <chunk text>
    """
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        lib = chunk.get("source_library", "unknown")
        src = chunk.get("page_or_section", "unknown")
        text = chunk.get("text", "").strip()
        block = f"[{idx}] (Library: {lib} | Source: {src})\n{text}"
        context_blocks.append(block)
    return "\n\n".join(context_blocks)


def generate_answer(
    query: str,
    retrieval_method: str = "hybrid",
    top_k: int = 5,
    client: Optional[Groq] = None
) -> Dict[str, Any]:
    """
    Executes the full RAG generation pipeline:
    1. Retrieves top-k chunks using the selected retrieval method ('hybrid' or 'vector').
    2. Builds an explicit citation-enforcing prompt.
    3. Calls Groq chat completion (llama-3.3-70b-versatile).
    4. Returns answer text alongside structured citation metadata.
    """
    if client is None:
        client = get_groq_client()

    # 1. Retrieve top-k chunks
    if retrieval_method == "vector":
        retrieved_chunks = retrieve_vector(query, top_k=top_k)
    elif retrieval_method == "hybrid":
        retrieved_chunks = retrieve_hybrid(query, top_k=top_k)
    else:
        raise ValueError(f"Unknown retrieval_method '{retrieval_method}'. Choose 'hybrid' or 'vector'.")

    # 2. Format context with bracketed citation markers
    context_text = build_prompt_context(retrieved_chunks)

    # 3. System prompt enforcing strict grounding and citation markers
    system_prompt = (
        "You are an expert technical assistant for Python data science libraries "
        "(Pandas, Scikit-Learn, and XGBoost).\n\n"
        "Instructions:\n"
        "1. Answer the user's question using ONLY the facts directly stated in the provided documentation context below.\n"
        "2. Do NOT extrapolate, speculate, or introduce external knowledge not present in the context.\n"
        "3. Cite every statement or claim with bracketed citation markers like [1], [2], etc., matching the numbered "
        "source chunk(s) where that information appears.\n"
        "4. If the provided context does not contain sufficient information to answer the question, state explicitly: "
        "'The provided documentation does not contain sufficient information to answer this question.' Do not guess."
    )

    user_message = (
        f"Documentation Context:\n"
        f"---------------------\n"
        f"{context_text}\n"
        f"---------------------\n\n"
        f"Question: {query}\n\n"
        f"Answer with citations:"
    )

    # 4. Direct Groq chat completion call with automatic model fallback
    completion = None
    chosen_model = None
    last_error = None

    for model_candidate in DEFAULT_GROQ_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model_candidate,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,  # Low temperature for factual grounding
                max_tokens=1024,
            )
            chosen_model = model_candidate
            break
        except Exception as err:
            last_error = err
            if "model_not_found" in str(err) or "does not exist" in str(err):
                continue
            raise err

    if completion is None:
        raise RuntimeError(f"All attempted Groq models failed. Last error: {last_error}")

    answer_text = completion.choices[0].message.content.strip()

    # 5. Build structured list of cited sources for UI consumption
    cited_sources = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        snippet = chunk.get("text", "")[:180].replace("\n", " ") + "..."
        cited_sources.append({
            "citation_id": idx,
            "marker": f"[{idx}]",
            "chunk_id": chunk.get("chunk_id"),
            "source_library": chunk.get("source_library"),
            "page_or_section": chunk.get("page_or_section"),
            "doc_version": chunk.get("doc_version"),
            "snippet": snippet,
            "full_text": chunk.get("text", "")
        })

    return {
        "query": query,
        "retrieval_method": retrieval_method,
        "answer": answer_text,
        "cited_sources": cited_sources,
        "model": chosen_model,
        "usage": {
            "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
            "total_tokens": completion.usage.total_tokens if completion.usage else 0,
        }
    }


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What's the difference between .loc and .iloc for indexing?"
    method = sys.argv[2] if len(sys.argv) > 2 else "hybrid"

    print(f"Query: {query}")
    print(f"Method: {method}\n" + "-" * 60)

    try:
        res = generate_answer(query, retrieval_method=method)
        print("\n--- GENERATED ANSWER ---")
        print(res["answer"])
        print("\n--- RETRIEVED SOURCES ---")
        for src in res["cited_sources"]:
            print(f"{src['marker']} [{src['source_library']}] {src['page_or_section']}: {src['snippet']}")
    except Exception as e:
        print(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
