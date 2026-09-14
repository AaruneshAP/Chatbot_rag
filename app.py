"""
app.py - Streamlit Interactive UI for ML & Data Science Documentation RAG

Provides an interactive chat interface to query documentation for Pandas,
Scikit-Learn, and XGBoost with:
- Configurable retrieval strategy (Vector-only vs Hybrid BM25+Vector)
- Mandatory citation rendering linked to source library & section/page
- Explainable lexical faithfulness indicators
- Separate inspectable panels for raw retrieved chunks
"""

import os
import sys
import streamlit as st
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

from generate import generate_answer
from faithfulness import evaluate_faithfulness
from retrieval_vector import get_collection

# Ensure UTF-8 console encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
CHROMA_DB_DIR = REPO_ROOT / "data" / "chroma_db"
CHROMA_SQLITE_PATH = CHROMA_DB_DIR / "chroma.sqlite3"
CHROMA_SQLITE_LFS_URL = "https://media.githubusercontent.com/media/AaruneshAP/Chatbot_rag/main/data/chroma_db/chroma.sqlite3"
MIN_VALID_SQLITE_BYTES = 100 * 1024 * 1024  # Real DB is ~157MB; pointer is ~134 bytes

# Page Configuration
st.set_page_config(
    page_title="Data Science Docs RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _is_lfs_pointer_or_missing(file_path: Path) -> bool:
    """
    Checks if the SQLite file is missing, corrupted, or an unsmudged Git LFS pointer stub.
    Requires BOTH the 16-byte SQLite header AND substantial binary size (>= 100MB).
    """
    if not file_path.exists():
        return True
    try:
        file_size = file_path.stat().st_size
        with open(file_path, "rb") as f:
            header = f.read(16)
        # Both checks must pass together: magic header AND realistic size
        is_valid = (header == b"SQLite format 3\x00") and (file_size >= MIN_VALID_SQLITE_BYTES)
        return not is_valid
    except Exception:
        return True


def _download_lfs_binary_with_retries(
    dest_path: Path,
    url: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    timeout: int = 60,
) -> bool:
    """
    Downloads remote LFS binary from GitHub media CDN with 3 retry attempts,
    exponential backoff, and 64KB chunk streaming directly to disk.
    Follows the fetch_docs.py streaming and verification pattern.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StreamlitApp/1.0"}

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as resp:
                if resp.status_code == 404:
                    return False
                resp.raise_for_status()

                # Stream to disk in 64KB chunks
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)

                # Validate downloaded temp file before replacing
                if temp_path.exists():
                    temp_size = temp_path.stat().st_size
                    with open(temp_path, "rb") as f:
                        header = f.read(16)
                    if (header == b"SQLite format 3\x00") and (temp_size >= MIN_VALID_SQLITE_BYTES):
                        temp_path.replace(dest_path)
                        return True
                    else:
                        if temp_path.exists():
                            temp_path.unlink()

        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)

    return False


@st.cache_resource(show_spinner=False)
def ensure_chroma_ready() -> bool:
    """
    Verifies that the pre-built ChromaDB vector store is present.
    If the deployment container only cloned the Git LFS pointer text file,
    downloads the real 157.7 MB SQLite binary directly from GitHub's LFS media CDN
    with streaming and exponential backoff.
    Wrapped in @st.cache_resource to run only once per container boot.
    """
    if _is_lfs_pointer_or_missing(CHROMA_SQLITE_PATH):
        with st.spinner("📦 First-boot setup: Hydrating pre-built ChromaDB vector database from Git LFS (~5–10s)..."):
            download_success = _download_lfs_binary_with_retries(
                dest_path=CHROMA_SQLITE_PATH,
                url=CHROMA_SQLITE_LFS_URL,
                max_retries=3,
                backoff_factor=2.0,
                timeout=60,
            )

        if not download_success or _is_lfs_pointer_or_missing(CHROMA_SQLITE_PATH):
            st.error(
                "❌ **Critical Deployment Error**: Failed to hydrate the pre-built ChromaDB database from Git LFS.\n\n"
                f"- **Expected File**: `{CHROMA_SQLITE_PATH}` (~157 MB with `b'SQLite format 3\\x00'`)\n"
                f"- **Current State**: {CHROMA_SQLITE_PATH.stat().st_size if CHROMA_SQLITE_PATH.exists() else 'Missing'} bytes\n\n"
                "Please check container outbound network access or rebuild the container."
            )
            st.stop()

    # Warm up ChromaDB collection on startup
    get_collection()
    return True


# Run startup readiness check
ensure_chroma_ready()

# -----------------------------------------------------------------------------
# Sidebar: Configuration & Inspector Panels
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    db_size_bytes = CHROMA_SQLITE_PATH.stat().st_size if CHROMA_SQLITE_PATH.exists() else 0
    db_status = f"{db_size_bytes / (1024 * 1024):.1f} MB (Hydrated)" if db_size_bytes >= MIN_VALID_SQLITE_BYTES else f"{db_size_bytes} bytes (LFS Pointer)"
    st.caption(f"🚀 **Build**: `048d3a1` | 🗄️ **ChromaDB**: `{db_status}`")

    # Retrieval Method Selector
    retrieval_choice = st.radio(
        label="Retrieval Architecture",
        options=["Vector-only (ChromaDB)", "Hybrid (BM25 + Vector RRF)"],
        index=0,  # Default to Vector-only per Phase 2 win criterion
        help="Select whether to retrieve using dense vector search only or hybrid BM25 + dense search with RRF fusion."
    )
    st.caption(
        "💡 **Default**: Vector-only was empirically selected over Hybrid in the "
        "Phase 2 benchmark (Recall@1 = 79.2% vs 75.0%, MRR = 0.812 vs 0.785)."
    )

    method_key = "vector" if "Vector-only" in retrieval_choice else "hybrid"

    st.markdown("---")

    # About Project Section
    with st.expander("ℹ️ About This Project", expanded=False):
        st.markdown(
            """
            **ML & Data Science Documentation RAG**
            
            - **Corpus**: Official documentation for **Pandas** (3.0.x), **Scikit-Learn** (1.9.x), and **XGBoost** (stable).
            - **Ingestion**: Custom paragraph-aware tokenizer chunking (500 tokens / 50 token overlap; `tiktoken` `cl100k_base`).
            - **Embeddings**: Local `all-MiniLM-L6-v2` vectors stored in persistent **ChromaDB**.
            - **Lexical Index**: `rank_bm25` (BM25Okapi) fused via Reciprocal Rank Fusion ($k=60$).
            - **Generation**: Direct **Groq API** calls enforcing mandatory `[n]` bracketed source citations.
            - **Auditing**: White-box lexical keyword overlap faithfulness heuristic.
            
            🔗 [GitHub Repository](https://github.com/AaruneshAP/Chatbot_rag)
            """
        )

    # Collapsible Latest Retrieved Chunks Inspector
    st.markdown("### 🔍 Retrieval Inspector")
    chunks_container = st.container()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_retrieved_chunks" not in st.session_state:
    st.session_state.latest_retrieved_chunks = []

# Update sidebar retrieved chunks inspector with latest query data
with chunks_container:
    with st.expander("Raw Retrieved Chunks (Top-5)", expanded=False):
        if st.session_state.latest_retrieved_chunks:
            for chunk in st.session_state.latest_retrieved_chunks:
                st.markdown(
                    f"**[{chunk.get('marker', '')}] `{chunk.get('source_library', 'unknown')}`** "
                    f"· `{chunk.get('page_or_section', 'unknown')}`"
                )
                st.caption(f"Chunk ID: `{chunk.get('chunk_id')}`")
                st.text_area(
                    label=f"Text for {chunk.get('marker', '')}",
                    value=chunk.get("full_text", ""),
                    height=100,
                    disabled=True,
                    key=f"sidebar_chunk_{chunk.get('citation_id')}"
                )
                st.divider()
        else:
            st.caption("No queries submitted yet. Ask a question to inspect retrieved chunks.")

# -----------------------------------------------------------------------------
# Main Chat Header & History Display
# -----------------------------------------------------------------------------
st.header("🤖 ML & Data Science Documentation Assistant")
st.markdown(
    "Ask technical questions about **Pandas**, **Scikit-Learn**, or **XGBoost**. "
    "Every response is strictly grounded in official documentation with mandatory source citations."
)

# Display historical messages from session state
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display model, citations, and faithfulness only for assistant responses
        if msg["role"] == "assistant":
            model_used = msg.get("model")
            failover_note = msg.get("failover_note")
            if model_used:
                caption = f"⚡ **Model**: `{model_used}`"
                if failover_note:
                    caption += f" *(🔄 {failover_note})*"
                st.caption(caption)

            if "cited_sources" in msg:
                cited_sources = msg["cited_sources"]
                faith = msg.get("faithfulness")

            # Faithfulness Metric Bar
            if faith:
                if faith.get("status") == "citation_format_not_recognized":
                    st.warning("⚠️ **Citation format not recognized**: No standard `[n]` citations were detected in this response.")
                col1, col2, col3 = st.columns(3)
                col1.metric("Faithfulness Rate", f"{faith['overall_faithfulness_rate']:.0%}")
                col2.metric("Mean Keyword Overlap", f"{faith['mean_overlap_score']:.0%}")
                col3.metric("Low Confidence Claims", f"{faith['low_confidence_sentences']}")

                with st.expander("🛡️ Faithfulness Audit Details", expanded=False):
                    st.caption(
                        "ℹ️ **Lexical Overlap Heuristic**: Checks keyword intersection between each claim and its cited chunk. "
                        "This is an explainable white-box proxy heuristic, not a full NLI entailment detector."
                    )
                    for s in faith.get("sentence_diagnostics", []):
                        flag_icon = "✅" if s["flag"] == "ok" else "⚠️" if s["flag"] == "low_confidence" else "❓"
                        st.markdown(f"{flag_icon} **Overlap: {s['overlap_score']:.0%}** | Citations: `{s['citations']}`")
                        st.markdown(f"> *{s['sentence']}*")

            # Expandable Source Citations
            if cited_sources:
                with st.expander(f"📚 Cited Sources ({len(cited_sources)} chunks)", expanded=False):
                    for src in cited_sources:
                        st.markdown(
                            f"**{src['marker']} [{src['source_library'].upper()}]** "
                            f"`{src['page_or_section']}`"
                        )
                        st.text(src.get("snippet", ""))

# -----------------------------------------------------------------------------
# Chat Input & Response Execution
# -----------------------------------------------------------------------------
if user_query := st.chat_input("Ask a question (e.g. 'What is the difference between .loc and .iloc?')..."):
    # 1. Display and save user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate response with clean error handling
    with st.chat_message("assistant"):
        with st.spinner(f"Retrieving context ({method_key}) and generating answer via Groq..."):
            try:
                gen_result = generate_answer(user_query, retrieval_method=method_key, top_k=5)
                answer_text = gen_result["answer"]
                cited_sources = gen_result["cited_sources"]

                # Run explainable faithfulness check
                faith_result = evaluate_faithfulness(answer_text, cited_sources)

                # Store latest retrieved chunks for sidebar inspector
                st.session_state.latest_retrieved_chunks = cited_sources

                # Display answer text
                st.markdown(answer_text)

                # Display generating model and failover indicator
                model_used = gen_result.get("model", "unknown")
                failover_note = gen_result.get("failover_note")
                caption = f"⚡ **Model**: `{model_used}`"
                if failover_note:
                    caption += f" *(🔄 {failover_note})*"
                st.caption(caption)

                # Display Faithfulness indicators
                if faith_result.get("status") == "citation_format_not_recognized":
                    st.warning("⚠️ **Citation format not recognized**: No standard `[n]` citations were detected in this response.")
                col1, col2, col3 = st.columns(3)
                col1.metric("Faithfulness Rate", f"{faith_result['overall_faithfulness_rate']:.0%}")
                col2.metric("Mean Keyword Overlap", f"{faith_result['mean_overlap_score']:.0%}")
                col3.metric("Low Confidence Claims", f"{faith_result['low_confidence_sentences']}")

                with st.expander("🛡️ Faithfulness Audit Details", expanded=False):
                    st.caption(
                        "ℹ️ **Lexical Overlap Heuristic**: Checks keyword intersection between each claim and its cited chunk. "
                        "This is an explainable white-box proxy heuristic, not a full NLI entailment detector."
                    )
                    for s in faith_result.get("sentence_diagnostics", []):
                        flag_icon = "✅" if s["flag"] == "ok" else "⚠️" if s["flag"] == "low_confidence" else "❓"
                        st.markdown(f"{flag_icon} **Overlap: {s['overlap_score']:.0%}** | Citations: `{s['citations']}`")
                        st.markdown(f"> *{s['sentence']}*")

                # Display expandable cited sources
                with st.expander(f"📚 Cited Sources ({len(cited_sources)} chunks)", expanded=False):
                    for src in cited_sources:
                        st.markdown(
                            f"**{src['marker']} [{src['source_library'].upper()}]** "
                            f"`{src['page_or_section']}`"
                        )
                        st.text(src.get("snippet", ""))

                # Save assistant message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "cited_sources": cited_sources,
                    "faithfulness": faith_result,
                    "model": model_used,
                    "failover_note": failover_note,
                })

            except ValueError as ve:
                st.error(f"⚠️ Configuration Error: {ve}")
            except Exception as exc:
                st.error(
                    f"⚠️ An error occurred during generation: {exc}\n\n"
                    "Please verify your `GROQ_API_KEY` in `.env` and check your network connection."
                )
