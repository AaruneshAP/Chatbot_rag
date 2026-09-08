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
from typing import List, Dict, Any

from generate import generate_answer
from faithfulness import evaluate_faithfulness

# Ensure UTF-8 console encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from embed import generate_and_store_embeddings, load_chunks, CHROMA_DB_DIR, CHUNKS_PATH

# Page Configuration
st.set_page_config(
    page_title="Data Science Docs RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner=False)
def ensure_chroma_ready() -> bool:
    """
    Ensures ChromaDB vector store is populated.
    If deployed on a fresh container (e.g. Streamlit Community Cloud)
    where data/chroma_db does not yet exist, builds it from data/processed/chunks.jsonl.
    """
    sqlite_file = CHROMA_DB_DIR / "chroma.sqlite3"
    if CHROMA_DB_DIR.exists() and sqlite_file.exists():
        return True

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Processed chunks file not found at {CHUNKS_PATH}. "
            "Please ensure data/processed/chunks.jsonl is present in the repository."
        )

    with st.spinner("📦 First-time deployment setup: Building ChromaDB vector database from chunks.jsonl (~1.5–2 min cold boot on CPU)..."):
        chunks = load_chunks(CHUNKS_PATH)
        generate_and_store_embeddings(chunks, chroma_dir=CHROMA_DB_DIR)
    return True


# Run startup readiness check
ensure_chroma_ready()

# -----------------------------------------------------------------------------
# Sidebar: Configuration & Inspector Panels
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG Configuration")

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

        # Display citations and faithfulness only for assistant responses
        if msg["role"] == "assistant" and "cited_sources" in msg:
            cited_sources = msg["cited_sources"]
            faith = msg.get("faithfulness")

            # Faithfulness Metric Bar
            if faith:
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

                # Display Faithfulness indicators
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
                    "faithfulness": faith_result
                })

            except ValueError as ve:
                st.error(f"⚠️ Configuration Error: {ve}")
            except Exception as exc:
                st.error(
                    f"⚠️ An error occurred during generation: {exc}\n\n"
                    "Please verify your `GROQ_API_KEY` in `.env` and check your network connection."
                )
