# Document Ingestion Pipeline for ML & Data Science Documentation RAG

A robust, self-contained document ingestion and retrieval pipeline designed for a RAG (Retrieval-Augmented Generation) system. This project automates the fetching, parsing, chunking, and vector indexing of official documentation for **Pandas**, **XGBoost**, and **Scikit-Learn**.

---

## 🚀 Key Features

- **Automated Multi-Source Fetching (`fetch_docs.py`)**: 
  - Dynamically inspects documentation hubs and endpoints with exponential backoff retry logic (3 attempts).
  - Fetches **XGBoost** as PDF from ReadTheDocs.
  - Fetches **Scikit-Learn** and **Pandas** as current stable zipped HTML documentation bundles (e.g., Pandas 3.0.x and Scikit-Learn 1.9.x).
  - Tracks all downloads in `data/raw/manifest.json`.
- **Heterogeneous Format Ingestion (`ingest.py`)**:
  - **PDF Extraction**: Extracts page-by-page text via `pdfplumber`, tracking exact page numbers as citation metadata.
  - **HTML Bundle Parsing**: Cleans Sphinx/PyData documentation HTML via `BeautifulSoup`, stripping headers, navigation sidebars, and boilerplates while preserving module hierarchy for both Pandas and Scikit-Learn.
- **Custom Tokenizer-Aware Chunking**:
  - **Paragraph-Aware Boundary Splitting**: Preserves coherent ideas before splitting.
  - **500 Target Tokens / 50 Token Overlap**: Token counting powered by `tiktoken` (`cl100k_base`).
  - **Zero Heavy Framework Bloat**: Custom algorithm built from scratch without LangChain.
- **Local Dense Vector Embeddings (`embed.py`)**: Vectorizes chunks using `sentence-transformers` (`all-MiniLM-L6-v2`) and persists them locally into **ChromaDB**.
- **Traceable Metadata**: Every vector is tagged with `source_library`, `page_or_section`, `doc_version`, `token_count`, and `chunk_id`.

---

## 📁 Repository Structure

```
Chatbot_rag/
├── data/
│   ├── raw/                      # Downloaded PDFs and HTML bundles
│   │   ├── xgboost.pdf           # XGBoost ReadTheDocs PDF export
│   │   ├── pandas.zip            # Pandas documentation ZIP
│   │   ├── pandas_html/          # Extracted Pandas HTML bundle
│   │   ├── scikit-learn-docs.zip # Scikit-Learn documentation ZIP
│   │   ├── sklearn_html/         # Extracted Scikit-Learn HTML bundle
│   │   └── manifest.json         # Download metadata & timestamps
│   ├── processed/
│   │   └── chunks.jsonl          # Standardized tagged chunks (13,116 records)
│   └── chroma_db/                # Persistent vector database
├── results/
│   └── benchmark_results.json    # Per-question evaluation metrics & rankings
├── eval_questions.json           # 24 fixed benchmark test questions & topic tags
├── fetch_docs.py                 # Source doc downloader with retry logic
├── ingest.py                     # Document parser & custom chunker
├── embed.py                      # Vectorizer & ChromaDB indexer
├── retrieval_vector.py           # Dense vector retriever (top-k)
├── retrieval_hybrid.py           # Hybrid BM25 + Vector retriever with RRF fusion
├── evaluate.py                   # Benchmark harness & comparative evaluation
├── requirements.txt              # Project dependencies
└── README.md                     # Architecture, benchmark evaluation & interview guide
```

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fetch Documentation
Downloads XGBoost PDF, Pandas HTML bundle, and Scikit-Learn HTML bundle:
```bash
python fetch_docs.py
```

### 3. Parse & Chunk Documents
Extracts text, applies paragraph-aware token chunking, and produces `data/processed/chunks.jsonl`:
```bash
python ingest.py
```

### 4. Generate Embeddings & Index into ChromaDB
Vectorizes all chunks using `all-MiniLM-L6-v2` and persists to `data/chroma_db/`:
```bash
python embed.py
```

---

## 🧠 Architectural & Design Decisions

### 1. Chunking Strategy: 500-Token Target with 50-Token Overlap

#### **Why 500 Tokens?**
- **Semantic Completeness**: Technical documentation (e.g., function parameter definitions, mathematical explanations of gradient boosting, code examples) requires sufficient context to be meaningful. A 100–200 token window frequently splits function signatures away from their parameter descriptions, causing retrieval degradation.
- **Embedding Model Alignment**: The `all-MiniLM-L6-v2` model supports up to a 512-token context window. Target chunking at 500 tokens maximizes the semantic density per embedding without truncating important details.
- **LLM Context Efficiency**: At retrieval time, injecting 3 to 5 chunks of ~500 tokens provides ~1,500–2,500 tokens of high-signal reference material to the LLM generation prompt, well within modern context budgets while avoiding "needle in a haystack" dilution.

#### **Why 50-Token Overlap?**
- **Boundary Continuity**: Splitting documents strictly at a hard token limit risks truncating a sentence or thought midway through. A 50-token (~10%) overlap ensures that boundary concepts are present in both adjacent chunks, avoiding context loss during similarity search.

#### **Why Paragraph-Aware Chunking (vs. Fixed Character Slicing)?**
- Naive fixed-character or fixed-token chunkers split sentences mid-phrase (e.g., cutting a formula or parameter name in half).
- Our custom chunker splits along natural paragraph boundaries (`\n\n` or section headers). It only falls back to token window slicing when an individual paragraph is exceptionally long (e.g., a massive docstring or code listing).

#### **Why Build a Custom Splitter Instead of Using LangChain?**
1. **Explainability & Transparency**: In portfolio reviews and technical interviews, understanding the exact token-slicing logic, boundary handling, and state machine is critical.
2. **Deterministic Token Counts**: LangChain's `CharacterTextSplitter` or `RecursiveCharacterTextSplitter` measure length by character count by default, leading to token estimate drift across languages and code. Our chunker uses `tiktoken` (`cl100k_base`) directly to guarantee strict token budgets.
3. **Zero Dependency Overhead**: Avoids heavy framework abstractions, dependency conflicts, and rapid breaking changes common in generic LLM orchestrators.

---

### 2. PDF vs. HTML Source Formats: Why the Difference Across Libraries?

During documentation ingestion, you will notice distinct distribution formats across the three libraries:

| Library | Distribution Format | Why This Format Exists | Ingestion & Extraction Strategy |
| :--- | :--- | :--- | :--- |
| **XGBoost** | PDF (ReadTheDocs) | XGBoost is hosted on ReadTheDocs, which automatically generates both HTML and compiled Sphinx LaTeX PDF exports for every release. | Extracted via `pdfplumber` page-by-page; metadata tracks exact `page_number`. |
| **Scikit-Learn** | Zipped HTML Bundle | Scikit-Learn maintains thousands of estimator classes, user guides, and auto-generated gallery examples. Generating a single monolithic LaTeX PDF causes compiler memory exhaustion and LaTeX overflow. As a result, Scikit-Learn distributes a zipped HTML archive instead of a PDF. | Extracted via `BeautifulSoup`; strips navbars, sidebars, search elements, and headers. Relative file paths (e.g. `modules/generated/sklearn.ensemble.RandomForestClassifier.html`) serve as section identifiers. |
| **Pandas** | Zipped HTML Bundle | Similar to Scikit-Learn, modern Pandas (v2.x / v3.x) removed monolithic PDF builds (>3,000 pages) and switched to distributing official zipped HTML bundles. If the legacy PDF URL 404s, the pipeline dynamically inspects the landing page and downloads `pandas.zip`. | Extracted via `BeautifulSoup` using the same unified HTML processing pipeline. |

#### **Normalizing Disparate Formats into a Unified Schema**
To allow seamless RAG retrieval regardless of whether the source was a PDF or HTML bundle, the ingestion pipeline standardizes every chunk into a unified JSON record:
```json
{
  "chunk_id": "sklearn_modules_generated_sklearn_ensemble_RandomForestClassifier_html_c0",
  "source_library": "sklearn",
  "source_format": "html",
  "page_or_section": "modules/generated/sklearn.ensemble.RandomForestClassifier.html",
  "doc_version": "1.9",
  "text": "class sklearn.ensemble.RandomForestClassifier(n_estimators=100, ...)...",
  "token_count": 487,
  "char_count": 2341
}
```

---

## 📊 Summary Statistics & Ingestion Metrics

When running `python ingest.py`, the pipeline aggregates corpus metrics:

- **Chunk Granularity**: Mean chunk size ~400–490 tokens, with 50-token sliding boundary preservation.
- **Traceability**: 100% of chunks maintain direct back-references to page numbers (for PDFs) or file paths (for HTML).

---

## 🔍 Semantic Search Retrieval Verification

When executing `python embed.py`, the system verifies retrieval against ChromaDB:

1. **Pandas Query**: *"How do you fill missing or null values in a DataFrame with fillna?"*  
   → Retrieves `DataFrame.fillna()` documentation from Pandas.
2. **XGBoost Query**: *"How does XGBoost handle missing values during tree split construction?"*  
   → Retrieves XGBoost split finding algorithm notes with default direction assignment.
3. **Scikit-Learn Query**: *"What hyperparameters control tree depth and regularization in RandomForestClassifier?"*  
   → Retrieves `RandomForestClassifier` parameter specifications (`max_depth`, `min_samples_split`, `ccp_alpha`).
---

## 🔬 Phase 2 Retrieval Benchmark: Vector-Only vs. Hybrid (BM25 + Vector)

A formal retrieval experiment comparing dense vector search against a hybrid BM25 + dense retrieval system with Reciprocal Rank Fusion (RRF).

### 🎯 Pre-Registered Win Criterion
> **Decision Rule**: *Hybrid retrieval is selected over vector-only if it achieves higher Recall@5 without a meaningful drop in Recall@1.*

### ⚙️ Benchmark Methodology & Setup
- **Evaluation Dataset (`eval_questions.json`)**: 24 reviewed technical questions partitioned equally across **Pandas (8)**, **Scikit-Learn (8)**, and **XGBoost (8)**, covering syntax, exact parameter flags, and conceptual mechanics.
- **Retriever Architectures**:
  1. **Vector-Only (`retrieval_vector.py`)**: Dense vector retrieval using `sentence-transformers/all-MiniLM-L6-v2` querying 13,116 persistent ChromaDB embeddings (top-5).
  2. **Hybrid (`retrieval_hybrid.py`)**: BM25Okapi (`rank_bm25`) + ChromaDB vector search (top-10 candidates each), fused with Reciprocal Rank Fusion (smoothing constant $k=60$):
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Vec}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- **Harness (`evaluate.py`)**: Executes both pipelines across all 24 questions, calculating Recall@1, Recall@3, Recall@5, Mean Reciprocal Rank (MRR), and topic-level breakdowns. Results are saved to `results/benchmark_results.json`.

> [!NOTE]
> **Evaluation Metric Limitation**: Recall@k uses a library-level routing proxy (evaluating whether a chunk from the `expected_library` is in the top-$k$ set). Exact gold chunk-level relevance is not evaluated because ground truth chunk IDs were not human-annotated.

### 📊 Comparative Performance Results

| Retrieval Method | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :--- | :--- | :--- | :--- |
| **Vector-Only (`all-MiniLM-L6-v2`)** | **0.7917** (79.2%) | **0.8333** (83.3%) | **0.8333** (83.3%) | **0.8125** |
| **Hybrid (BM25 + Vector RRF)** | 0.7500 (75.0%) | **0.8333** (83.3%) | **0.8333** (83.3%) | 0.7847 |

#### Topic-Level Recall@5 Breakdown
| Topic Tag | Questions | Vector Recall@5 | Hybrid Recall@5 |
| :--- | :--- | :--- | :--- |
| **binning / reshaping / duplicates / grouping / time series** | 5 | 100.0% | 100.0% |
| **indexing/selection (`.loc` vs `.iloc`)** | 1 | 100.0% | 100.0% |
| **preprocessing / cross-validation / pipelines / imputation** | 4 | 100.0% | 100.0% |
| **hyperparameter tuning / classifier outputs** | 2 | 100.0% | 100.0% |
| **training control (`early_stopping_rounds`, `eval_set`)** | 1 | 100.0% | 100.0% |
| **booster types (`gbtree` vs `gblinear`)** | 1 | 100.0% | 100.0% |
| **missing values** | 1 | 100.0% | 100.0% |
| **imbalanced classification** | 2 | 50.0% | 50.0% |
| **regularization (`subsample`, L1/L2)** | 2 | 50.0% | 50.0% |
| **tree complexity (`max_depth` vs `max_leaves`)** | 1 | 0.0% | 0.0% |
| **interpretability (`feature_importances_`)** | 1 | 0.0% | 0.0% |

### 🏆 Selection & Failure Mode Analysis

**Selected Architecture: Vector-Only Retrieval**

Under the pre-registered decision rule, hybrid retrieval **failed to win**: Recall@5 remained identical (83.33%), while Recall@1 dropped by 4.2% (75.0% vs. 79.2%) and MRR dropped from 0.8125 to 0.7847.

#### Hypotheses & Engineering Post-Mortem:
1. **Corpus Imbalance & Semantic Cannibalization**:
   - The corpus contains **6,248 Scikit-Learn chunks** (47.6%) and **6,051 Pandas chunks** (46.1%), but only **817 XGBoost chunks** (6.2%) — an **8:1 size disparity**.
   - Shared machine learning parameters (such as `subsample`, `max_depth`, `max_leaves`, and `feature_importances_`) exist in both Scikit-Learn and XGBoost. Because Scikit-Learn includes extensive narrative user guides, unconditioned queries on generic ML concepts pull Scikit-Learn chunks into the top-5 for both retrievers.
2. **Lexical Flooding in BM25**:
   - In Question 19 (*"How does XGBoost's built-in handling of missing values work?"*), Vector-Only correctly placed XGBoost at **Rank #1** by capturing the semantic intent of tree-split direction assignment.
   - However, BM25 flooded the candidate pool with Scikit-Learn chunks (`SimpleImputer`, `MissingIndicator`) due to high raw term frequencies of "missing values", displacing XGBoost's top chunk from Rank #1 down to Rank #3 in the fused ranking.
3. **Interview Takeaway**:
   - Hybrid search with simple BM25 is not an automatic silver bullet in multi-library RAG systems when corpus sizes are highly skewed. Lexical frequency without corpus-weight normalization or library-level metadata routing can degrade top-1 precision on shared technical vocabularies.

### ⚠️ Benchmark Limitations
- **Sample Size ($N=24$)**: The test suite comprises 24 curated questions (8 per library). Because several topic buckets contain only $n=1$ or $n=2$ queries, percentage metrics in the topic-level breakdown table should be interpreted as **illustrative rather than statistically robust**.
- **Library-Level Proxy Metric**: Recall@$k$ tracks whether the expected documentation library appears in the top-$k$ retrieved chunks. It does not measure paragraph-level semantic correctness against human gold standards.
- **Cross-Library Vocabulary Overlap**: Concepts like `feature_importances_`, `max_depth`, and `subsample` are valid across multiple frameworks (e.g. Scikit-Learn and XGBoost). When queries omit explicit framework keywords, a retriever retrieving high-quality chunks from an alternate valid library is penalized as a miss under strict single-label evaluation.

### 🔎 Case Study: Two Retrieval Misses (`q21` & `q24`)

A forensic audit of the two queries scoring 0% Recall@5 across both methods revealed critical insights into evaluation design and document extraction:

1. **Question 24 (`feature_importances_`) — Cross-Library Ambiguity**:
   - **Query**: *"How do you interpret feature_importances_ — what does the default importance type measure?"* (Tagged `expected_library: xgboost`).
   - **Diagnosis**: A valid, highly accurate answer was retrieved in Rank #1, but from **Scikit-Learn** (`sklearn_auto_examples_release_highlights_plot_release_highlights_1_4_0_html_c9`), explaining Gini / impurity-based importance. 
   - `feature_importances_` is a universal Scikit-Learn estimator convention also implemented by XGBoost's scikit-learn wrapper API. Because the query was unconditioned and Scikit-Learn documentation accounts for nearly half the total corpus, retrieving Scikit-Learn is a genuine cross-library ambiguity rather than a retrieval malfunction.
2. **Question 21 (`max_depth` vs `max_leaves`) — Cross-Library Overlap**:
   - **Query**: *"What's the difference between max_depth and max_leaves for controlling tree complexity?"* (Tagged `expected_library: xgboost`).
   - **Diagnosis**: Both vector and hybrid retrieval retrieved Rank #1 chunks from **Scikit-Learn** (`sklearn_auto_examples_ensemble_plot_monotonic_constraints_html_c8`), explaining tree depth constraints and maximum leaf nodes. Without library-explicit query conditioning, general tree complexity terminology naturally favors Scikit-Learn's richer tutorial documentation.
3. **PDF Extraction Hyphenation Artifact**:
   - Forensic analysis of `chunks.jsonl` revealed that `pdfplumber` extracted line-wrapped hyphenated words across line breaks in `xgboost.pdf` (for example, `fea-\nture_importances_` on page 199).
   - This hyphenation split exact programming identifiers, preventing BM25 from matching the exact token `feature_importances_` against XGBoost chunks while Scikit-Learn contained clean, unbroken HTML tokens.
   - **Resolution & Future Work**: A regex de-hyphenation preprocessor (`dehyphenate_text`) was implemented in `ingest.py` to heal line-wrapped hyphens (`\b\w+-\s*\n\s*\w+\b`) during extraction. While the original benchmark results are preserved to transparently reflect the initial evaluation run, resolving PDF line-wrap artifacts stands as a documented engineering improvement for future pipeline re-indexing.

---

## 💼 Interview Talking Points / Portfolio Defense

1. **How does this pipeline handle rate limits and network drops?**  
   `fetch_docs.py` implements exponential backoff (retrying up to 3 times) with a desktop User-Agent and chunked file streaming to avoid memory bottlenecks on multi-megabyte downloads.
2. **How does your chunking strategy impact retrieval precision?**  
   By preserving natural paragraph boundaries and maintaining a 50-token overlap, we prevent semantic cliffhangers where key definitions are truncated across chunk boundaries.
3. **Why use ChromaDB?**  
   ChromaDB provides a lightweight, embedded, persistent vector database that runs entirely in-process without requiring Docker or cloud infrastructure, ideal for reproducible local evaluation.
4. **How do you handle citations in downstream generation?**  
   Every chunk in ChromaDB retains metadata (`source_library`, `page_or_section`, `doc_version`), enabling the downstream RAG response generator to output verifiable citations (e.g. `[xgboost: page_42]` or `[pandas: reference/api/pandas.DataFrame.fillna.html]`).
