"""
ingest.py - Document Extraction, Normalization & Custom Token-Aware Chunking

Processes heterogeneous raw documentation (PDFs and HTML bundles) into a clean, 
standardized corpus of chunked documents ready for vector embeddings.

Key Features:
- PDF text extraction via pdfplumber (preserving exact page numbers)
- Unified HTML boilerplate stripping via BeautifulSoup (supporting Pandas and Scikit-Learn Sphinx bundles)
- Custom paragraph-aware token chunking (500 tokens target, 50 tokens overlap)
- Token counting using tiktoken (cl100k_base)
- Serializes chunks to data/processed/chunks.jsonl
- Detailed statistical reporting on chunk distribution and formats
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
from bs4 import BeautifulSoup
import tiktoken
from tqdm import tqdm

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = RAW_DIR / "manifest.json"
CHUNKS_OUTPUT_PATH = PROCESSED_DIR / "chunks.jsonl"

TARGET_CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
TOKENIZER_ENCODING = "cl100k_base"


class CustomTokenChunker:
    """
    Custom tokenizer-driven chunker that prioritizes natural paragraph boundaries
    and falls back to sliding token windows with exact overlap.
    """

    def __init__(self, target_tokens: int = TARGET_CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS, encoding_name: str = TOKENIZER_ENCODING):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, disallowed_special=())

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits raw text into chunks aiming for target_tokens with overlap_tokens.
        
        Strategy:
        1. Clean and normalize whitespace.
        2. Split text on natural paragraph boundaries (\n\n or \n).
        3. Accumulate paragraphs until target_tokens limit is reached.
        4. If a single paragraph exceeds target_tokens, slice it with a sliding token window.
        5. Maintain overlap across adjacent chunks by prepending the tail tokens of the
           previous chunk to the next chunk.
        """
        cleaned_text = re.sub(r"\r\n", "\n", text).strip()
        if not cleaned_text:
            return []

        # Split into paragraph units
        raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned_text) if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [p.strip() for p in cleaned_text.split("\n") if p.strip()]

        chunks: List[str] = []
        current_chunk_paragraphs: List[str] = []
        current_chunk_tokens = 0
        overlap_prefix_tokens: List[int] = []

        for para in raw_paragraphs:
            para_tokens = self.encode(para)
            para_token_len = len(para_tokens)

            # If a single paragraph exceeds the maximum target, chunk the paragraph itself with sliding window
            if para_token_len > self.target_tokens:
                # Flush existing accumulated paragraphs first
                if current_chunk_paragraphs:
                    chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                    if chunk_str:
                        chunks.append(chunk_str)
                        all_tokens = self.encode(chunk_str)
                        overlap_prefix_tokens = all_tokens[-self.overlap_tokens:] if len(all_tokens) > self.overlap_tokens else all_tokens
                    current_chunk_paragraphs = []
                    current_chunk_tokens = len(overlap_prefix_tokens)

                # Sliding window over the oversized paragraph
                step = self.target_tokens - self.overlap_tokens
                for i in range(0, para_token_len, step):
                    window_tokens = para_tokens[i : i + self.target_tokens]
                    # Prepend overlap if at the start and overlap exists
                    if i == 0 and overlap_prefix_tokens:
                        combined_tokens = overlap_prefix_tokens + window_tokens
                        if len(combined_tokens) > self.target_tokens:
                            combined_tokens = combined_tokens[: self.target_tokens]
                        chunk_str = self.decode(combined_tokens).strip()
                    else:
                        chunk_str = self.decode(window_tokens).strip()

                    if chunk_str:
                        chunks.append(chunk_str)
                    
                    if i + self.target_tokens >= para_token_len:
                        overlap_prefix_tokens = window_tokens[-self.overlap_tokens:] if len(window_tokens) > self.overlap_tokens else window_tokens
                        current_chunk_tokens = len(overlap_prefix_tokens)
                        break
                continue

            # Check if adding this paragraph exceeds target chunk size
            if current_chunk_tokens + para_token_len > self.target_tokens and current_chunk_paragraphs:
                # Emit current chunk
                chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                    all_tokens = self.encode(chunk_str)
                    overlap_prefix_tokens = all_tokens[-self.overlap_tokens:] if len(all_tokens) > self.overlap_tokens else all_tokens

                # Reset for next chunk with overlap prefix decoded
                current_chunk_paragraphs = []
                if overlap_prefix_tokens:
                    current_chunk_paragraphs.append(self.decode(overlap_prefix_tokens).strip())
                    current_chunk_tokens = len(overlap_prefix_tokens)
                else:
                    current_chunk_tokens = 0

            current_chunk_paragraphs.append(para)
            current_chunk_tokens += para_token_len

        # Emit any remaining paragraphs
        if current_chunk_paragraphs:
            chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
            if chunk_str and (not chunks or chunk_str != chunks[-1]):
                chunks.append(chunk_str)

        return chunks


def load_manifest() -> Dict[str, Any]:
    """Loads manifest.json metadata if available."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read manifest: {e}")
    return {}


def dehyphenate_text(text: str) -> str:
    """
    Joins line-wrapped hyphenated words produced by PDF extractors
    (e.g., 'fea-\\nture_importances_' -> 'feature_importances_', 'word-\\nword' -> 'wordword').
    Preserves valid mid-sentence hyphens and programmatic underscores.
    """
    return re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", text)


def process_pdf_file(
    pdf_path: Path,
    source_library: str,
    doc_version: str,
    chunker: CustomTokenChunker,
) -> List[Dict[str, Any]]:
    """
    Extracts pages from a PDF file using pdfplumber, keeping page numbers as metadata,
    and returns a list of tagged chunk records.
    """
    if not pdf_path.exists():
        print(f"  [Warning] PDF file not found: {pdf_path}")
        return []

    print(f"\nProcessing PDF: {pdf_path.name} ({source_library.upper()})...")
    chunks_out: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"  Total Pages: {total_pages}")

        for page_idx, page in enumerate(tqdm(pdf.pages, desc=f"  Extracting {source_library}"), start=1):
            try:
                page_text = page.extract_text() or ""
                # De-hyphenate line-wrapped words from PDF layout
                page_text = dehyphenate_text(page_text)
                # Basic cleaning
                page_text = page_text.strip()
                if len(page_text) < 40: # Skip blank / cover / sparse pages
                    continue

                page_chunks = chunker.chunk_text(page_text)
                for c_idx, c_text in enumerate(page_chunks):
                    token_count = chunker.count_tokens(c_text)
                    chunk_record = {
                        "chunk_id": f"{source_library}_p{page_idx}_c{c_idx}",
                        "source_library": source_library,
                        "source_format": "pdf",
                        "page_or_section": f"page_{page_idx}",
                        "doc_version": doc_version,
                        "text": c_text,
                        "token_count": token_count,
                        "char_count": len(c_text)
                    }
                    chunks_out.append(chunk_record)
            except Exception as e:
                print(f"  [Error] Failed processing page {page_idx} of {pdf_path.name}: {e}")

    print(f"  [OK] Created {len(chunks_out)} chunks from {source_library.upper()} PDF.")
    return chunks_out


def clean_html_soup(soup: BeautifulSoup) -> str:
    """
    Strips navbars, headers, footers, sidebars, search bars, scripts, styles,
    and returns clean text from primary content containers.
    Works for both PyData and Sphinx theme structures.
    """
    # Remove unwanted tags completely
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "dialog", "noscript", "svg"]):
        tag.decompose()

    # Remove Sphinx / PyData theme navigation & boilerplate classes
    unwanted_classes = [
        "bd-header", "bd-sidebar", "bd-sidebar-primary", "bd-sidebar-secondary",
        "pst-skip-link", "pst-async-banner-revealer", "sphx-glr-download-link-note",
        "prev-next-bottom", "footer", "navbar", "search-dialog", "toc-drawer"
    ]
    for cls in unwanted_classes:
        for el in soup.find_all(class_=cls):
            el.decompose()

    # Priority search for main content containers
    main_content = (
        soup.find("article", class_="bd-article") or
        soup.find("div", attrs={"role": "main"}) or
        soup.find("main") or
        soup.find("div", class_="document") or
        soup.find("div", class_="section") or
        soup.body
    )

    if not main_content:
        return ""

    # Extract cleaned text
    text = main_content.get_text(separator="\n", strip=True)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_html_bundle(
    html_root: Path,
    source_library: str,
    doc_version: str,
    chunker: CustomTokenChunker,
) -> List[Dict[str, Any]]:
    """
    Recursively processes extracted HTML documentation files (Pandas or Scikit-Learn),
    strips boilerplate, and creates tagged chunks with relative HTML path as section identifiers.
    """
    if not html_root.exists():
        print(f"  [Warning] HTML directory not found: {html_root}")
        return []

    print(f"\nProcessing HTML Bundle: {html_root.name} ({source_library.upper()})...")
    html_files = [
        p for p in html_root.rglob("*.html")
        if not any(part.startswith(("_", ".")) for part in p.relative_to(html_root).parts)
        and p.name not in {"search.html", "genindex.html", "py-modindex.html"}
    ]

    print(f"  Found {len(html_files)} content HTML pages.")
    chunks_out: List[Dict[str, Any]] = []

    for html_path in tqdm(html_files, desc=f"  Extracting {source_library} HTML"):
        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            text = clean_html_soup(soup)
            if len(text) < 40:
                continue

            rel_section = str(html_path.relative_to(html_root)).replace("\\", "/")
            safe_section_id = re.sub(r"[^a-zA-Z0-9_]", "_", rel_section)

            file_chunks = chunker.chunk_text(text)
            for c_idx, c_text in enumerate(file_chunks):
                token_count = chunker.count_tokens(c_text)
                chunk_record = {
                    "chunk_id": f"{source_library}_{safe_section_id}_c{c_idx}",
                    "source_library": source_library,
                    "source_format": "html",
                    "page_or_section": rel_section,
                    "doc_version": doc_version,
                    "text": c_text,
                    "token_count": token_count,
                    "char_count": len(c_text)
                }
                chunks_out.append(chunk_record)

        except Exception as e:
            print(f"  [Error] Failed processing {html_path.name}: {e}")

    print(f"  [OK] Created {len(chunks_out)} chunks from {source_library.upper()} HTML bundle.")
    return chunks_out


def print_summary_statistics(all_chunks: List[Dict[str, Any]]):
    """
    Calculates and displays summary statistics of the processed corpus.
    """
    total_chunks = len(all_chunks)
    if total_chunks == 0:
        print("\n[Warning] No chunks were produced.")
        return

    # Grouping by library
    stats_by_lib: Dict[str, Dict[str, Any]] = {}
    for c in all_chunks:
        lib = c["source_library"]
        fmt = c["source_format"]
        tokens = c["token_count"]
        chars = c["char_count"]

        if lib not in stats_by_lib:
            stats_by_lib[lib] = {
                "format": fmt,
                "count": 0,
                "total_tokens": 0,
                "total_chars": 0,
                "version": c.get("doc_version", "unknown")
            }
        stats_by_lib[lib]["count"] += 1
        stats_by_lib[lib]["total_tokens"] += tokens
        stats_by_lib[lib]["total_chars"] += chars

    overall_tokens = sum(c["token_count"] for c in all_chunks)
    overall_chars = sum(c["char_count"] for c in all_chunks)
    avg_tokens_overall = overall_tokens / total_chunks if total_chunks else 0
    avg_chars_overall = overall_chars / total_chunks if total_chunks else 0

    print("\n" + "=" * 70)
    print("DOCUMENT INGESTION & CHUNKING SUMMARY STATISTICS")
    print("=" * 70)
    print(f"{'Source Library':<15} | {'Format':<8} | {'Version':<10} | {'Chunks':<8} | {'Avg Tokens':<12} | {'Avg Chars':<10}")
    print("-" * 70)

    for lib, data in sorted(stats_by_lib.items()):
        count = data["count"]
        avg_tok = data["total_tokens"] / count if count else 0
        avg_ch = data["total_chars"] / count if count else 0
        print(f"{lib:<15} | {data['format'].upper():<8} | {data['version']:<10} | {count:<8} | {avg_tok:<12.1f} | {avg_ch:<10.1f}")

    print("-" * 70)
    print(f"{'TOTAL / OVERALL':<15} | {'MIXED':<8} | {'-':<10} | {total_chunks:<8} | {avg_tokens_overall:<12.1f} | {avg_chars_overall:<10.1f}")
    print("=" * 70 + "\n")


def main():
    print("=" * 60)
    print("RAG Ingestion Pipeline: Ingest & Chunking Engine")
    print("=" * 60)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    chunker = CustomTokenChunker(
        target_tokens=TARGET_CHUNK_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
        encoding_name=TOKENIZER_ENCODING
    )

    all_chunks: List[Dict[str, Any]] = []

    # 1. Pandas (HTML bundle or PDF)
    pandas_html_path = RAW_DIR / "pandas_html"
    pandas_pdf_path = RAW_DIR / "pandas.pdf"
    pandas_version = manifest.get("pandas", {}).get("doc_version", "3.0")

    if pandas_html_path.exists():
        all_chunks.extend(process_html_bundle(pandas_html_path, "pandas", pandas_version, chunker))
    elif pandas_pdf_path.exists():
        all_chunks.extend(process_pdf_file(pandas_pdf_path, "pandas", pandas_version, chunker))

    # 2. XGBoost PDF
    xgboost_path = RAW_DIR / "xgboost.pdf"
    xgboost_version = manifest.get("xgboost", {}).get("doc_version", "stable")
    if xgboost_path.exists():
        all_chunks.extend(process_pdf_file(xgboost_path, "xgboost", xgboost_version, chunker))

    # 3. Scikit-Learn HTML Bundle
    sklearn_html_path = RAW_DIR / "sklearn_html"
    sklearn_version = manifest.get("sklearn", {}).get("doc_version", "stable")
    if sklearn_html_path.exists():
        all_chunks.extend(process_html_bundle(sklearn_html_path, "sklearn", sklearn_version, chunker))

    # Save to chunks.jsonl
    print(f"\nWriting {len(all_chunks)} chunks to {CHUNKS_OUTPUT_PATH} ...")
    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {CHUNKS_OUTPUT_PATH} ({CHUNKS_OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # Print Summary Statistics
    print_summary_statistics(all_chunks)


if __name__ == "__main__":
    main()
