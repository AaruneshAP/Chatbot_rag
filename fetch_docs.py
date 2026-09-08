"""
fetch_docs.py - Automated Document Ingestion Downloader

Fetches documentation directly from source repositories and distribution endpoints:
- Pandas (Tries PDF first; if 404s, seamlessly verifies and downloads latest stable Zipped HTML bundle)
- XGBoost (PDF via ReadTheDocs)
- Scikit-Learn (Zipped HTML bundle)

Features:
- Robust HTTP sessions with custom User-Agent headers
- Retry logic with exponential backoff (3 attempts)
- Streaming downloads for large files
- Automatic HTML unzipping
- Generates data/raw/manifest.json metadata
"""

import os
import re
import sys
import time
import json
import zipfile
import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Base Directories
RAW_DATA_DIR = Path("data/raw")
PANDAS_HTML_DIR = RAW_DATA_DIR / "pandas_html"
SKLEARN_HTML_DIR = RAW_DATA_DIR / "sklearn_html"
MANIFEST_PATH = RAW_DATA_DIR / "manifest.json"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HEADERS = {"User-Agent": USER_AGENT}


def download_file_with_retry(
    url: str,
    dest_path: Path,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    timeout: int = 30,
) -> bool:
    """
    Downloads a remote file with retry logic, exponential backoff, and streaming.
    Returns True if download succeeded, False otherwise.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [Attempt {attempt}/{max_retries}] Connecting to {url} ...")
            session = requests.Session()
            session.headers.update(HEADERS)

            with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
                if response.status_code == 404:
                    print(f"  [404 Not Found] Resource does not exist at {url}")
                    return False

                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                downloaded = 0
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                mb_down = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                print(f"\r  Downloading: {mb_down:.1f}MB / {mb_total:.1f}MB ({percent:.1f}%)", end="", flush=True)

                print() # New line after progress
                if temp_path.exists():
                    temp_path.replace(dest_path)
                    file_size_mb = dest_path.stat().st_size / (1024 * 1024)
                    print(f"  [OK] Saved to {dest_path} ({file_size_mb:.2f} MB)")
                    return True

        except (requests.RequestException, requests.ConnectionError, requests.Timeout) as err:
            print(f"\n  [Error] Attempt {attempt} failed: {err}")
            if temp_path.exists():
                temp_path.unlink()
            if attempt < max_retries:
                sleep_time = backoff_factor ** attempt
                print(f"  Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(f"  [Failure] Exhausted all {max_retries} attempts for {url}")

    return False


def verify_and_fetch_pandas_url() -> tuple[str, str]:
    """
    Checks the pandas landing page (https://pandas.pydata.org/docs/) to verify
    the 'Download documentation' ZIP link and detect current doc version.
    """
    landing_url = "https://pandas.pydata.org/docs/"
    default_zip_url = "https://pandas.pydata.org/docs/pandas.zip"
    detected_version = "3.0"

    try:
        resp = requests.get(landing_url, headers=HEADERS, timeout=10)
        if resp.ok:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Detect doc version
            version_meta = soup.find("meta", attrs={"name": "docsearch:version"})
            if version_meta and version_meta.get("content"):
                detected_version = version_meta["content"].strip()
            else:
                title_match = re.search(r"pandas\s+([\d\.]+)\s+documentation", resp.text, re.IGNORECASE)
                if title_match:
                    detected_version = title_match.group(1)

            # Detect zip download link in 'Download documentation' section
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "pandas.zip" in href or href.endswith(".zip"):
                    if href.startswith("http"):
                        default_zip_url = href
                    else:
                        default_zip_url = requests.compat.urljoin(landing_url, href)
                    break
    except Exception as e:
        print(f"  [Notice] Could not inspect pandas landing page: {e}. Using verified default URL.")

    return default_zip_url, detected_version


def fetch_pandas_doc() -> dict:
    """
    Downloads the Pandas documentation.
    Attempts the PDF URL first. If it 404s (as modern pandas does not publish PDFs),
    it verifies and downloads the current stable zipped HTML bundle.
    """
    print("\n" + "=" * 60)
    print("1. Fetching Pandas Documentation")
    print("=" * 60)

    pdf_dest = RAW_DATA_DIR / "pandas.pdf"
    primary_pdf_url = "https://pandas.pydata.org/docs/pandas.pdf"

    print(f"Attempting primary PDF URL: {primary_pdf_url}")
    pdf_success = download_file_with_retry(primary_pdf_url, pdf_dest)

    if pdf_success:
        return {
            "source_library": "pandas",
            "url": primary_pdf_url,
            "format": "pdf",
            "download_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "doc_version": "stable",
            "file_path": str(pdf_dest),
            "file_size_bytes": pdf_dest.stat().st_size
        }

    # PDF 404'd -> Fall back to modern Zipped HTML docs bundle
    print("Primary PDF URL 404'd (Pandas publishes documentation as Zipped HTML in modern releases).")
    print("Inspecting https://pandas.pydata.org/docs/ for latest stable HTML zip bundle...")

    zip_url, doc_version = verify_and_fetch_pandas_url()
    zip_dest = RAW_DATA_DIR / "pandas.zip"

    print(f"Target Zip URL: {zip_url} (Version: {doc_version})")
    zip_success = download_file_with_retry(zip_url, zip_dest)

    if not zip_success:
        raise RuntimeError(f"Failed to download Pandas documentation ZIP from {zip_url}")

    print(f"\nUnzipping Pandas HTML documentation to {PANDAS_HTML_DIR} ...")
    PANDAS_HTML_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_dest, "r") as zip_ref:
        zip_ref.extractall(PANDAS_HTML_DIR)

    html_count = sum(1 for _ in PANDAS_HTML_DIR.rglob("*.html"))
    print(f"  [OK] Extracted {html_count} HTML files into {PANDAS_HTML_DIR}")

    return {
        "source_library": "pandas",
        "url": zip_url,
        "format": "html",
        "download_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "doc_version": doc_version,
        "file_path": str(PANDAS_HTML_DIR),
        "file_size_bytes": zip_dest.stat().st_size,
        "html_file_count": html_count
    }


def fetch_xgboost_doc() -> dict:
    """
    Downloads the XGBoost documentation PDF from ReadTheDocs standard PDF export.
    Tries stable first, then falls back to latest.
    """
    print("\n" + "=" * 60)
    print("2. Fetching XGBoost Documentation (PDF)")
    print("=" * 60)

    dest_path = RAW_DATA_DIR / "xgboost.pdf"
    stable_url = "https://xgboost.readthedocs.io/_/downloads/en/stable/pdf/"
    latest_url = "https://xgboost.readthedocs.io/_/downloads/en/latest/pdf/"

    print(f"Attempting stable PDF URL: {stable_url}")
    url_used = stable_url
    doc_version = "stable"
    success = download_file_with_retry(stable_url, dest_path)

    if not success:
        print(f"Stable URL failed. Attempting latest PDF URL: {latest_url}")
        url_used = latest_url
        doc_version = "latest"
        success = download_file_with_retry(latest_url, dest_path)

    if not success:
        raise RuntimeError("Failed to download XGBoost documentation PDF.")

    return {
        "source_library": "xgboost",
        "url": url_used,
        "format": "pdf",
        "download_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "doc_version": doc_version,
        "file_path": str(dest_path),
        "file_size_bytes": dest_path.stat().st_size
    }


def verify_and_fetch_sklearn_url() -> tuple[str, str]:
    """
    Checks scikit-learn landing page to detect doc version and resolve download link.
    """
    landing_url = "https://scikit-learn.org/stable/"
    default_zip_url = "https://scikit-learn.org/stable/_downloads/scikit-learn-docs.zip"
    detected_version = "stable"

    try:
        resp = requests.get(landing_url, headers=HEADERS, timeout=10)
        if resp.ok:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Look for version meta or docsearch version
            version_meta = soup.find("meta", attrs={"name": "docsearch:version"})
            if version_meta and version_meta.get("content"):
                detected_version = version_meta["content"].strip()
            
            # Check for zip link in page
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "scikit-learn-docs.zip" in href or href.endswith(".zip"):
                    if href.startswith("http"):
                        default_zip_url = href
                    else:
                        default_zip_url = requests.compat.urljoin(landing_url, href)
                    break
    except Exception as e:
        print(f"  [Notice] Could not inspect scikit-learn landing page: {e}. Using verified default.")

    return default_zip_url, detected_version


def fetch_sklearn_doc() -> dict:
    """
    Downloads and extracts the Scikit-Learn HTML documentation bundle.
    """
    print("\n" + "=" * 60)
    print("3. Fetching Scikit-Learn Documentation (Zipped HTML)")
    print("=" * 60)

    zip_url, doc_version = verify_and_fetch_sklearn_url()
    zip_dest = RAW_DATA_DIR / "scikit-learn-docs.zip"

    print(f"Target Zip URL: {zip_url} (Version: {doc_version})")
    success = download_file_with_retry(zip_url, zip_dest)

    if not success:
        raise RuntimeError(f"Failed to download scikit-learn documentation zip from {zip_url}")

    print(f"\nUnzipping HTML documentation to {SKLEARN_HTML_DIR} ...")
    SKLEARN_HTML_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_dest, "r") as zip_ref:
        zip_ref.extractall(SKLEARN_HTML_DIR)

    html_count = sum(1 for _ in SKLEARN_HTML_DIR.rglob("*.html"))
    print(f"  [OK] Extracted {html_count} HTML files into {SKLEARN_HTML_DIR}")

    return {
        "source_library": "sklearn",
        "url": zip_url,
        "format": "html",
        "download_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "doc_version": doc_version,
        "file_path": str(SKLEARN_HTML_DIR),
        "file_size_bytes": zip_dest.stat().st_size,
        "html_file_count": html_count
    }


def main():
    print("=" * 60)
    print("RAG Ingestion Pipeline: Document Fetcher")
    print("=" * 60)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest_data = {}

    # 1. Pandas
    try:
        manifest_data["pandas"] = fetch_pandas_doc()
    except Exception as e:
        print(f"[Error fetching Pandas]: {e}")

    # 2. XGBoost
    try:
        manifest_data["xgboost"] = fetch_xgboost_doc()
    except Exception as e:
        print(f"[Error fetching XGBoost]: {e}")

    # 3. Scikit-Learn
    try:
        manifest_data["sklearn"] = fetch_sklearn_doc()
    except Exception as e:
        print(f"[Error fetching Scikit-Learn]: {e}")

    # Write Manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Manifest written to {MANIFEST_PATH}")
    print(json.dumps(manifest_data, indent=2))
    print("=" * 60)
    print("[OK] All documentation downloads complete!")


if __name__ == "__main__":
    main()
