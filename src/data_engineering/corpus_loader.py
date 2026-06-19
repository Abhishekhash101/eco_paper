"""
Milestone 1 — Step 1: Corpus Loading & Filtering
=================================================
Loads the CBS dataset (.rds), applies language and quality filters,
and produces a clean document registry.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyreadr

logger = logging.getLogger(__name__)


def load_rds_corpus(filepath: str) -> pd.DataFrame:
    """Load the CBS .rds file into a pandas DataFrame."""
    logger.info(f"Loading RDS file: {filepath}")
    result = pyreadr.read_r(filepath)
    df = list(result.values())[0]
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def filter_corpus(
    df: pd.DataFrame,
    language: str = "English",
    min_text_length: int = 500,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply filters to the raw CBS corpus.

    Parameters
    ----------
    df : pd.DataFrame
        Raw corpus from load_rds_corpus
    language : str
        Language filter (default: English)
    min_text_length : int
        Minimum character count for text field
    start_date : str, optional
        Earliest date to include (YYYY-MM-DD)
    end_date : str, optional
        Latest date to include (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        Filtered corpus
    """
    initial_count = len(df)

    # Language filter
    mask = df["Language"] == language
    df = df[mask].copy()
    logger.info(f"Language filter ({language}): {initial_count} -> {len(df)}")

    # Text length filter
    df["text_length"] = df["text"].str.len()
    df = df[df["text_length"] >= min_text_length].copy()
    logger.info(f"Min text length ({min_text_length}): -> {len(df)}")

    # Date parsing and filtering
    df["date_parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    null_dates = df["date_parsed"].isna().sum()
    if null_dates > 0:
        logger.warning(f"Dropping {null_dates} rows with unparseable dates")
        df = df.dropna(subset=["date_parsed"]).copy()

    if start_date:
        df = df[df["date_parsed"] >= pd.Timestamp(start_date)].copy()
        logger.info(f"Start date filter ({start_date}): -> {len(df)}")

    if end_date:
        df = df[df["date_parsed"] <= pd.Timestamp(end_date)].copy()
        logger.info(f"End date filter ({end_date}): -> {len(df)}")

    return df


def build_document_registry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a structured document registry from the filtered corpus.

    Each document gets:
    - doc_id: unique identifier (from Filename column)
    - content_hash: SHA-256 of text for deduplication
    - Metadata: central bank, country, date, author, role, etc.
    """
    logger.info("Building document registry...")

    registry = pd.DataFrame({
        "doc_id": df["Filename"],
        "title": df["Title"],
        "date": df["date_parsed"],
        "author": df["Authorname"],
        "role": df["Role"],
        "gender": df["Gender"],
        "central_bank": df["CentralBank"],
        "country": df["Country"],
        "source": df["Source"],
        "url": df["URL"],
        "text": df["text"],
        "text_length": df["text_length"],
        "content_hash": df["text"].apply(compute_content_hash),
    })

    # Check for duplicates by content hash
    dup_count = registry.duplicated(subset=["content_hash"]).sum()
    if dup_count > 0:
        logger.warning(
            f"Found {dup_count} duplicate documents by content hash. "
            "Keeping first occurrence."
        )
        registry = registry.drop_duplicates(subset=["content_hash"], keep="first")

    # Sort by date
    registry = registry.sort_values("date").reset_index(drop=True)

    logger.info(
        f"Document registry: {len(registry)} unique documents, "
        f"{registry['central_bank'].nunique()} central banks, "
        f"{registry['country'].nunique()} countries, "
        f"date range: {registry['date'].min().date()} to {registry['date'].max().date()}"
    )

    return registry


def run_corpus_loading(
    rds_path: str,
    language: str = "English",
    min_text_length: int = 500,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    Full corpus loading pipeline: load -> filter -> deduplicate -> save.

    Returns the document registry DataFrame.
    """
    # Load
    raw_df = load_rds_corpus(rds_path)

    # Filter
    filtered_df = filter_corpus(
        raw_df,
        language=language,
        min_text_length=min_text_length,
        start_date=start_date,
        end_date=end_date,
    )

    # Build registry
    registry = build_document_registry(filtered_df)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    registry_path = output_path / "document_registry.parquet"
    registry.to_parquet(registry_path, index=False)
    logger.info(f"Saved document registry to {registry_path}")

    return registry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    registry = run_corpus_loading(
        rds_path="CBS_dataset_v1.0.rds",
        language="English",
        min_text_length=500,
    )
    print(f"\nFinal registry: {len(registry)} documents")
    print(f"Central banks: {registry['central_bank'].nunique()}")
    print(f"Countries: {registry['country'].nunique()}")
    print(f"Date range: {registry['date'].min().date()} — {registry['date'].max().date()}")
