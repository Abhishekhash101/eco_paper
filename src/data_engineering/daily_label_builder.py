"""
Phase 1 — Milestone 2 (Option A): Daily Event-Window Label Construction
=========================================================================
Uses daily sovereign bond yield data to compute Δy₂ labels for each
central bank speech in the CBS corpus.

Label definition (Option A — daily frequency):
  Δy₂ = US_2Y(t) - US_2Y(t-1)  in basis points
  where t = the date of the speech/FOMC event

This is the pragmatic alternative to intraday tick data.
Many published papers use daily yield changes (Gürkaynak et al., 2005).

Usage:
    python -m src.data_engineering.daily_label_builder
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yield_data(yields_path: str) -> pd.DataFrame:
    """Load daily sovereign bond yield data."""
    df = pd.read_csv(yields_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded yield data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def compute_daily_yield_changes(yields: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily basis-point changes for all yield columns.
    
    Returns DataFrame indexed by date with columns:
    - delta_US_2Y_bp: 1-day change in US 2Y yield (basis points)
    - delta_US_10Y_bp: 1-day change in US 10Y yield (basis points)
    - delta_slope_2s10s_bp: change in 2s10s curve slope
    - etc.
    """
    df = yields.copy()
    df = df.set_index("date").sort_index()
    
    changes = pd.DataFrame(index=df.index)
    
    # Primary label: US 2Y daily change (basis points)
    if "US_2Y" in df.columns:
        changes["delta_US_2Y_bp"] = df["US_2Y"].diff() * 100  # percentage points → basis points
    
    # Secondary: US 10Y
    if "US_10Y" in df.columns:
        changes["delta_US_10Y_bp"] = df["US_10Y"].diff() * 100
    
    # 2s10s slope change
    if "US_2Y" in df.columns and "US_10Y" in df.columns:
        slope = df["US_10Y"] - df["US_2Y"]
        changes["delta_slope_2s10s_bp"] = slope.diff() * 100
    
    # Euro Area
    if "EA_2Y" in df.columns:
        changes["delta_EA_2Y_bp"] = df["EA_2Y"].diff() * 100
    if "EA_10Y" in df.columns:
        changes["delta_EA_10Y_bp"] = df["EA_10Y"].diff() * 100
    
    # Germany
    if "DE_10Y" in df.columns:
        changes["delta_DE_10Y_bp"] = df["DE_10Y"].diff() * 100
    
    # UK
    if "GB_10Y" in df.columns:
        changes["delta_GB_10Y_bp"] = df["GB_10Y"].diff() * 100
    
    # Japan
    if "JP_10Y" in df.columns:
        changes["delta_JP_10Y_bp"] = df["JP_10Y"].diff() * 100
    
    logger.info(f"Computed yield changes: {len(changes)} days, {changes.columns.tolist()}")
    return changes


def load_document_registry(registry_path: str) -> pd.DataFrame:
    """Load the document registry from Phase 1."""
    df = pd.read_parquet(registry_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded document registry: {len(df)} documents")
    return df


def match_labels_to_documents(
    registry: pd.DataFrame,
    yield_changes: pd.DataFrame,
    primary_label: str = "delta_US_2Y_bp",
) -> pd.DataFrame:
    """
    Match yield change labels to each document by date.
    
    For each speech date, assigns the yield change on that day.
    If the speech falls on a non-trading day, uses the next available
    trading day's change.
    
    Parameters
    ----------
    registry : pd.DataFrame
        Document registry with 'date' column
    yield_changes : pd.DataFrame
        Daily yield changes indexed by date
    primary_label : str
        Column name for primary label
    
    Returns
    -------
    pd.DataFrame
        Registry with label columns appended
    """
    labeled = registry.copy()
    
    # For each document date, find the matching or next-available yield change
    trading_dates = yield_changes.index.sort_values()
    
    labels_matched = []
    for doc_date in labeled["date"]:
        # Find the nearest trading day on or after the speech date
        mask = trading_dates >= doc_date
        if mask.any():
            match_date = trading_dates[mask][0]
            row = yield_changes.loc[match_date]
            labels_matched.append(row)
        else:
            # No match (speech after last available yield data)
            labels_matched.append(pd.Series({col: np.nan for col in yield_changes.columns}))
    
    labels_df = pd.DataFrame(labels_matched, index=labeled.index)
    labeled = pd.concat([labeled, labels_df], axis=1)
    
    # Stats
    primary_valid = labeled[primary_label].notna().sum()
    primary_pct = primary_valid / len(labeled) * 100
    logger.info(
        f"Label matching: {primary_valid}/{len(labeled)} ({primary_pct:.1f}%) "
        f"documents have a valid {primary_label}"
    )
    
    if primary_valid > 0:
        stats = labeled[primary_label].describe()
        logger.info(
            f"Label distribution: mean={stats['mean']:.2f} bp, "
            f"std={stats['std']:.2f} bp, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}"
        )
    
    return labeled


def match_labels_to_chunks(
    chunk_registry: pd.DataFrame,
    labeled_docs: pd.DataFrame,
    label_columns: list = None,
) -> pd.DataFrame:
    """
    Propagate document-level labels to chunk-level.
    
    Each chunk inherits the label of its parent document.
    """
    if label_columns is None:
        label_columns = [
            "delta_US_2Y_bp", "delta_US_10Y_bp", 
            "delta_slope_2s10s_bp",
        ]
    
    # Available label columns
    available = [c for c in label_columns if c in labeled_docs.columns]
    
    # Build doc_id → labels mapping
    doc_labels = labeled_docs[["doc_id"] + available].copy()
    
    # Merge onto chunks
    chunks_labeled = chunk_registry.merge(
        doc_labels, on="doc_id", how="left"
    )
    
    primary = available[0] if available else None
    if primary:
        valid = chunks_labeled[primary].notna().sum()
        logger.info(f"Chunk labeling: {valid}/{len(chunks_labeled)} chunks have valid {primary}")
    
    return chunks_labeled


def run_daily_label_construction(
    yields_path: str = None,
    registry_path: str = None,
    chunk_path: str = None,
    output_dir: str = None,
) -> dict:
    """
    Full daily label construction pipeline.
    
    Returns dict with labeled_docs and labeled_chunks DataFrames.
    """
    if yields_path is None:
        yields_path = str(PROJECT_ROOT / "notebooks" / "sovereign_bond_yields.csv")
    if registry_path is None:
        registry_path = str(PROJECT_ROOT / "data" / "processed" / "document_registry.parquet")
    if chunk_path is None:
        chunk_path = str(PROJECT_ROOT / "data" / "processed" / "chunk_registry_paragraph.parquet")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "processed")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("PHASE 1 — M2: DAILY EVENT-WINDOW LABEL CONSTRUCTION (Option A)")
    print("=" * 70)
    
    # Step 1: Load yield data
    print("\n  Step 1: Loading yield data...")
    yields = load_yield_data(yields_path)
    
    # Step 2: Compute daily changes
    print("  Step 2: Computing daily yield changes...")
    yield_changes = compute_daily_yield_changes(yields)
    
    # Step 3: Load document registry
    print("  Step 3: Loading document registry...")
    registry = load_document_registry(registry_path)
    
    # Step 4: Match labels to documents
    print("  Step 4: Matching labels to documents...")
    labeled_docs = match_labels_to_documents(registry, yield_changes)
    
    # Step 5: Match labels to chunks
    print("  Step 5: Propagating labels to chunks...")
    chunks = pd.read_parquet(chunk_path)
    labeled_chunks = match_labels_to_chunks(chunks, labeled_docs)
    
    # Step 6: Save
    print("\n  Saving outputs...")
    docs_out = output_path / "labeled_document_registry.parquet"
    chunks_out = output_path / "labeled_chunk_registry.parquet"
    changes_out = output_path / "daily_yield_changes.parquet"
    
    labeled_docs.to_parquet(docs_out, index=False)
    labeled_chunks.to_parquet(chunks_out, index=False)
    yield_changes.to_parquet(changes_out)
    
    print(f"    ✓ {docs_out.name}")
    print(f"    ✓ {chunks_out.name}")
    print(f"    ✓ {changes_out.name}")
    
    # Summary
    primary = "delta_US_2Y_bp"
    valid_docs = labeled_docs[primary].notna().sum()
    valid_chunks = labeled_chunks[primary].notna().sum()
    
    print(f"\n  SUMMARY:")
    print(f"    Documents with labels: {valid_docs}/{len(labeled_docs)} ({valid_docs/len(labeled_docs)*100:.1f}%)")
    print(f"    Chunks with labels: {valid_chunks}/{len(labeled_chunks)} ({valid_chunks/len(labeled_chunks)*100:.1f}%)")
    
    if valid_docs > 0:
        stats = labeled_docs[primary].describe()
        print(f"    Δy₂ distribution: mean={stats['mean']:.2f} bp, std={stats['std']:.2f} bp")
        print(f"    Δy₂ range: [{stats['min']:.1f}, {stats['max']:.1f}] bp")
    
    print("\n" + "=" * 70)
    print("  PHASE 1 — M2 COMPLETE (Option A)")
    print("=" * 70)
    
    return {
        "labeled_docs": labeled_docs,
        "labeled_chunks": labeled_chunks,
        "yield_changes": yield_changes,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = run_daily_label_construction()
