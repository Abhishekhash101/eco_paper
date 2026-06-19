"""
Phase 1 — Complete Data Engineering Pipeline
=============================================
Orchestrates the full Phase 1 pipeline:
1. Load and filter CBS corpus (English-only)
2. Build document registry with deduplication
3. Segment into paragraph/sentence chunks
4. (When tick data available) Build event-window labels
5. Generate quality audit report

Usage:
    python -m src.data_engineering.run_phase1
    # or
    python src/data_engineering/run_phase1.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_engineering.corpus_loader import run_corpus_loading
from src.data_engineering.text_segmenter import run_segmentation

logger = logging.getLogger(__name__)


def run_phase1(
    rds_path: str = "CBS_dataset_v1.0.rds",
    output_dir: str = "data/processed",
    language: str = "English",
    min_text_length: int = 500,
    chunk_types: list = None,
    min_tokens: int = 20,
    max_tokens: int = 512,
):
    """
    Execute the full Phase 1 data engineering pipeline.

    Steps:
    1. Load CBS dataset and filter to English
    2. Build deduplicated document registry
    3. Segment documents into chunks
    4. Generate summary statistics
    """
    if chunk_types is None:
        chunk_types = ["paragraph"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Corpus Loading & Filtering
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 — STEP 1: CORPUS LOADING & FILTERING")
    print("=" * 70)

    registry = run_corpus_loading(
        rds_path=rds_path,
        language=language,
        min_text_length=min_text_length,
        output_dir=output_dir,
    )

    print(f"\n  ✓ {len(registry)} documents loaded")
    print(f"  ✓ {registry['central_bank'].nunique()} central banks")
    print(f"  ✓ {registry['country'].nunique()} countries")
    print(f"  ✓ Date range: {registry['date'].min().date()} to {registry['date'].max().date()}")

    # =========================================================================
    # Step 2: Text Segmentation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 — STEP 2: TEXT SEGMENTATION")
    print("=" * 70)

    chunk_results = run_segmentation(
        document_registry=registry,
        output_dir=output_dir,
        chunk_types=chunk_types,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    for ct, chunk_df in chunk_results.items():
        print(f"\n  ✓ {ct}: {len(chunk_df)} chunks")
        print(f"    avg tokens/chunk: {chunk_df['token_count'].mean():.0f}")
        print(f"    median tokens/chunk: {chunk_df['token_count'].median():.0f}")
        print(f"    chunks/document: {len(chunk_df) / len(registry):.1f} avg")

    # =========================================================================
    # Step 3: Quality Audit Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 — QUALITY AUDIT SUMMARY")
    print("=" * 70)

    # Document-level stats
    print("\n  DOCUMENT REGISTRY:")
    print(f"    Total documents: {len(registry)}")
    print(f"    Central banks: {registry['central_bank'].nunique()}")
    print(f"    Countries: {registry['country'].nunique()}")
    print(f"    Authors: {registry['author'].nunique()}")
    print(f"    Date range: {registry['date'].min().date()} — {registry['date'].max().date()}")
    print(f"    Avg text length: {registry['text_length'].mean():.0f} chars")
    print(f"    Total text: {registry['text_length'].sum() / 1e6:.1f}M chars")

    # Top central banks
    print("\n  TOP 10 CENTRAL BANKS (by speech count):")
    top_cbs = registry["central_bank"].value_counts().head(10)
    for cb, count in top_cbs.items():
        print(f"    {cb}: {count}")

    # Chunk-level stats
    for ct, chunk_df in chunk_results.items():
        print(f"\n  CHUNK REGISTRY ({ct.upper()}):")
        print(f"    Total chunks: {len(chunk_df)}")
        print(f"    Token count distribution:")
        print(f"      min: {chunk_df['token_count'].min()}")
        print(f"      25%: {chunk_df['token_count'].quantile(0.25):.0f}")
        print(f"      50%: {chunk_df['token_count'].quantile(0.50):.0f}")
        print(f"      75%: {chunk_df['token_count'].quantile(0.75):.0f}")
        print(f"      max: {chunk_df['token_count'].max()}")

    # =========================================================================
    # Step 4: Market Data Status
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 — MARKET DATA STATUS")
    print("=" * 70)

    market_dir = project_root / "data" / "market"
    tick_files = list(market_dir.glob("*.parquet")) + list(market_dir.glob("*.csv"))

    if tick_files:
        print(f"\n  ✓ Found {len(tick_files)} tick data file(s):")
        for f in tick_files:
            print(f"    {f.name}")
        print("\n  → Ready to run label construction.")
    else:
        print("\n  ⚠ No tick data found in data/market/")
        print("  → Place your tick data (Parquet or CSV) in data/market/")
        print("  → Required columns: timestamp, country, instrument, mid_price")
        print("  → Run: python -m src.data_engineering.market_data_pipeline")
        print("    for format details.")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"\n  Output directory: {output_path.resolve()}")
    print(f"  Files generated:")
    for f in sorted(output_path.glob("*.parquet")):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name} ({size_mb:.1f} MB)")

    print("\n  Next steps:")
    print("  1. Provide tick data → data/market/tick_data.parquet")
    print("  2. Run label construction")
    print("  3. Proceed to Phase 2 (Baselines)")

    return registry, chunk_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    registry, chunks = run_phase1(
        rds_path=str(project_root / "CBS_dataset_v1.0.rds"),
        output_dir=str(project_root / "data" / "processed"),
        language="English",
        min_text_length=500,
        chunk_types=["window"],  # Sentence-windows (~256 tokens, max 512)
        min_tokens=20,
        max_tokens=512,
    )
