"""
Milestone 2 — Market Data Pipeline & Event-Window Label Construction
=====================================================================
Framework for ingesting high-frequency sovereign bond yield data,
computing event-window labels, and matching them to text chunks.

NOTE: This module provides the STRUCTURE. The user will plug in their
own tick data source. The pipeline handles:
1. Loading tick data from a standardized format
2. Event-window extraction around speech timestamps
3. Label computation (Δy₂ in basis points)
4. Label-to-chunk matching
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Format Specification
# =============================================================================

REQUIRED_TICK_COLUMNS = [
    "timestamp",       # UTC datetime
    "country",         # ISO 3-letter code (USA, GBR, DEU, etc.)
    "instrument",      # e.g., "2Y_YIELD", "10Y_YIELD", "EQUITY_INDEX"
    "mid_price",       # Mid price or yield value
]

EXPECTED_INSTRUMENTS = {
    "2Y_YIELD": "2-Year sovereign bond yield (primary label)",
    "10Y_YIELD": "10-Year sovereign bond yield (for slope)",
    "EQUITY_INDEX": "National equity index (secondary label)",
}


def validate_tick_data(df: pd.DataFrame) -> bool:
    """Validate that tick data conforms to required format."""
    missing_cols = set(REQUIRED_TICK_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Tick data missing required columns: {missing_cols}\n"
            f"Required: {REQUIRED_TICK_COLUMNS}\n"
            f"Got: {list(df.columns)}"
        )

    # Verify timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("'timestamp' column must be datetime64 type (UTC)")

    logger.info(
        f"Tick data validated: {len(df)} records, "
        f"{df['country'].nunique()} countries, "
        f"{df['instrument'].nunique()} instruments, "
        f"range: {df['timestamp'].min()} to {df['timestamp'].max()}"
    )
    return True


def load_tick_data(filepath: str) -> pd.DataFrame:
    """
    Load tick data from a Parquet or CSV file.

    Expected format (each row = one tick):
    | timestamp (UTC) | country | instrument | mid_price |

    Example:
    | 2022-09-21 18:00:01 | USA | 2Y_YIELD | 4.0125 |
    | 2022-09-21 18:00:01 | USA | EQUITY_INDEX | 3789.50 |
    """
    path = Path(filepath)

    if path.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif path.suffix == ".csv":
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    validate_tick_data(df)
    return df


# =============================================================================
# Event Window Extraction
# =============================================================================


def extract_event_window(
    tick_data: pd.DataFrame,
    event_time: pd.Timestamp,
    country: str,
    instrument: str = "2Y_YIELD",
    pre_minutes: int = 15,
    post_minutes: int = 60,
) -> Optional[pd.DataFrame]:
    """
    Extract tick data within the event window around a speech timestamp.

    Parameters
    ----------
    tick_data : pd.DataFrame
        Full tick dataset
    event_time : pd.Timestamp
        UTC timestamp of the speech/release
    country : str
        ISO country code for filtering
    instrument : str
        Financial instrument to extract
    pre_minutes : int
        Minutes before event to include
    post_minutes : int
        Minutes after event to include

    Returns
    -------
    pd.DataFrame or None
        Tick data within the event window, or None if insufficient data
    """
    window_start = event_time - pd.Timedelta(minutes=pre_minutes)
    window_end = event_time + pd.Timedelta(minutes=post_minutes)

    mask = (
        (tick_data["timestamp"] >= window_start)
        & (tick_data["timestamp"] <= window_end)
        & (tick_data["country"] == country)
        & (tick_data["instrument"] == instrument)
    )

    window_data = tick_data[mask].sort_values("timestamp").copy()

    if len(window_data) < 2:
        logger.debug(
            f"Insufficient data for {country}/{instrument} at {event_time}"
        )
        return None

    return window_data


# =============================================================================
# Label Computation
# =============================================================================


def compute_yield_change(
    window_data: pd.DataFrame,
    event_time: pd.Timestamp,
    pre_ref_minutes: int = 1,
    post_peak_minutes: int = 30,
) -> Optional[float]:
    """
    Compute the signed basis-point change in yield from pre-release
    reference to post-release peak within the specified window.

    Δy₂ = (peak post-release yield) - (last pre-release yield)
    Positive = hawkish surprise; Negative = dovish surprise

    Parameters
    ----------
    window_data : pd.DataFrame
        Tick data within the event window
    event_time : pd.Timestamp
        UTC timestamp of the speech/release
    pre_ref_minutes : int
        How many minutes before event to take reference price
    post_peak_minutes : int
        Window after event to search for peak price

    Returns
    -------
    float or None
        Yield change in basis points, or None if computation fails
    """
    # Pre-release reference: last observation before event
    pre_ref_start = event_time - pd.Timedelta(minutes=pre_ref_minutes)
    pre_data = window_data[
        (window_data["timestamp"] >= pre_ref_start)
        & (window_data["timestamp"] < event_time)
    ]

    if pre_data.empty:
        return None

    pre_yield = pre_data.iloc[-1]["mid_price"]

    # Post-release: find peak within window
    post_end = event_time + pd.Timedelta(minutes=post_peak_minutes)
    post_data = window_data[
        (window_data["timestamp"] >= event_time)
        & (window_data["timestamp"] <= post_end)
    ]

    if post_data.empty:
        return None

    # Signed peak: max absolute deviation from pre-yield
    post_yields = post_data["mid_price"].values
    deviations = post_yields - pre_yield

    # Use the deviation with the largest absolute value (preserving sign)
    max_abs_idx = np.argmax(np.abs(deviations))
    peak_deviation = deviations[max_abs_idx]

    # Convert to basis points (yields are typically in percentage, e.g., 4.25)
    delta_bp = peak_deviation * 100  # 0.01% = 1 bp

    return delta_bp


def compute_equity_change(
    window_data: pd.DataFrame,
    event_time: pd.Timestamp,
    pre_ref_minutes: int = 1,
    post_peak_minutes: int = 30,
) -> Optional[float]:
    """
    Compute percentage change in equity index from pre- to post-release.

    Returns
    -------
    float or None
        Percentage change, or None if computation fails
    """
    pre_ref_start = event_time - pd.Timedelta(minutes=pre_ref_minutes)
    pre_data = window_data[
        (window_data["timestamp"] >= pre_ref_start)
        & (window_data["timestamp"] < event_time)
    ]

    if pre_data.empty:
        return None

    pre_price = pre_data.iloc[-1]["mid_price"]

    post_end = event_time + pd.Timedelta(minutes=post_peak_minutes)
    post_data = window_data[
        (window_data["timestamp"] >= event_time)
        & (window_data["timestamp"] <= post_end)
    ]

    if post_data.empty:
        return None

    # Use the peak absolute deviation
    post_prices = post_data["mid_price"].values
    pct_changes = (post_prices - pre_price) / pre_price * 100

    max_abs_idx = np.argmax(np.abs(pct_changes))
    return pct_changes[max_abs_idx]


# =============================================================================
# Label Construction Pipeline
# =============================================================================


def build_event_labels(
    document_registry: pd.DataFrame,
    tick_data: pd.DataFrame,
    pre_minutes: int = 15,
    post_minutes: int = 60,
    pre_ref_minutes: int = 1,
    post_peak_minutes: int = 30,
) -> pd.DataFrame:
    """
    Build event-window labels for all documents in the registry.

    For each document with a matching timestamp and country in the tick data,
    compute Δy₂ (yield change in basis points) and secondary labels.

    Parameters
    ----------
    document_registry : pd.DataFrame
        Document registry with 'date', 'country' columns
    tick_data : pd.DataFrame
        Validated tick data

    Returns
    -------
    pd.DataFrame
        Labels indexed by doc_id
    """
    logger.info(
        f"Building event labels for {len(document_registry)} documents..."
    )

    labels = []

    for _, row in document_registry.iterrows():
        doc_id = row["doc_id"]
        country = row["country"]
        event_time = row["date"]  # NOTE: This is date-level; needs time precision

        # Extract yield window
        yield_window = extract_event_window(
            tick_data, event_time, country,
            instrument="2Y_YIELD",
            pre_minutes=pre_minutes,
            post_minutes=post_minutes,
        )

        delta_yield = None
        if yield_window is not None:
            delta_yield = compute_yield_change(
                yield_window, event_time,
                pre_ref_minutes=pre_ref_minutes,
                post_peak_minutes=post_peak_minutes,
            )

        # Extract equity window
        equity_window = extract_event_window(
            tick_data, event_time, country,
            instrument="EQUITY_INDEX",
            pre_minutes=pre_minutes,
            post_minutes=post_minutes,
        )

        delta_equity = None
        if equity_window is not None:
            delta_equity = compute_equity_change(
                equity_window, event_time,
                pre_ref_minutes=pre_ref_minutes,
                post_peak_minutes=post_peak_minutes,
            )

        labels.append({
            "doc_id": doc_id,
            "country": country,
            "event_time": event_time,
            "delta_yield_2y_bp": delta_yield,
            "delta_equity_pct": delta_equity,
            "has_yield_label": delta_yield is not None,
            "has_equity_label": delta_equity is not None,
        })

    labels_df = pd.DataFrame(labels)

    n_with_yield = labels_df["has_yield_label"].sum()
    n_with_equity = labels_df["has_equity_label"].sum()
    logger.info(
        f"Labels computed: {n_with_yield}/{len(labels_df)} have yield labels, "
        f"{n_with_equity}/{len(labels_df)} have equity labels"
    )

    return labels_df


def match_labels_to_chunks(
    chunk_registry: pd.DataFrame,
    event_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign event-window labels to text chunks via their parent document.

    All chunks from a given document inherit the event-level label
    of that document's speech.
    """
    # Merge on doc_id
    labeled_chunks = chunk_registry.merge(
        event_labels[["doc_id", "delta_yield_2y_bp", "delta_equity_pct"]],
        on="doc_id",
        how="left",
    )

    n_labeled = labeled_chunks["delta_yield_2y_bp"].notna().sum()
    logger.info(
        f"Label matching: {n_labeled}/{len(labeled_chunks)} chunks "
        f"({n_labeled / len(labeled_chunks) * 100:.1f}%) have yield labels"
    )

    return labeled_chunks


# =============================================================================
# Placeholder for user's data
# =============================================================================


def create_sample_tick_format() -> pd.DataFrame:
    """
    Create a small sample showing the expected tick data format.
    This helps the user understand what format to prepare their data in.
    """
    sample = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-09-21 17:55:00", "2022-09-21 17:58:00",
            "2022-09-21 18:00:00", "2022-09-21 18:05:00",
            "2022-09-21 18:15:00", "2022-09-21 18:30:00",
        ]),
        "country": ["USA"] * 6,
        "instrument": ["2Y_YIELD"] * 6,
        "mid_price": [4.0100, 4.0125, 4.0150, 4.0350, 4.0500, 4.0425],
    })
    return sample


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Show expected format
    sample = create_sample_tick_format()
    print("=== EXPECTED TICK DATA FORMAT ===")
    print(sample.to_string(index=False))
    print("\nSave your tick data in this format as data/market/tick_data.parquet")
    print("Columns: timestamp (UTC), country (ISO3), instrument, mid_price")
