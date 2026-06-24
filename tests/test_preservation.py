"""
Property-Based Preservation Tests
==================================
Validates that analytical functions in src/ produce correct outputs.
These tests establish baseline behavior BEFORE the notebook fix is applied.

Uses hypothesis for property-based testing to verify:
- LM lexicon scoring produces valid bounded scores
- Hawk-Dove scoring produces valid scores with correct net formula
- Content hashing is deterministic and produces valid hex strings
- Corpus filtering never increases row count
- Paragraph segmentation respects minimum length constraint

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**
"""

import sys
from pathlib import Path

# Add project root to path so src/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.data_engineering.corpus_loader import compute_content_hash, filter_corpus, build_document_registry
from src.data_engineering.text_segmenter import segment_into_paragraphs
from src.baselines.lm_lexicon import compute_lm_scores
from src.baselines.hawk_dove_lexicon import compute_hd_scores


# ============================================================================
# Strategy helpers
# ============================================================================

# Text strategy: printable strings that may contain words
text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z', 'S')),
    min_size=0,
    max_size=500
)

# Non-empty text with at least one word token
nonempty_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
    min_size=1,
    max_size=500
).filter(lambda t: len(t.strip()) > 0)


# ============================================================================
# Property 1: LM Lexicon scores are bounded correctly
# ============================================================================

@given(text=text_strategy)
@settings(max_examples=200)
def test_lm_scores_bounded(text):
    """
    Property: For all random text strings, compute_lm_scores(text) returns
    float values where lm_neg, lm_pos, lm_uncertainty in [0.0, 1.0]
    and lm_net in [-1.0, 1.0].

    **Validates: Requirements 3.4**
    """
    scores = compute_lm_scores(text)

    # Check all expected keys are present
    assert "lm_neg" in scores
    assert "lm_pos" in scores
    assert "lm_net" in scores
    assert "lm_uncertainty" in scores
    assert "token_count" in scores

    # Bounded checks
    assert 0.0 <= scores["lm_neg"] <= 1.0, f"lm_neg={scores['lm_neg']} out of bounds"
    assert 0.0 <= scores["lm_pos"] <= 1.0, f"lm_pos={scores['lm_pos']} out of bounds"
    assert -1.0 <= scores["lm_net"] <= 1.0, f"lm_net={scores['lm_net']} out of bounds"
    assert 0.0 <= scores["lm_uncertainty"] <= 1.0, f"lm_uncertainty={scores['lm_uncertainty']} out of bounds"

    # token_count is non-negative integer
    assert scores["token_count"] >= 0
    assert isinstance(scores["token_count"], int)


# ============================================================================
# Property 2: Hawk-Dove scores follow correct formula
# ============================================================================

@given(text=text_strategy)
@settings(max_examples=200)
def test_hd_scores_valid(text):
    """
    Property: For all random text strings, compute_hd_scores(text) returns
    hd_hawk >= 0, hd_dove >= 0, and hd_net = (hawk_count - dove_count) / token_count.

    **Validates: Requirements 3.5**
    """
    scores = compute_hd_scores(text)

    # Check all expected keys
    assert "hd_hawk" in scores
    assert "hd_dove" in scores
    assert "hd_net" in scores
    assert "hawk_count" in scores
    assert "dove_count" in scores

    # Non-negative fractions
    assert scores["hd_hawk"] >= 0.0, f"hd_hawk={scores['hd_hawk']} is negative"
    assert scores["hd_dove"] >= 0.0, f"hd_dove={scores['hd_dove']} is negative"

    # hawk_count and dove_count are non-negative
    assert scores["hawk_count"] >= 0
    assert scores["dove_count"] >= 0

    # Verify hd_net formula: (hawk_count - dove_count) / token_count
    # When there are no tokens, all should be zero
    import re
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    n = len(tokens)

    if n == 0:
        assert scores["hd_net"] == 0.0
        assert scores["hd_hawk"] == 0.0
        assert scores["hd_dove"] == 0.0
    else:
        expected_net = (scores["hawk_count"] - scores["dove_count"]) / n
        assert abs(scores["hd_net"] - expected_net) < 1e-10, \
            f"hd_net={scores['hd_net']} != expected {expected_net}"


# ============================================================================
# Property 3: Content hash is deterministic and valid hex
# ============================================================================

@given(text=st.text(min_size=0, max_size=1000))
@settings(max_examples=200)
def test_content_hash_deterministic(text):
    """
    Property: For all strings, compute_content_hash(s) is deterministic
    (same input -> same output) and returns a 64-char hex string.

    **Validates: Requirements 3.1**
    """
    hash1 = compute_content_hash(text)
    hash2 = compute_content_hash(text)

    # Deterministic
    assert hash1 == hash2, "Hash is not deterministic"

    # 64 character hex string (SHA-256)
    assert len(hash1) == 64, f"Hash length is {len(hash1)}, expected 64"
    assert all(c in '0123456789abcdef' for c in hash1), \
        f"Hash contains non-hex characters: {hash1}"


# ============================================================================
# Property 4: filter_corpus output rows <= input rows
# ============================================================================

@given(
    n_rows=st.integers(min_value=1, max_value=20),
    min_text_length=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=50)
def test_filter_corpus_reduces_rows(n_rows, min_text_length):
    """
    Property: For all valid DataFrames with required columns,
    filter_corpus() output rows <= input rows.

    **Validates: Requirements 3.2**
    """
    # Build a DataFrame with the required columns for filter_corpus
    texts = [f"Sample text number {i} " * 50 for i in range(n_rows)]
    df = pd.DataFrame({
        "Language": np.random.choice(["English", "French", "German"], n_rows),
        "text": texts,
        "Date": pd.date_range("2010-01-01", periods=n_rows, freq="ME").strftime("%Y-%m-%d").tolist(),
        "Filename": [f"doc_{i}" for i in range(n_rows)],
        "Title": [f"Title {i}" for i in range(n_rows)],
        "Authorname": [f"Author {i}" for i in range(n_rows)],
        "Role": ["Governor"] * n_rows,
        "Gender": ["M"] * n_rows,
        "CentralBank": ["Fed"] * n_rows,
        "Country": ["US"] * n_rows,
        "Source": ["website"] * n_rows,
        "URL": ["http://example.com"] * n_rows,
    })

    initial_count = len(df)
    filtered = filter_corpus(df, language="English", min_text_length=min_text_length)

    assert len(filtered) <= initial_count, \
        f"Filtered ({len(filtered)}) > initial ({initial_count})"


# ============================================================================
# Property 5: segment_into_paragraphs respects min_length
# ============================================================================

@given(text=st.text(min_size=0, max_size=2000))
@settings(max_examples=200)
def test_segment_paragraphs_min_length(text):
    """
    Property: segment_into_paragraphs(text, min_length=80) returns only
    paragraphs with len >= 80.

    **Validates: Requirements 3.3**
    """
    min_length = 80
    paragraphs = segment_into_paragraphs(text, min_length=min_length)

    # All returned paragraphs must meet the minimum length
    for i, p in enumerate(paragraphs):
        assert len(p) >= min_length, \
            f"Paragraph {i} has length {len(p)} < min_length={min_length}: '{p[:50]}...'"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
