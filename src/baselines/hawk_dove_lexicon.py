"""
Phase 2 — Baseline 2: Custom Hawkish-Dovish Word List
=======================================================
Domain-specific lexicon for central bank communications.
~200 terms per class capturing monetary policy stance vocabulary.

Score: HD_net = (hawkish_count - dovish_count) / total_tokens
"""

import logging
import re
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Hawkish terms: words/phrases associated with tighter monetary policy
# ============================================================================

HAWKISH_TERMS = {
    # Inflation concerns
    "inflation", "inflationary", "overheating", "overshoot", "overshooting",
    "price pressure", "price pressures", "cost push", "demand pull",
    "above target", "elevated", "persistent", "persistently",
    "entrenched", "upside risk", "upside risks", "second round",
    "second-round", "wage pressure", "wage growth", "wage spiral",
    "expectations unanchored", "de-anchoring", "pass-through",
    
    # Tightening language
    "tighten", "tightening", "hike", "hiking", "raise", "raising",
    "increase", "increasing", "restrictive", "contractionary",
    "normalize", "normalization", "normalise", "normalisation",
    "remove accommodation", "less accommodative", "withdrawal",
    "reduce stimulus", "tapering", "taper", "wind down",
    "quantitative tightening", "balance sheet reduction",
    
    # Strong economy / labor market tightness
    "tight labor", "tight labour", "labor shortage", "labour shortage",
    "full employment", "beyond full employment", "overemployment",
    "robust", "strong growth", "momentum", "buoyant", "vigorous",
    "accelerating", "brisk", "exceeding potential", "above potential",
    "output gap positive", "capacity constraints", "bottlenecks",
    "supply constraints", "excess demand",
    
    # Forward guidance (hawkish)
    "further action", "additional tightening", "prepared to act",
    "whatever it takes", "determined", "committed to price stability",
    "vigilant", "vigilance", "decisive", "forceful", "appropriate pace",
    "sufficiently restrictive", "higher for longer", "data dependent",
    
    # Risk assessment (hawkish)
    "upward revision", "stronger than expected", "hotter than expected",
    "surprised to the upside", "broadening", "broad-based",
    "underlying", "core inflation", "sticky", "stickiness",
    "firmly", "resolutely",
}

# ============================================================================
# Dovish terms: words/phrases associated with looser monetary policy
# ============================================================================

DOVISH_TERMS = {
    # Weakness / slack
    "slack", "spare capacity", "underemployment", "unemployment",
    "jobless", "layoffs", "weakness", "weakening", "softening",
    "slowing", "slowdown", "downturn", "contraction", "recessionary",
    "recession", "below potential", "output gap negative",
    "subdued", "muted", "modest", "moderate", "gradual",
    "disinflationary", "disinflation", "deflationary", "deflation",
    "below target", "undershooting",
    
    # Easing language
    "ease", "easing", "cut", "cutting", "lower", "lowering",
    "reduce", "reducing", "accommodative", "accommodation",
    "stimulus", "stimulative", "supportive", "support",
    "expansionary", "inject", "injection", "liquidity",
    "quantitative easing", "asset purchases", "purchase program",
    "forward guidance", "extended period", "considerable time",
    
    # Patience / gradualism
    "patient", "patience", "wait", "wait and see", "cautious",
    "cautiously", "carefully", "measured", "gradual", "gradualism",
    "flexible", "flexibility", "optionality", "nimble",
    "appropriate time", "premature", "premature tightening",
    "data dependent", "evolving", "monitor", "monitoring",
    "assess", "assessing", "watchful",
    
    # Downside risks
    "downside risk", "downside risks", "headwinds", "fragile",
    "fragility", "vulnerable", "uncertainty", "uncertain",
    "global slowdown", "financial conditions", "tightened unduly",
    "credit conditions", "spillovers", "contagion",
    "geopolitical", "trade tensions", "trade war",
    
    # Forward guidance (dovish)
    "lower for longer", "low rates", "near zero", "zero lower bound",
    "effective lower bound", "negative rates", "negative interest",
    "unconventional", "extraordinary", "emergency", "crisis",
    "whatever is needed", "stand ready", "backstop",
}


def compute_hd_scores(text: str) -> Dict[str, float]:
    """
    Compute Hawkish-Dovish lexicon scores.
    
    Uses simple word matching (case-insensitive).
    Multi-word phrases are matched first.
    
    Returns:
    - hd_hawk: fraction of hawkish tokens/phrases
    - hd_dove: fraction of dovish tokens/phrases
    - hd_net: (hawk - dove) / total_tokens  (positive = hawkish)
    - hawk_count: raw hawkish matches
    - dove_count: raw dovish matches
    """
    text_lower = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text_lower)
    n = len(tokens)
    
    if n == 0:
        return {
            "hd_hawk": 0.0, "hd_dove": 0.0, "hd_net": 0.0,
            "hawk_count": 0, "dove_count": 0,
        }
    
    # Count multi-word phrases first (simple substring matching)
    hawk_count = 0
    dove_count = 0
    
    # Single-word matching via token set
    hawk_single = {t for t in HAWKISH_TERMS if " " not in t and "-" not in t}
    dove_single = {t for t in DOVISH_TERMS if " " not in t and "-" not in t}
    
    # Multi-word matching
    hawk_multi = {t for t in HAWKISH_TERMS if " " in t or "-" in t}
    dove_multi = {t for t in DOVISH_TERMS if " " in t or "-" in t}
    
    # Count single-word matches
    for token in tokens:
        if token in hawk_single:
            hawk_count += 1
        if token in dove_single:
            dove_count += 1
    
    # Count multi-word matches (phrase-level)
    for phrase in hawk_multi:
        hawk_count += text_lower.count(phrase)
    for phrase in dove_multi:
        dove_count += text_lower.count(phrase)
    
    return {
        "hd_hawk": hawk_count / n,
        "hd_dove": dove_count / n,
        "hd_net": (hawk_count - dove_count) / n,
        "hawk_count": hawk_count,
        "dove_count": dove_count,
    }


def score_dataframe(df: pd.DataFrame, text_column: str = "chunk_text") -> pd.DataFrame:
    """Score all rows using Hawk-Dove lexicon."""
    logger.info(f"Scoring {len(df)} chunks with Hawk-Dove lexicon...")
    
    scores = df[text_column].apply(compute_hd_scores)
    scores_df = pd.DataFrame(scores.tolist(), index=df.index)
    
    result = pd.concat([df, scores_df], axis=1)
    
    logger.info(
        f"HD scoring complete. "
        f"Mean HD_net: {scores_df['hd_net'].mean():.4f}, "
        f"Std: {scores_df['hd_net'].std():.4f}"
    )
    
    return result
