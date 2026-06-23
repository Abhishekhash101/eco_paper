"""
Phase 2 — Baseline 1: Loughran-McDonald Lexicon Scoring
=========================================================
Implements LM lexicon-based sentiment scoring for central bank text chunks.

Scores computed:
- LM_neg: fraction of negative-category tokens
- LM_pos: fraction of positive tokens
- LM_net: LM_pos - LM_neg
- LM_uncertainty: fraction of uncertainty tokens
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Loughran-McDonald Word Lists (Core subsets for financial/economic text)
# Full lists: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# ============================================================================

LM_NEGATIVE = {
    "abandon", "abandonment", "abuse", "abuses", "accident", "adverse",
    "adversely", "against", "aggravate", "allegations", "alleged", "annul",
    "argue", "arrest", "bad", "bail", "bankrupt", "bankruptcy", "breach",
    "burden", "catastrophe", "caution", "cease", "claim", "claims", "close",
    "closing", "collapse", "collusion", "complaint", "concern", "concerns",
    "condemn", "conflict", "constraint", "constraints", "contraction",
    "controversy", "conviction", "corrective", "costly", "crisis", "critical",
    "criticize", "damage", "damages", "danger", "dangerous", "deadlock",
    "death", "decline", "declined", "declining", "default", "defaults",
    "defeat", "deficiency", "deficit", "degrade", "delay", "delaying",
    "delinquency", "delinquent", "denial", "depressed", "depression",
    "destabilize", "deteriorate", "deteriorating", "deterioration", "detriment",
    "detrimental", "difficult", "difficulties", "difficulty", "diminish",
    "disadvantage", "disapproval", "disaster", "disclose", "discontinue",
    "discourage", "disruption", "dissent", "distort", "distortion", "distress",
    "doubt", "downgrade", "downturn", "drag", "drop", "drought", "egregious",
    "erode", "erosion", "error", "escalate", "eviction", "exacerbate",
    "excessive", "exclusion", "expose", "fail", "failed", "failing", "failure",
    "falling", "fear", "fines", "force", "fraud", "freeze", "frustrate",
    "hamper", "hardship", "harm", "harmful", "harsh", "hinder", "hostile",
    "hurt", "idle", "illegal", "impair", "impairment", "impediment",
    "inability", "inadequate", "inadvertent", "insolvent", "instability",
    "insufficient", "investigation", "involuntary", "jeopardize", "lack",
    "lag", "lags", "late", "lawsuit", "layoff", "layoffs", "liability",
    "limitation", "liquidate", "liquidation", "litigation", "lose", "loss",
    "losses", "lost", "low", "lower", "malfunction", "manipulate",
    "misappropriate", "misconduct", "misrepresent", "miss", "mistake",
    "negative", "negatively", "neglect", "negligence", "obstacle",
    "obsolete", "offend", "oppose", "outage", "overdue", "overestimate",
    "penalty", "peril", "persist", "plague", "plummet", "poor", "poorly",
    "postpone", "problem", "problems", "prohibit", "prolong", "protest",
    "punish", "punitive", "recession", "recessionary", "reckless", "reduce",
    "reduced", "reduction", "redundancy", "reject", "rejected", "restate",
    "restrain", "restrict", "restriction", "restructure", "risk", "risks",
    "risky", "sanction", "scandal", "scarce", "scarcity", "seize", "serious",
    "setback", "severe", "sharply", "shock", "shortage", "shrink", "shut",
    "slippage", "slow", "slowdown", "slower", "slowing", "slump", "stagnant",
    "stagnation", "strain", "stress", "stringent", "struggle", "sue",
    "suffer", "suspend", "suspension", "tariff", "terminate", "termination",
    "threat", "threaten", "tighten", "tightening", "trouble", "unable",
    "uncertain", "uncertainty", "undermine", "underperform", "unfavorable",
    "unfortunate", "unpaid", "unprofitable", "unresolved", "unsuccessful",
    "untimely", "urgent", "violate", "violation", "volatile", "volatility",
    "vulnerable", "warn", "warning", "weak", "weaken", "weakening",
    "weakness", "worsen", "worsening", "worst", "writedown", "writeoff",
}

LM_POSITIVE = {
    "able", "abundance", "abundant", "accomplish", "accomplishment",
    "achieve", "achievement", "adequate", "advance", "advancement",
    "advantage", "advantageous", "benefit", "beneficial", "bolster",
    "boost", "brilliant", "certain", "complement", "confident",
    "constructive", "creative", "delight", "desirable", "diligent",
    "distinctive", "efficient", "efficiently", "enable", "encourage",
    "encouraging", "enhance", "enhancement", "enjoy", "enthusiasm",
    "excellent", "exceptional", "excitement", "exclusive", "favorable",
    "favorably", "gain", "gained", "gains", "good", "great", "greater",
    "greatest", "grew", "grow", "growing", "growth", "guarantee",
    "happy", "highest", "honor", "ideal", "improve", "improved",
    "improvement", "improving", "increase", "increased", "incredible",
    "innovative", "insight", "integrity", "leadership", "lucrative",
    "maximize", "momentum", "notable", "opportunity", "optimal",
    "optimism", "optimistic", "outperform", "outstanding", "overcome",
    "pleased", "pleasure", "popular", "positive", "positively",
    "proactive", "proficiency", "proficient", "profit", "profitable",
    "profitability", "progress", "progressive", "prosper", "prosperity",
    "prosperous", "reassure", "rebound", "recover", "recovery",
    "regain", "remarkable", "resilience", "resilient", "resolve",
    "restore", "reward", "rewarding", "rise", "rising", "robust",
    "satisfaction", "satisfactory", "smooth", "solution", "stability",
    "stabilize", "stable", "steady", "strength", "strengthen",
    "strong", "stronger", "strongest", "succeed", "success", "successful",
    "successfully", "superior", "support", "supportive", "surge",
    "surpass", "sustain", "sustainable", "tremendous", "upturn",
    "valuable", "win", "winner",
}

LM_UNCERTAINTY = {
    "almost", "ambiguity", "ambiguous", "apparent", "apparently",
    "appear", "appeared", "appears", "approximate", "approximately",
    "assumption", "believe", "believed", "cautious", "conceivable",
    "conceivably", "conditional", "confuse", "contingency", "contingent",
    "could", "depend", "dependent", "depending", "doubt", "doubtful",
    "estimate", "estimated", "estimates", "exposure", "fluctuate",
    "fluctuation", "forecast", "generally", "guess", "imprecise",
    "imprecision", "indefinite", "indefinitely", "indeterminate",
    "indicate", "inherent", "inherently", "likelihood", "may", "maybe",
    "might", "nearly", "occasionally", "pending", "perhaps",
    "possibility", "possible", "possibly", "potential", "potentially",
    "predict", "prediction", "preliminary", "presumably", "probable",
    "probably", "project", "projected", "projecting", "projection",
    "random", "randomness", "reassess", "recalculate", "reconsider",
    "redefine", "reestimate", "reexamine", "reinterpret", "revise",
    "revision", "risk", "risky", "roughly", "seem", "seemed", "seemingly",
    "seems", "somewhat", "speculate", "speculation", "suggest",
    "suggesting", "tentative", "tentatively", "uncertain", "uncertainty",
    "unclear", "unconfirmed", "undecided", "undefined", "undetermined",
    "unexpected", "unexpectedly", "unknown", "unlikely", "unplanned",
    "unpredictable", "unpredictability", "unproven", "unquantifiable",
    "unsettled", "unspecified", "untested", "unusual", "usually",
    "variable", "variability", "vary", "varying", "volatile", "volatility",
}


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for lexicon matching."""
    # Lowercase and extract word tokens
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    return tokens


def compute_lm_scores(text: str) -> Dict[str, float]:
    """
    Compute Loughran-McDonald lexicon scores for a text chunk.
    
    Returns dict with:
    - lm_neg: fraction of negative tokens
    - lm_pos: fraction of positive tokens
    - lm_net: lm_pos - lm_neg
    - lm_uncertainty: fraction of uncertainty tokens
    - token_count: total tokens
    """
    tokens = tokenize_simple(text)
    n = len(tokens)
    
    if n == 0:
        return {
            "lm_neg": 0.0, "lm_pos": 0.0, "lm_net": 0.0,
            "lm_uncertainty": 0.0, "token_count": 0,
        }
    
    neg_count = sum(1 for t in tokens if t in LM_NEGATIVE)
    pos_count = sum(1 for t in tokens if t in LM_POSITIVE)
    unc_count = sum(1 for t in tokens if t in LM_UNCERTAINTY)
    
    return {
        "lm_neg": neg_count / n,
        "lm_pos": pos_count / n,
        "lm_net": (pos_count - neg_count) / n,
        "lm_uncertainty": unc_count / n,
        "token_count": n,
    }


def score_dataframe(df: pd.DataFrame, text_column: str = "chunk_text") -> pd.DataFrame:
    """
    Score all rows in a DataFrame using LM lexicon.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a text column
    text_column : str
        Name of column containing text
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with LM score columns appended
    """
    logger.info(f"Scoring {len(df)} chunks with LM lexicon...")
    
    scores = df[text_column].apply(compute_lm_scores)
    scores_df = pd.DataFrame(scores.tolist(), index=df.index)
    
    result = pd.concat([df, scores_df], axis=1)
    
    logger.info(
        f"LM scoring complete. "
        f"Mean LM_net: {scores_df['lm_net'].mean():.4f}, "
        f"Std: {scores_df['lm_net'].std():.4f}"
    )
    
    return result
