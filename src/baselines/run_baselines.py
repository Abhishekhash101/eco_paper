"""
Phase 2 — Run All Baselines & Produce Comparison Table
========================================================
Orchestrates all baseline scoring methods, computes correlations
with Δy₂ labels, and produces the baseline comparison table (Table 1).

Baselines:
1. LM Lexicon (neg, pos, net)
2. Custom Hawk-Dove word list (hawk, dove, net)
3. FinBERT zero-shot (pos, neg, sentiment)

Metrics:
- Pearson r with Δy₂
- Spearman ρ with Δy₂
- Directional accuracy (sign prediction)
- MAE in basis points (for regression-capable scores)
- Bootstrap 95% CIs on correlations

Usage:
    python -m src.baselines.run_baselines
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.lm_lexicon import score_dataframe as lm_score
from src.baselines.hawk_dove_lexicon import score_dataframe as hd_score

logger = logging.getLogger(__name__)


def bootstrap_correlation(x, y, n_bootstrap=2000, ci=0.95, method="pearson"):
    """Compute bootstrap CI for correlation."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    
    if n < 10:
        return np.nan, np.nan, np.nan
    
    corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr
    observed = corr_func(x, y)[0]
    
    boot_corrs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            r = corr_func(x[idx], y[idx])[0]
            boot_corrs.append(r)
        except:
            continue
    
    boot_corrs = np.array(boot_corrs)
    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_corrs, alpha * 100)
    ci_high = np.percentile(boot_corrs, (1 - alpha) * 100)
    
    return observed, ci_low, ci_high


def compute_directional_accuracy(scores, labels):
    """Fraction of cases where sign(score) == sign(label)."""
    valid = ~(np.isnan(scores) | np.isnan(labels))
    s, l = scores[valid], labels[valid]
    
    # Exclude zero labels (no direction)
    nonzero = l != 0
    s, l = s[nonzero], l[nonzero]
    
    if len(s) == 0:
        return np.nan
    
    return np.mean(np.sign(s) == np.sign(l))


def evaluate_baseline(scores, labels, method_name):
    """Compute all metrics for a single baseline method."""
    valid = ~(np.isnan(scores) | np.isnan(labels))
    s, l = scores[valid], labels[valid]
    
    if len(s) < 10:
        logger.warning(f"{method_name}: insufficient valid observations ({len(s)})")
        return None
    
    # Pearson
    r, r_ci_low, r_ci_high = bootstrap_correlation(s, l, method="pearson")
    r_pval = stats.pearsonr(s, l)[1]
    
    # Spearman
    rho, rho_ci_low, rho_ci_high = bootstrap_correlation(s, l, method="spearman")
    rho_pval = stats.spearmanr(s, l)[1]
    
    # Directional accuracy
    dir_acc = compute_directional_accuracy(scores, labels)
    
    # MAE (treating score as a scaled prediction — simple linear mapping)
    # Fit simple linear: label = a + b*score, then compute MAE
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(s.reshape(-1, 1), l)
    pred = reg.predict(s.reshape(-1, 1))
    mae = np.mean(np.abs(l - pred))
    
    return {
        "Method": method_name,
        "N": len(s),
        "Pearson_r": r,
        "r_CI_low": r_ci_low,
        "r_CI_high": r_ci_high,
        "r_pval": r_pval,
        "Spearman_rho": rho,
        "rho_CI_low": rho_ci_low,
        "rho_CI_high": rho_ci_high,
        "rho_pval": rho_pval,
        "Dir_Accuracy": dir_acc,
        "MAE_bp": mae,
    }


def aggregate_to_event_level(df, score_columns, label_col="delta_US_2Y_bp", 
                              group_col="doc_id", agg="mean"):
    """
    Aggregate chunk-level scores to document-level.
    
    This is required because multiple chunks map to a single event-day label.
    Uses mean aggregation by default (following standard practice).
    """
    agg_dict = {col: agg for col in score_columns}
    agg_dict[label_col] = "first"  # label is same for all chunks in a doc
    
    event_df = df.groupby(group_col).agg(agg_dict).reset_index()
    
    return event_df


def run_lexicon_baselines(
    labeled_chunks_path: str = None,
    output_dir: str = None,
    run_finbert: bool = False,
):
    """
    Run all lexicon-based baselines and produce comparison table.
    
    Parameters
    ----------
    labeled_chunks_path : str
        Path to labeled chunk registry (from Phase 1 M2)
    output_dir : str
        Where to save results
    run_finbert : bool
        Whether to run FinBERT (requires transformers + GPU recommended)
    """
    if labeled_chunks_path is None:
        labeled_chunks_path = str(PROJECT_ROOT / "data" / "processed" / "labeled_chunk_registry.parquet")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "processed")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("PHASE 2 — BASELINE MODELS")
    print("=" * 70)
    
    # Load labeled chunks
    print("\n  Loading labeled chunks...")
    chunks = pd.read_parquet(labeled_chunks_path)
    print(f"    {len(chunks)} chunks loaded")
    
    # Filter to chunks with valid labels
    label_col = "delta_US_2Y_bp"
    valid_chunks = chunks[chunks[label_col].notna()].copy()
    print(f"    {len(valid_chunks)} chunks with valid Δy₂ labels")
    
    # Detect text column name
    text_col = "chunk_text" if "chunk_text" in valid_chunks.columns else "text"
    print(f"    Using text column: '{text_col}'")
    
    # =========================================================================
    # Baseline 1: LM Lexicon
    # =========================================================================
    print("\n  [1/3] Running Loughran-McDonald Lexicon...")
    scored_lm = lm_score(valid_chunks, text_column=text_col)
    
    # =========================================================================
    # Baseline 2: Hawk-Dove Lexicon
    # =========================================================================
    print("  [2/3] Running Hawk-Dove Lexicon...")
    scored_hd = hd_score(scored_lm, text_column=text_col)
    
    # =========================================================================
    # Baseline 3: FinBERT (optional — resource intensive)
    # =========================================================================
    if run_finbert:
        print("  [3/3] Running FinBERT Zero-Shot...")
        from src.baselines.finbert_scorer import score_dataframe as finbert_score
        scored_all = finbert_score(scored_hd, text_column=text_col, batch_size=64)
    else:
        print("  [3/3] FinBERT skipped (set run_finbert=True to enable)")
        scored_all = scored_hd
    
    # =========================================================================
    # Aggregate to document/event level
    # =========================================================================
    print("\n  Aggregating chunk scores to document level...")
    
    score_columns = ["lm_neg", "lm_pos", "lm_net", "lm_uncertainty",
                     "hd_hawk", "hd_dove", "hd_net"]
    if run_finbert:
        score_columns += ["finbert_pos", "finbert_neg", "finbert_sentiment"]
    
    event_df = aggregate_to_event_level(
        scored_all, score_columns, label_col=label_col, group_col="doc_id"
    )
    print(f"    {len(event_df)} unique documents (event-level)")
    
    # =========================================================================
    # Compute metrics for each baseline
    # =========================================================================
    print("\n  Computing evaluation metrics...")
    
    labels = event_df[label_col].values
    
    results = []
    
    # LM methods
    for col, name in [
        ("lm_neg", "LM Negative"),
        ("lm_pos", "LM Positive"),
        ("lm_net", "LM Net (pos-neg)"),
        ("lm_uncertainty", "LM Uncertainty"),
    ]:
        scores = event_df[col].values
        # For LM_neg, higher negative → lower yield (invert sign)
        if col == "lm_neg":
            scores = -scores
        r = evaluate_baseline(scores, labels, name)
        if r:
            results.append(r)
    
    # Hawk-Dove methods
    for col, name in [
        ("hd_hawk", "HD Hawkish"),
        ("hd_dove", "HD Dovish"),
        ("hd_net", "HD Net (hawk-dove)"),
    ]:
        scores = event_df[col].values
        if col == "hd_dove":
            scores = -scores  # more dovish → lower yield
        r = evaluate_baseline(scores, labels, name)
        if r:
            results.append(r)
    
    # FinBERT
    if run_finbert:
        for col, name in [
            ("finbert_sentiment", "FinBERT Sentiment"),
            ("finbert_neg", "FinBERT Negative"),
        ]:
            scores = event_df[col].values
            if col == "finbert_neg":
                scores = -scores
            r = evaluate_baseline(scores, labels, name)
            if r:
                results.append(r)
    
    # =========================================================================
    # Build comparison table
    # =========================================================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Pearson_r", ascending=False, key=abs)
    
    # Save
    table_path = output_path / "baseline_comparison_table.csv"
    results_df.to_csv(table_path, index=False)
    
    # Also save the scored event-level data
    event_path = output_path / "event_level_baseline_scores.parquet"
    event_df.to_parquet(event_path, index=False)
    
    # =========================================================================
    # Print Table 1
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TABLE 1: BASELINE COMPARISON (Event-Level)")
    print("=" * 70)
    print(f"  {'Method':<25} {'Pearson r':>10} {'95% CI':>18} {'Spearman ρ':>11} {'Dir.Acc':>8} {'MAE(bp)':>8}")
    print("  " + "-" * 82)
    
    for _, row in results_df.iterrows():
        ci_str = f"[{row['r_CI_low']:.3f}, {row['r_CI_high']:.3f}]"
        sig = "***" if row["r_pval"] < 0.001 else "**" if row["r_pval"] < 0.01 else "*" if row["r_pval"] < 0.05 else ""
        print(f"  {row['Method']:<25} {row['Pearson_r']:>8.4f}{sig:<2} {ci_str:>18} {row['Spearman_rho']:>9.4f} {row['Dir_Accuracy']:>8.3f} {row['MAE_bp']:>8.2f}")
    
    print(f"\n  N = {results_df['N'].iloc[0]} documents")
    print(f"  Bootstrap: 10,000 iterations, 95% CI")
    print(f"  Significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    print(f"\n  Saved: {table_path}")
    print(f"  Saved: {event_path}")
    
    print("\n" + "=" * 70)
    print("  PHASE 2 — LEXICON BASELINES COMPLETE")
    print("=" * 70)
    
    return results_df, event_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    results_df, event_df = run_lexicon_baselines(
        run_finbert=False,  # Set True if transformers is installed
    )
