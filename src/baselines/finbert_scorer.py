"""
Phase 2 — Baseline 3: FinBERT Zero-Shot Sentiment
===================================================
Runs FinBERT (ProsusAI/finbert) zero-shot inference on text chunks.
Returns positive/negative/neutral probability as continuous scores.

Note: This requires the transformers library and a GPU is recommended
for speed but not required (runs on CPU for smaller batch sizes).
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"


def load_finbert(device: str = None):
    """Load FinBERT model and tokenizer."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading FinBERT ({MODEL_NAME}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    
    logger.info("FinBERT loaded successfully.")
    return tokenizer, model, device


def score_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    max_length: int = 512,
) -> List[Dict[str, float]]:
    """
    Score a batch of texts with FinBERT.
    
    Returns list of dicts with:
    - finbert_pos: positive probability
    - finbert_neg: negative probability
    - finbert_neu: neutral probability
    - finbert_sentiment: pos - neg (continuous score)
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    
    # FinBERT labels: positive=0, negative=1, neutral=2
    results = []
    for i in range(len(texts)):
        results.append({
            "finbert_pos": float(probs[i, 0]),
            "finbert_neg": float(probs[i, 1]),
            "finbert_neu": float(probs[i, 2]),
            "finbert_sentiment": float(probs[i, 0] - probs[i, 1]),
        })
    
    return results


def score_dataframe(
    df: pd.DataFrame,
    text_column: str = "chunk_text",
    batch_size: int = 32,
    device: str = None,
    max_length: int = 512,
) -> pd.DataFrame:
    """
    Score all rows in a DataFrame using FinBERT.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text column
    text_column : str
        Column containing text to score
    batch_size : int
        Batch size for inference
    device : str
        'cuda' or 'cpu' (auto-detected if None)
    max_length : int
        Max token length for FinBERT input
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with FinBERT score columns
    """
    tokenizer, model, device = load_finbert(device)
    
    texts = df[text_column].tolist()
    all_scores = []
    
    n_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"Scoring {len(texts)} chunks with FinBERT (batch_size={batch_size}, {n_batches} batches)...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Handle empty/NaN texts
        batch_clean = [str(t) if pd.notna(t) and len(str(t)) > 0 else "neutral" for t in batch_texts]
        
        batch_scores = score_batch(batch_clean, tokenizer, model, device, max_length)
        all_scores.extend(batch_scores)
        
        if (i // batch_size + 1) % 50 == 0:
            logger.info(f"  Progress: {i + len(batch_texts)}/{len(texts)} chunks scored")
    
    scores_df = pd.DataFrame(all_scores, index=df.index)
    result = pd.concat([df, scores_df], axis=1)
    
    logger.info(
        f"FinBERT scoring complete. "
        f"Mean sentiment: {scores_df['finbert_sentiment'].mean():.4f}, "
        f"Std: {scores_df['finbert_sentiment'].std():.4f}"
    )
    
    return result
