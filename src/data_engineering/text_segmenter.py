"""
Milestone 1 — Step 2: Text Segmentation
========================================
Segments documents into paragraph-level and sentence-level chunks
using spaCy with economic-domain tokenization overrides.

Each chunk receives:
- chunk_id: unique identifier
- doc_id: parent document reference
- chunk_type: paragraph or sentence
- position_in_doc: sequential position
- text: chunk content
- token_count: approximate token count
- All parent document metadata (central_bank, country, date, author, role)
"""

import hashlib
import logging
import re
from typing import List, Dict, Optional

import pandas as pd
import spacy
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Economic-domain patterns that should NOT be split
ECONOMIC_PATTERNS = [
    r"\d+\.\d+%",        # 2.25%
    r"Q[1-4]\s?\d{4}",   # Q3 2022
    r"H[12]\s?\d{4}",    # H1 2023
    r"\d{4}[-–]\d{4}",   # 2022-2024
    r"pp\.",             # percentage points
    r"bps?",            # basis points
    r"[A-Z]{2,5}\d*",   # Acronyms like FOMC, GDP, CPI
]


def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.Language:
    """Load spaCy model, downloading if necessary."""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.info(f"Downloading spaCy model: {model_name}")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)

    # Increase max length for long speeches
    nlp.max_length = 500_000
    return nlp


def segment_into_paragraphs(text: str, min_length: int = 80) -> List[str]:
    """
    Split text into paragraphs based on double newlines or
    single newlines with indentation patterns.

    Parameters
    ----------
    text : str
        Full document text
    min_length : int
        Minimum character length for a valid paragraph

    Returns
    -------
    List[str]
        List of paragraph strings
    """
    # Split on double newlines (standard paragraph separator)
    paragraphs = re.split(r"\n\s*\n", text)

    # If that yields only 1 paragraph, try single newlines
    if len(paragraphs) <= 1:
        paragraphs = re.split(r"\n(?=[A-Z])", text)

    # Clean and filter
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)  # Normalize whitespace
        if len(p) >= min_length:
            cleaned.append(p)

    # If still no paragraphs, treat the whole text as one
    if not cleaned:
        text_clean = re.sub(r"\s+", " ", text.strip())
        if len(text_clean) >= min_length:
            cleaned = [text_clean]

    return cleaned


def segment_into_windows(
    text: str,
    nlp: spacy.Language,
    target_tokens: int = 256,
    max_tokens: int = 512,
    min_tokens: int = 20,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Split text into overlapping sentence-windows of ~target_tokens each.
    This produces chunks sized appropriately for transformer input.

    Strategy: accumulate sentences until reaching target_tokens, then start
    a new window with overlap_sentences carried over for context continuity.

    Parameters
    ----------
    text : str
        Full document text
    nlp : spacy.Language
        spaCy model for sentence segmentation
    target_tokens : int
        Target token count per window (default 256 for good granularity)
    max_tokens : int
        Hard maximum tokens per window
    min_tokens : int
        Minimum tokens to keep a window
    overlap_sentences : int
        Number of sentences to overlap between consecutive windows

    Returns
    -------
    List[str]
        List of sentence-window strings
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

    if not sentences:
        return []

    windows = []
    current_sents = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(sent.split())

        # If adding this sentence exceeds max, flush current window
        if current_tokens + sent_tokens > max_tokens and current_sents:
            window_text = " ".join(current_sents)
            if len(window_text.split()) >= min_tokens:
                windows.append(window_text)

            # Start new window with overlap
            if overlap_sentences > 0 and len(current_sents) > overlap_sentences:
                current_sents = current_sents[-overlap_sentences:]
                current_tokens = sum(len(s.split()) for s in current_sents)
            else:
                current_sents = []
                current_tokens = 0

        current_sents.append(sent)
        current_tokens += sent_tokens

        # If we've reached target, flush
        if current_tokens >= target_tokens:
            window_text = " ".join(current_sents)
            if len(window_text.split()) >= min_tokens:
                windows.append(window_text)

            # Start new window with overlap
            if overlap_sentences > 0 and len(current_sents) > overlap_sentences:
                current_sents = current_sents[-overlap_sentences:]
                current_tokens = sum(len(s.split()) for s in current_sents)
            else:
                current_sents = []
                current_tokens = 0

    # Don't forget the last window
    if current_sents:
        window_text = " ".join(current_sents)
        if len(window_text.split()) >= min_tokens:
            windows.append(window_text)

    return windows


def segment_into_sentences(
    text: str,
    nlp: spacy.Language,
    min_tokens: int = 20,
    max_tokens: int = 512,
) -> List[str]:
    """
    Split text into sentences using spaCy's sentence segmenter
    with economic-domain awareness.

    Parameters
    ----------
    text : str
        Text to segment (paragraph or full document)
    nlp : spacy.Language
        Loaded spaCy model
    min_tokens : int
        Minimum token count per sentence
    max_tokens : int
        Maximum token count; longer sentences are kept as-is
        (will be truncated at model input stage)

    Returns
    -------
    List[str]
        List of sentence strings
    """
    # Process with spaCy
    doc = nlp(text)

    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        token_count = len(sent_text.split())

        if token_count >= min_tokens:
            sentences.append(sent_text)
        elif sentences:
            # Merge short sentence with previous if possible
            sentences[-1] = sentences[-1] + " " + sent_text

    return sentences


def build_chunk_registry(
    document_registry: pd.DataFrame,
    chunk_type: str = "window",
    nlp: Optional[spacy.Language] = None,
    min_tokens: int = 20,
    max_tokens: int = 512,
    target_tokens: int = 256,
    overlap_sentences: int = 1,
) -> pd.DataFrame:
    """
    Build a chunk-level registry from the document registry.

    Parameters
    ----------
    document_registry : pd.DataFrame
        Document-level registry from corpus_loader
    chunk_type : str
        'window' (sentence-windows, recommended for transformers),
        'paragraph', or 'sentence'
    nlp : spacy.Language, optional
        Required for sentence and window segmentation
    min_tokens : int
        Minimum tokens per chunk
    max_tokens : int
        Maximum tokens per chunk
    target_tokens : int
        Target tokens per window (for chunk_type='window')
    overlap_sentences : int
        Overlap between consecutive windows

    Returns
    -------
    pd.DataFrame
        Chunk-level registry with all metadata
    """
    if chunk_type in ("sentence", "window") and nlp is None:
        nlp = load_spacy_model()

    chunks = []
    logger.info(
        f"Segmenting {len(document_registry)} documents into {chunk_type}-level chunks..."
    )

    for idx, row in tqdm(
        document_registry.iterrows(),
        total=len(document_registry),
        desc=f"Segmenting ({chunk_type})",
    ):
        text = row["text"]
        doc_id = row["doc_id"]

        if chunk_type == "paragraph":
            segments = segment_into_paragraphs(text, min_length=min_tokens * 5)
        elif chunk_type == "sentence":
            segments = segment_into_sentences(
                text, nlp, min_tokens=min_tokens, max_tokens=max_tokens
            )
        elif chunk_type == "window":
            segments = segment_into_windows(
                text, nlp,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap_sentences=overlap_sentences,
            )
        else:
            raise ValueError(f"Unknown chunk_type: {chunk_type}")

        for pos, segment in enumerate(segments):
            token_count = len(segment.split())
            if token_count < min_tokens:
                continue

            chunk_id = f"{doc_id}_{chunk_type[0]}_{pos:04d}"
            content_hash = hashlib.sha256(segment.encode("utf-8")).hexdigest()[:16]

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_type": chunk_type,
                "position_in_doc": pos,
                "text": segment,
                "token_count": token_count,
                "content_hash": content_hash,
                # Propagate document metadata
                "date": row["date"],
                "author": row["author"],
                "role": row["role"],
                "central_bank": row["central_bank"],
                "country": row["country"],
                "source": row["source"],
            })

    chunk_df = pd.DataFrame(chunks)
    logger.info(
        f"Created {len(chunk_df)} {chunk_type}-level chunks "
        f"from {len(document_registry)} documents "
        f"(avg {len(chunk_df) / len(document_registry):.1f} chunks/doc)"
    )

    return chunk_df


def run_segmentation(
    document_registry: pd.DataFrame,
    output_dir: str = "data/processed",
    chunk_types: List[str] = None,
    min_tokens: int = 20,
    max_tokens: int = 512,
) -> Dict[str, pd.DataFrame]:
    """
    Full segmentation pipeline: segment documents into chunks and save.

    Parameters
    ----------
    document_registry : pd.DataFrame
        Document registry from corpus_loader
    output_dir : str
        Directory to save chunk registries
    chunk_types : list
        Types of segmentation to perform
    min_tokens : int
        Minimum tokens per chunk
    max_tokens : int
        Maximum tokens per chunk

    Returns
    -------
    dict
        Mapping of chunk_type -> DataFrame
    """
    from pathlib import Path

    if chunk_types is None:
        chunk_types = ["paragraph"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nlp = None
    if "sentence" in chunk_types or "window" in chunk_types:
        nlp = load_spacy_model()

    results = {}

    for ct in chunk_types:
        chunk_df = build_chunk_registry(
            document_registry,
            chunk_type=ct,
            nlp=nlp,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

        # Save
        out_file = output_path / f"chunk_registry_{ct}.parquet"
        chunk_df.to_parquet(out_file, index=False)
        logger.info(f"Saved {ct} chunk registry to {out_file}")

        results[ct] = chunk_df

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Load document registry
    registry = pd.read_parquet("data/processed/document_registry.parquet")
    print(f"Loaded {len(registry)} documents")

    # Run paragraph segmentation (faster, no spaCy needed)
    results = run_segmentation(
        registry,
        chunk_types=["paragraph"],
        min_tokens=20,
        max_tokens=512,
    )

    for ct, df in results.items():
        print(f"\n{ct}: {len(df)} chunks, avg {df['token_count'].mean():.0f} tokens/chunk")
