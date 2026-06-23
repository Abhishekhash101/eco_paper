# Quantifying Latent Policy Stance
## Extracting Implicit Macroeconomic Sentiment from Central Bank Communications

NLP + High-Frequency Finance research project targeting JFE / RFS / ACL / EMNLP.

---

## Quick Start

### 1. Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Obtain the CBS Dataset

Download `CBS_dataset_v1.0.rds` and place it in the project root.
- Source: [BIS Central Bank Speeches](https://www.bis.org/cbspeeches/)
- This file is ~50MB and contains 35,487 central bank speeches

### 3. Run the Pipeline (in order)

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | `notebooks/Ticker_Fetch.ipynb` | Fetches sovereign bond yields & equity data → CSVs |
| 2 | `notebooks/Phase1_Data_Engineering.ipynb` | Loads CBS corpus, segments into chunks |
| 3 | `notebooks/Phase1B_Daily_Labels.ipynb` | Matches daily Δy₂ labels to speech dates |
| 4 | `notebooks/Phase2_Baseline_Models.ipynb` | LM lexicon + Hawk-Dove + FinBERT baselines |
| 5 | `notebooks/Phase3_Advanced_ML.ipynb` | LoRA fine-tuning (**run on Colab GPU**) |
| 6 | `notebooks/Phase4_Robustness_OOS.ipynb` | Walk-forward CV, ablations, placebo tests |

### 4. Phase 3 (GPU Required)

Phase 3 requires a GPU (A100 recommended, T4 minimum).
- Use Google Colab Pro or cloud GPU (Lambda Labs, RunPod)
- Full code is in `scripts/phase3_colab_cells.py` — copy cells into Colab
- Upload `data/processed/labeled_chunk_registry.parquet` and
  `data/processed/labeled_document_registry.parquet` to Colab

---

## Project Structure

```
.
├── CBS_dataset_v1.0.rds          # INPUT: Central Bank Speeches (obtain separately)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── research_overview.pdf         # Research design document
├── research_roadmap.pdf          # Full 5-phase project roadmap
│
├── configs/
│   └── pipeline_config.yaml      # Pipeline configuration
│
├── notebooks/
│   ├── Ticker_Fetch.ipynb        # Step 1: Fetch market data
│   ├── Phase1_Data_Engineering.ipynb   # Step 2: Corpus processing
│   ├── Phase1B_Daily_Labels.ipynb      # Step 3: Label construction
│   ├── Phase2_Baseline_Models.ipynb    # Step 4: Baselines
│   ├── Phase3_Advanced_ML.ipynb        # Step 5: LoRA (Colab)
│   └── Phase4_Robustness_OOS.ipynb     # Step 6: Robustness
│
├── scripts/
│   └── phase3_colab_cells.py     # Phase 3 complete code for Colab
│
├── src/
│   ├── data_engineering/         # Phase 1 pipeline code
│   │   ├── corpus_loader.py      # CBS dataset loading & filtering
│   │   ├── text_segmenter.py     # Text segmentation (paragraph/sentence/window)
│   │   ├── market_data_pipeline.py  # Event-window label construction
│   │   ├── daily_label_builder.py   # Option A: daily frequency labels
│   │   └── run_phase1.py         # Orchestrator
│   │
│   └── baselines/                # Phase 2 baseline models
│       ├── lm_lexicon.py         # Loughran-McDonald lexicon scorer
│       ├── hawk_dove_lexicon.py  # Custom hawkish-dovish word list
│       ├── finbert_scorer.py     # FinBERT zero-shot sentiment
│       └── run_baselines.py      # Run all baselines
│
└── data/                         # Generated (not in git)
    ├── processed/                # Parquet files from pipeline
    └── market/                   # Tick data (if available)
```

---

## Research Overview

This project introduces a framework that maps central bank text to a continuous
hawkish-dovish policy stance dimension using:

1. **LoRA fine-tuning** of Mistral-7B on high-frequency yield changes
2. **Contextual embeddings** from the fine-tuned model
3. **PCA-based latent stance space** validated against market reactions

Key insight: Central bank communications encode policy signals through contextual
framing (not explicit sentiment words), which lexicon-based methods cannot capture.

---

## Hardware Requirements

- **Phases 1-2, 4:** Any machine with 8GB+ RAM
- **Phase 3:** GPU with 24GB+ VRAM (A100 80GB recommended)
  - Colab Pro ($10/month) provides A100 access
  - Training takes ~2-4 hours on A100

---

## Citation

If using this code, please cite:
```
[Author]. Quantifying Latent Policy Stance: Extracting Implicit Macroeconomic
Sentiment from Central Bank Communications using Contextual Embeddings. 2024.
```
