# Bugfix Requirements Document

## Introduction

The `Full_Pipeline.ipynb` notebook has multiple interrelated bugs that prevent it from running end-to-end on Google Colab (A100 GPU, high RAM). The issues span incorrect package names, broken `src.*` imports that assume a local folder structure unavailable on Colab, conceptual mismatches between module design and actual data loading, missing directory creation before file saves, redundant imports across cells, and suboptimal GPU utilization. Together these bugs cause `ModuleNotFoundError`, `FileNotFoundError`, and degraded performance, making the notebook non-functional in its target Colab environment.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the install cell is executed THEN the system installs `pyreader` (a non-existent/wrong package) instead of `pyreadr` (the correct R-data-file reader)

1.2 WHEN any cell containing `from src.data_engineering.*` or `from src.baselines.*` is executed on Colab THEN the system raises `ModuleNotFoundError` because the `src/` folder structure does not exist in the Colab environment

1.3 WHEN the corpus_loader module is imported with its `load_rds_corpus()` / `pyreadr.read_r()` design THEN the system has a conceptual mismatch because the actual next cell loads a CSV file (`CBS_dataset_v1.0.csv`) via `pd.read_csv()`, making the RDS-oriented import dead code

1.4 WHEN `plt.savefig()` is called with a path like `PROJECT_ROOT / 'data' / 'processed' / '...'` THEN the system raises `FileNotFoundError` because `os.makedirs` is not called before the save operation for all output subdirectories

1.5 WHEN the notebook is executed top-to-bottom THEN the system imports the same libraries (pandas, numpy, matplotlib, pathlib, etc.) redundantly in multiple cells, increasing memory footprint and reducing readability

1.6 WHEN FinBERT scoring or LoRA fine-tuning is performed THEN the system does not optimally utilize the A100 GPU because batch sizes are not tuned for high-VRAM GPUs, DataLoader `num_workers` is not set, and mixed-precision (`fp16`/`bf16`) is not enabled by default

1.7 WHEN the notebook references output directories (e.g., `PROJECT_ROOT / 'data' / 'processed'`, `PROJECT_ROOT / 'data' / 'market'`) THEN the system inconsistently handles directory creation — some cells create directories while others assume they exist

### Expected Behavior (Correct)

2.1 WHEN the install cell is executed THEN the system SHALL install `pyreadr` (correct package name for reading .rds files in Python)

2.2 WHEN the notebook is executed on Colab THEN the system SHALL NOT import from `src.*` packages; instead, all module code (corpus_loader, text_segmenter, market_data_pipeline, lm_lexicon, hawk_dove_lexicon, finbert_scorer) SHALL be inlined as self-contained code cells within the notebook

2.3 WHEN corpus loading is performed THEN the system SHALL use a unified approach that matches reality — loading from CSV (`CBS_dataset_v1.0.csv`) with the filtering and registry-building logic inlined, without referencing `pyreadr.read_r()` for RDS files that are not actually used in the CSV-based workflow

2.4 WHEN any figure is saved via `plt.savefig()` THEN the system SHALL ensure the target directory exists by calling `os.makedirs(parent_dir, exist_ok=True)` before the save, either in a consolidated setup cell or immediately before each save call

2.5 WHEN the notebook is structured THEN the system SHALL have a single consolidated import cell at the top containing all library imports used throughout the notebook, with no duplicate imports in subsequent cells

2.6 WHEN FinBERT scoring or LoRA fine-tuning is performed on an A100 GPU THEN the system SHALL use optimized batch sizes (≥64 for FinBERT inference on A100), enable mixed-precision (bf16 where supported), set DataLoader `num_workers > 0`, and use `torch.cuda.amp` or Hugging Face `Trainer` with `fp16=True`/`bf16=True` for training

2.7 WHEN the notebook initializes THEN the system SHALL create all required output directories once in a single setup cell near the top, with `PROJECT_ROOT` consistently set to `/content/drive/MyDrive/nlp_econ` and all subdirectories (`data/processed`, `data/market`, `models`, `outputs`) created via `os.makedirs(..., exist_ok=True)`

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the notebook loads the CBS dataset from CSV THEN the system SHALL CONTINUE TO produce a filtered DataFrame of ~30,000+ English documents with columns matching the existing schema (URL, Title, Date, Authorname, Role, Gender, CentralBank, Country, text, Filename, Language, Source)

3.2 WHEN the document registry is built THEN the system SHALL CONTINUE TO deduplicate by content hash, sort by date, and save as `document_registry.parquet` with the same column structure (doc_id, title, date, author, role, gender, central_bank, country, source, url, text, text_length, content_hash)

3.3 WHEN text segmentation is performed THEN the system SHALL CONTINUE TO produce chunk registries with the same schema (chunk_id, doc_id, chunk_type, position_in_doc, text, token_count, content_hash, date, author, role, central_bank, country, source)

3.4 WHEN LM lexicon scoring is performed THEN the system SHALL CONTINUE TO produce columns (lm_neg, lm_pos, lm_net, lm_uncertainty, token_count) with values computed using the same Loughran-McDonald word lists

3.5 WHEN Hawk-Dove lexicon scoring is performed THEN the system SHALL CONTINUE TO produce columns (hd_hawk, hd_dove, hd_net, hawk_count, dove_count) using the same HAWKISH_TERMS and DOVISH_TERMS dictionaries

3.6 WHEN FinBERT scoring is performed THEN the system SHALL CONTINUE TO produce columns (finbert_pos, finbert_neg, finbert_neu, finbert_sentiment) using the ProsusAI/finbert model with the same softmax-based probability computation

3.7 WHEN market data pipeline functions are used THEN the system SHALL CONTINUE TO support the same event-window extraction and label computation logic (validate_tick_data, extract_event_window, compute_yield_change, compute_equity_change, build_event_labels, match_labels_to_chunks)

3.8 WHEN the 4-phase pipeline structure (Data Engineering → Baselines → Advanced ML → Robustness) is executed THEN the system SHALL CONTINUE TO follow the same logical sequence and produce equivalent analytical outputs
