# Full Pipeline Notebook Fix — Bugfix Design

## Overview

The `Full_Pipeline.ipynb` notebook contains seven interrelated bugs that prevent end-to-end execution on Google Colab (A100 GPU). The bugs range from a typo in a package name (`pyreader` → `pyreadr`) to structural issues like broken `src.*` imports, conceptual mismatches between RDS-oriented code and actual CSV usage, missing directory creation, redundant imports, suboptimal GPU settings, and inconsistent directory handling.

The fix strategy is to transform the notebook into a fully self-contained Colab artifact: inline all `src/` module logic directly into notebook cells, consolidate imports and directory setup into early cells, align the corpus loading approach with the actual CSV-based workflow, and add A100-optimized GPU configuration.

## Glossary

- **Bug_Condition (C)**: Any cell execution in the notebook that triggers `ModuleNotFoundError`, `FileNotFoundError`, installs the wrong package, or runs suboptimally on A100 due to the seven identified defects
- **Property (P)**: The notebook SHALL execute all cells top-to-bottom on Colab without errors, produce correct analytical outputs, and utilize A100 GPU efficiently
- **Preservation**: Existing analytical logic (filtering, deduplication, segmentation, scoring algorithms, pipeline sequence) must produce equivalent results
- **Full_Pipeline.ipynb**: The Jupyter notebook at `notebooks/Full_Pipeline.ipynb` — the sole file being fixed
- **PROJECT_ROOT**: The Google Drive mount path `/content/drive/MyDrive/nlp_econ` used as the base for all I/O
- **CBS Dataset**: The Central Bank Speeches dataset loaded from `CBS_dataset_v1.0.csv` (35,487 records)
- **Inlining**: Replacing `from src.module import func` with the actual function definitions pasted directly into notebook cells

## Bug Details

### Bug Condition

The bug manifests when the notebook is executed top-to-bottom on Google Colab (A100 GPU, high RAM). The notebook fails at multiple points due to incorrect package names, unavailable local module imports, missing directories, and misconfigured GPU settings.

**Formal Specification:**
```
FUNCTION isBugCondition(cell)
  INPUT: cell of type NotebookCell
  OUTPUT: boolean
  
  RETURN cell.installs("pyreader")                                           -- Bug 1.1
         OR cell.imports_from("src.data_engineering.*")                       -- Bug 1.2
         OR cell.imports_from("src.baselines.*")                              -- Bug 1.2
         OR cell.uses_pyreadr_read_r_but_actual_data_is_csv()                -- Bug 1.3
         OR cell.calls_savefig_without_makedirs()                            -- Bug 1.4
         OR cell.reimports_already_imported_library()                         -- Bug 1.5
         OR (cell.runs_gpu_workload() AND NOT cell.uses_optimal_a100_config())-- Bug 1.6
         OR cell.assumes_directory_exists_without_creation()                  -- Bug 1.7
END FUNCTION
```

### Examples

- **Bug 1.1**: `!pip install pyreader` → installs wrong package; should be `pyreadr`
- **Bug 1.2**: `from src.data_engineering.corpus_loader import load_rds_corpus, filter_corpus` → raises `ModuleNotFoundError` because `src/` is not on Colab's filesystem
- **Bug 1.3**: Cell imports `load_rds_corpus` (which calls `pyreadr.read_r()`) but the very next cell loads data via `pd.read_csv("CBS_dataset_v1.0.csv")` — the RDS import is dead code
- **Bug 1.4**: `plt.savefig(PROJECT_ROOT / 'data' / 'processed' / 'corpus_overview.png')` → `FileNotFoundError` if `data/processed/` doesn't exist yet
- **Bug 1.5**: `import pandas as pd` appears in 3+ cells; `from pathlib import Path` appears in 4+ cells
- **Bug 1.6**: FinBERT uses `batch_size=32` instead of ≥64 for A100's 80GB VRAM; no `bf16` enabled
- **Bug 1.7**: Some cells call `output_path.mkdir(parents=True, exist_ok=True)` while others just write to `data/processed/` assuming it exists

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Corpus filtering logic: English-only filter, min 500 chars, date parsing, deduplication by SHA-256 content hash — must produce equivalent document_registry.parquet
- Text segmentation: paragraph splitting (min 80 chars), sentence-window chunking (target 256 tokens, max 512, overlap 1 sentence) — same chunk_registry schema
- LM Lexicon scoring: same LM_NEGATIVE, LM_POSITIVE, LM_UNCERTAINTY word sets; same tokenize_simple logic; same score columns (lm_neg, lm_pos, lm_net, lm_uncertainty, token_count)
- Hawk-Dove scoring: same HAWKISH_TERMS and DOVISH_TERMS dictionaries; same multi-word + single-word matching; same output columns (hd_hawk, hd_dove, hd_net, hawk_count, dove_count)
- FinBERT scoring: ProsusAI/finbert model, softmax probabilities, same output columns (finbert_pos, finbert_neg, finbert_neu, finbert_sentiment)
- Market data pipeline structures: same event-window extraction logic, label computation, chunk matching
- Daily label builder: same Δy₂ computation (US_2Y diff × 100 = basis points)
- 4-phase pipeline sequence: Data Engineering → Baselines → Advanced ML → Robustness

**Scope:**
All analytical computations, scoring algorithms, data schemas, and pipeline ordering must remain functionally identical. Only the delivery mechanism changes (from `src.*` imports to inlined code), along with infrastructure fixes (package name, directories, GPU config).

## Hypothesized Root Cause

Based on the bug description and notebook analysis, the root causes are:

1. **Typo in Package Name (Bug 1.1)**: The install cell lists `pyreader` instead of `pyreadr`. This is a simple typo — `pyreader` is either non-existent or a different package entirely.

2. **Structural Assumption Mismatch (Bug 1.2)**: The notebook was developed locally where `sys.path.insert(0, str(PROJECT_ROOT))` makes `src.*` importable. On Colab, the `src/` directory doesn't exist on the filesystem — only files uploaded to Google Drive are available. The notebook never clones the repo or uploads `src/`.

3. **Dead Code from Iteration (Bug 1.3)**: The `corpus_loader.py` module was designed for `.rds` files, but the actual workflow pivoted to using a pre-exported CSV. The import of `load_rds_corpus` is leftover from an earlier iteration.

4. **Missing Directory Guards (Bugs 1.4, 1.7)**: The notebook was developed incrementally — some cells were written after directories already existed from prior runs. First-time execution on a fresh Colab instance fails because no cell creates all required directories upfront.

5. **Copy-Paste Development (Bug 1.5)**: Cells were likely developed independently (possibly across multiple notebook files) and combined, carrying duplicate imports.

6. **Default Configurations Not A100-Aware (Bug 1.6)**: The `finbert_scorer.py` defaults to `batch_size=32` which is conservative for A100's 80GB HBM2e. No mixed-precision is configured because the code was written to also work on CPU.

## Correctness Properties

Property 1: Bug Condition - Notebook Executes End-to-End Without Errors

_For any_ cell in the fixed notebook when executed sequentially on Google Colab (A100 GPU runtime), the cell SHALL complete without raising `ModuleNotFoundError`, `FileNotFoundError`, `ImportError`, or package installation failures, and GPU-intensive cells SHALL utilize mixed-precision and batch sizes appropriate for A100.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

Property 2: Preservation - Analytical Output Equivalence

_For any_ analytical computation (corpus filtering, text segmentation, lexicon scoring, FinBERT inference, label construction) in the fixed notebook, the function SHALL produce output DataFrames with identical schemas and statistically equivalent values compared to running the same logic via the original `src/` modules on the same input data, preserving all word lists, hash algorithms, filtering thresholds, and scoring formulas.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `notebooks/Full_Pipeline.ipynb`

**Specific Changes**:

1. **Fix Package Name (Bug 1.1)**: In the install cell, replace `pyreader` with `pyreadr` in the `!pip install` command.

2. **Inline All src.* Modules (Bug 1.2)**: Remove all `from src.data_engineering.*` and `from src.baselines.*` import statements. Replace them with self-contained code cells that define the same functions directly:
   - Inline `corpus_loader.py` functions: `compute_content_hash`, `filter_corpus`, `build_document_registry`
   - Inline `text_segmenter.py` functions: `segment_into_paragraphs`, `segment_into_windows`, `build_chunk_registry`
   - Inline `daily_label_builder.py` functions: `compute_daily_yield_changes`, `match_labels_to_documents`, `match_labels_to_chunks`
   - Inline `lm_lexicon.py`: `LM_NEGATIVE`, `LM_POSITIVE`, `LM_UNCERTAINTY` sets + `compute_lm_scores`, `score_dataframe`
   - Inline `hawk_dove_lexicon.py`: `HAWKISH_TERMS`, `DOVISH_TERMS` + `compute_hd_scores`, `score_dataframe`
   - Inline `finbert_scorer.py`: `load_finbert`, `score_batch`, `score_dataframe`
   - Inline `market_data_pipeline.py` structures as needed
   - Inline `run_baselines.py`: `bootstrap_correlation`, `evaluate_baseline`, `aggregate_to_event_level`

3. **Align Corpus Loading with CSV Reality (Bug 1.3)**: Remove the `load_rds_corpus` import and any `pyreadr.read_r()` calls from the corpus loading cell. Keep the existing `pd.read_csv("CBS_dataset_v1.0.csv")` approach and wire it directly into the inlined `filter_corpus` and `build_document_registry` functions.

4. **Add Directory Creation Before Saves (Bug 1.4)**: Create a single "Setup Directories" cell near the top (after PROJECT_ROOT is set) that creates all output directories:
   ```python
   for subdir in ['data/processed', 'data/market', 'models', 'outputs']:
       (PROJECT_ROOT / subdir).mkdir(parents=True, exist_ok=True)
   ```
   Additionally, add `os.makedirs(parent, exist_ok=True)` guards before any `plt.savefig()` or `.to_parquet()` calls that target dynamic paths.

5. **Consolidate Imports (Bug 1.5)**: Create a single "Imports" cell at the top of the notebook containing all library imports used throughout:
   ```python
   import os, sys, re, hashlib, logging
   from pathlib import Path
   from typing import Optional, List, Dict
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from scipy import stats
   from tqdm import tqdm
   import spacy
   import torch
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   ```
   Remove all duplicate imports from subsequent cells.

6. **Optimize GPU Configuration (Bug 1.6)**: Add an A100 GPU configuration cell:
   - Set FinBERT `batch_size=128` (A100 80GB can handle this comfortably for 512-token inputs)
   - Enable `torch.backends.cuda.matmul.allow_tf32 = True` for A100 tensor cores
   - Use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` for inference
   - Set `DataLoader(num_workers=2)` for data loading
   - For LoRA fine-tuning: enable `bf16=True` in TrainingArguments, set `per_device_train_batch_size=16` or higher

7. **Standardize Directory Handling (Bug 1.7)**: Ensure every cell that writes output uses the centrally-created directories and does not redundantly call `mkdir`. The single setup cell handles all directory creation; individual cells simply write to the guaranteed-existing paths.

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bugs on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bugs BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Execute the unfixed notebook cells in sequence on a Colab-like environment and document the exact errors. Specifically, verify that:
- The `pyreader` package either fails to install or is the wrong package
- `from src.data_engineering.*` raises `ModuleNotFoundError`
- `plt.savefig()` to non-existent directories raises `FileNotFoundError`

**Test Cases**:
1. **Package Install Test**: Run `!pip install pyreader` and verify it either fails or installs a non-functional package (will fail on unfixed code)
2. **Module Import Test**: Run `from src.data_engineering.corpus_loader import load_rds_corpus` without `src/` on path (will fail on unfixed code)
3. **Directory Save Test**: Attempt `plt.savefig()` to `data/processed/test.png` without prior `mkdir` (will fail on unfixed code)
4. **GPU Config Test**: Check that FinBERT scoring cell uses `batch_size=32` without bf16 (suboptimal on unfixed code)

**Expected Counterexamples**:
- `ModuleNotFoundError: No module named 'src'`
- `FileNotFoundError: [Errno 2] No such file or directory: '/content/drive/MyDrive/nlp_econ/data/processed/corpus_overview.png'`
- Possible causes: missing sys.path, missing directory creation, wrong package name

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed notebook executes without errors.

**Pseudocode:**
```
FOR ALL cell WHERE isBugCondition(cell) DO
  result := execute_fixed_cell(cell)
  ASSERT no_error_raised(result)
  ASSERT output_matches_expected_schema(result)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all analytical computations, the fixed notebook produces the same results as the original src/ modules.

**Pseudocode:**
```
FOR ALL computation WHERE NOT isBugCondition(computation) DO
  ASSERT inlined_function(input) == original_src_function(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many random text inputs to verify scoring functions produce identical results
- It catches edge cases in tokenization, hash computation, and filtering
- It provides strong guarantees that the inlining didn't alter any logic

**Test Plan**: Run the original `src/` module functions and the inlined versions on the same inputs, asserting output equality.

**Test Cases**:
1. **Corpus Filter Preservation**: Verify that `filter_corpus()` inlined produces the same filtered DataFrame as the `src/` version for the same raw input
2. **LM Scoring Preservation**: Verify `compute_lm_scores()` inlined produces identical scores to `src.baselines.lm_lexicon.compute_lm_scores()` for random text inputs
3. **HD Scoring Preservation**: Verify `compute_hd_scores()` inlined produces identical scores to `src.baselines.hawk_dove_lexicon.compute_hd_scores()` for random text inputs
4. **Document Registry Preservation**: Verify `build_document_registry()` produces same schema and deduplication behavior

### Unit Tests

- Test that `pyreadr` is importable after the fixed install cell
- Test that all inlined functions are callable (no `NameError`)
- Test that directories exist after the setup cell
- Test that GPU config cell properly sets bf16 and batch size
- Test edge cases: empty text scoring, single-character text, text with no matches

### Property-Based Tests

- Generate random text strings (ASCII, Unicode) and verify `compute_lm_scores` produces valid float outputs in [0, 1] for fractions
- Generate random DataFrames with 'text' column and verify `filter_corpus` filtering is deterministic and monotonically reduces row count
- Generate random chunk registries and verify `build_chunk_registry` always produces valid chunk_id format (`{doc_id}_{type_char}_{position:04d}`)
- Test that `compute_content_hash` is deterministic: same input always yields same SHA-256

### Integration Tests

- Execute all cells in sequence on a clean Colab-like environment and verify zero errors
- Verify that `document_registry.parquet` is written successfully with expected columns
- Verify that `chunk_registry_paragraph.parquet` is written with expected schema
- Verify that FinBERT scoring utilizes GPU (torch.cuda.is_available() == True and model is on 'cuda')
- Verify end-to-end: raw CSV → filtered → segmented → scored → saved, producing the same column schemas as the original pipeline
