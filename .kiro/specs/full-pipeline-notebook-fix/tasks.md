# Implementation Plan

## Overview

Fix the `Full_Pipeline.ipynb` notebook to run end-to-end on Google Colab (A100 GPU) by addressing seven interrelated bugs: wrong package name, broken `src.*` imports, RDS/CSV mismatch, missing directory creation, redundant imports, suboptimal GPU utilization, and inconsistent directory handling. The fix inlines all `src/` module logic into notebook cells, consolidates imports and directory setup, and adds A100 GPU optimization.

## Tasks

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Notebook Cells Fail on Colab Environment
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bugs exist
  - **DO NOT attempt to fix the test or the code when it fails**
  - Write a Python test script that parses `Full_Pipeline.ipynb` JSON and checks seven bug conditions: wrong package name (pyreader), broken src imports, RDS/CSV mismatch, missing directory creation, redundant imports, suboptimal GPU utilization, inconsistent directory handling
  - Run test on UNFIXED notebook - Test FAILS confirming all 7 bugs exist
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

- [x] 2. Write preservation property tests
  - **Property 2: Preservation** - Analytical Logic Equivalence
  - Write property-based tests using `hypothesis` library that verify analytical functions produce correct outputs
  - Import functions directly from `src/` modules to establish baseline behavior
  - Verify all property tests PASS on unfixed code (the analytical logic itself is correct)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 3. Fix package name and consolidate imports cell
  - Replace `pyreader` with `pyreadr` in the `!pip install` cell
  - Create a single consolidated imports cell immediately after Drive mount + PROJECT_ROOT setup
  - Remove ALL duplicate imports from subsequent cells
  - _Requirements: 2.1, 2.5_

- [x] 4. Add consolidated directory creation cell
  - Add a single Setup Directories cell after PROJECT_ROOT that creates: data/processed, data/market, models, outputs
  - Remove ALL scattered mkdir calls from individual cells
  - _Requirements: 2.4, 2.7_

- [x] 5. Inline corpus_loader.py functions and align with CSV workflow
  - Remove `from src.data_engineering.corpus_loader import ...` statement
  - Remove any references to `load_rds_corpus` or `pyreadr.read_r()`
  - Add code cell with inlined `compute_content_hash`, `filter_corpus`, `build_document_registry`
  - Wire existing `pd.read_csv("CBS_dataset_v1.0.csv")` into inlined functions
  - _Requirements: 2.2, 2.3, 3.1, 3.2_

- [x] 6. Inline text_segmenter.py functions
  - Add code cell with inlined `load_spacy_model`, `segment_into_paragraphs`, `segment_into_windows`, `build_chunk_registry`
  - Preserve chunk_id format: `{doc_id}_{type_char}_{position:04d}`
  - Remove any `from src.data_engineering.text_segmenter import ...`
  - _Requirements: 2.2, 3.3_

- [x] 7. Inline daily_label_builder.py functions
  - Add code cell with inlined `load_yield_data`, `compute_daily_yield_changes`, `match_labels_to_documents`, `match_labels_to_chunks`
  - Preserve delta-y2 computation: diff() x 100 for basis points
  - Remove any `from src.data_engineering.daily_label_builder import ...`
  - _Requirements: 2.2, 3.7_

- [x] 8. Inline lm_lexicon.py and hawk_dove_lexicon.py functions
  - Add code cell with inlined LM_NEGATIVE, LM_POSITIVE, LM_UNCERTAINTY word sets, tokenize_simple, compute_lm_scores, HAWKISH_TERMS, DOVISH_TERMS dictionaries, compute_hd_scores, and score_dataframe wrappers
  - All word lists must be character-for-character identical to src/ versions
  - Remove any `from src.baselines.lm_lexicon import ...` and `from src.baselines.hawk_dove_lexicon import ...`
  - _Requirements: 2.2, 3.4, 3.5_

- [x] 9. Inline finbert_scorer.py with A100 GPU optimization
  - Add code cell with inlined `load_finbert`, `score_batch` (with bf16 autocast), `score_dataframe` (batch_size=128 for A100)
  - Add A100 GPU configuration: `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`
  - Use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` in score_batch inference
  - Remove any `from src.baselines.finbert_scorer import ...`
  - _Requirements: 2.2, 2.6, 3.6_

- [x] 10. Inline run_baselines.py evaluation functions
  - Add code cell with inlined `bootstrap_correlation`, `compute_directional_accuracy`, `evaluate_baseline`, `aggregate_to_event_level`
  - Remove any `from src.baselines.run_baselines import ...`
  - _Requirements: 2.2, 3.8_

- [x] 11. Inline market_data_pipeline.py structures
  - Add code cell with inlined REQUIRED_TICK_COLUMNS, EXPECTED_INSTRUMENTS, validate_tick_data, extract_event_window, compute_yield_change, build_event_labels, match_labels_to_chunks
  - Remove any `from src.data_engineering.market_data_pipeline import ...`
  - _Requirements: 2.2, 3.7_

- [x] 12. Verify bug condition exploration test now passes
  - Re-run the SAME test from task 1 - do NOT write a new test
  - **EXPECTED OUTCOME**: Test PASSES (confirms all 7 bugs are fixed)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [x] 13. Verify preservation tests still pass
  - Re-run the SAME tests from task 2 - do NOT write new tests
  - Verify compute_lm_scores, compute_hd_scores, compute_content_hash, filter_corpus, segment_into_paragraphs all produce identical outputs
  - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 14. Checkpoint - Ensure all tests pass
  - Run the full test suite (exploration + preservation tests)
  - Verify the fixed notebook can be loaded as valid JSON
  - Confirm no `src.*` imports remain anywhere in the notebook
  - Confirm `pyreader` does not appear (only `pyreadr`)
  - Confirm all output directories are created in a single setup cell
  - Confirm GPU optimization settings are present

## Task Dependency Graph

```json
{
  "waves": [
    ["1", "2"],
    ["3", "4"],
    ["5", "6", "7"],
    ["8", "9", "10", "11"],
    ["12", "13"],
    ["14"]
  ]
}
```

## Notes

- All inlined functions must be character-for-character identical in logic to the `src/` module versions — only the import mechanism changes
- The notebook is the single file being modified: `notebooks/Full_Pipeline.ipynb`
- Testing is done by parsing the `.ipynb` JSON structure to verify structural correctness
- Property-based tests validate analytical equivalence between `src/` modules and inlined versions
- The A100 GPU optimization (batch_size=128, bf16, TF32) is specific to the Colab A100 runtime
