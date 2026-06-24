import json
import re

NOTEBOOK_PATH = r"c:\Users\abhis\Desktop\Abhishek Kumar\Ecnomic project\notebooks\Full_Pipeline.ipynb"

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
code_cells = []
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        code_cells.append((i, source))

all_source = '\n'.join(s for _, s in code_cells)

print(f"Total cells: {len(cells)}")
print(f"Code cells: {len(code_cells)}")
print()

# Check 1: pyreader gone?
pyreader_count = sum(1 for _, s in code_cells if re.search(r'\bpyreader\b', s))
print(f"1. 'pyreader' occurrences: {pyreader_count} {'✓' if pyreader_count == 0 else '✗'}")

# Check 2: src imports gone?
src_imports = sum(1 for _, s in code_cells if 'from src.' in s)
print(f"2. 'from src.*' imports: {src_imports} {'✓' if src_imports == 0 else '✗'}")

# Check 3: load_rds_corpus gone?
rds_refs = sum(1 for _, s in code_cells if 'load_rds_corpus' in s)
print(f"3. 'load_rds_corpus' references: {rds_refs} {'✓' if rds_refs == 0 else '✗'}")

# Check 4: pyreadr.read_r gone?
read_r_refs = sum(1 for _, s in code_cells if 'pyreadr.read_r' in s)
print(f"4. 'pyreadr.read_r' references: {read_r_refs} {'✓' if read_r_refs == 0 else '✗'}")

# Check 5: GPU optimization
has_tf32 = 'torch.backends.cuda.matmul.allow_tf32 = True' in all_source
has_bf16 = 'bfloat16' in all_source or 'autocast' in all_source
batch_matches = re.findall(r'batch_size\s*=\s*(\d+)', all_source)
has_big_batch = any(int(m) >= 64 for m in batch_matches) if batch_matches else False
print(f"5. GPU: TF32={'✓' if has_tf32 else '✗'}, bf16={'✓' if has_bf16 else '✗'}, batch≥64={'✓' if has_big_batch else '✗'}")

# Check 6: Consolidated dirs
dir_strings = ["data/processed", "data/market", "models", "outputs"]
dir_cells = [(i, s) for i, s in code_cells if all(d in s for d in dir_strings)]
print(f"6. Directory setup cell(s): {[i for i, _ in dir_cells]} {'✓' if len(dir_cells) >= 1 else '✗'}")

# Check 7: Inlined functions
required_funcs = [
    'compute_content_hash', 'filter_corpus', 'build_document_registry',
    'segment_into_paragraphs', 'segment_into_windows', 'build_chunk_registry',
    'compute_daily_yield_changes', 'match_labels_to_documents', 'match_labels_to_chunks',
    'compute_lm_scores', 'compute_hd_scores', 'load_finbert', 'score_batch',
    'bootstrap_correlation', 'compute_directional_accuracy', 'evaluate_baseline',
    'aggregate_to_event_level', 'validate_tick_data'
]
found_funcs = [f for f in required_funcs if re.search(rf'def\s+{f}\b', all_source)]
missing_funcs = [f for f in required_funcs if f not in found_funcs]
print(f"7. Inlined functions: {len(found_funcs)}/{len(required_funcs)} {'✓' if not missing_funcs else '✗'}")
if missing_funcs:
    print(f"   MISSING: {missing_funcs}")

# Check 8: Consolidated imports (each in 1 cell only)
pandas_cells = [i for i, s in code_cells if re.search(r'^\s*import pandas', s, re.MULTILINE)]
numpy_cells = [i for i, s in code_cells if re.search(r'^\s*import numpy', s, re.MULTILINE)]
pathlib_cells = [i for i, s in code_cells if re.search(r'^\s*from pathlib import Path', s, re.MULTILINE)]
print(f"8. Import cells: pandas={pandas_cells}, numpy={numpy_cells}, pathlib={pathlib_cells}")
imports_ok = len(pandas_cells) <= 1 and len(numpy_cells) <= 1 and len(pathlib_cells) <= 1
print(f"   Each in ≤1 cell: {'✓' if imports_ok else '✗'}")

# Check 9: No adhoc mkdir with writes
adhoc = [i for i, s in code_cells if '.mkdir' in s and ('to_parquet' in s or 'savefig' in s)]
print(f"9. Ad-hoc mkdir in write cells: {adhoc} {'✓' if not adhoc else '✗'}")

# Check 10: First few cells structure
print(f"\n--- First 6 code cells (abbreviated) ---")
for idx, (cell_idx, source) in enumerate(code_cells[:6]):
    first_line = source.split('\n')[0][:80]
    print(f"  Code cell {idx} (notebook cell {cell_idx}): {first_line}")

# FINAL VERDICT
all_pass = (pyreader_count == 0 and src_imports == 0 and rds_refs == 0 and 
            read_r_refs == 0 and has_tf32 and has_bf16 and has_big_batch and
            len(dir_cells) >= 1 and not missing_funcs and imports_ok and not adhoc)
print(f"\n{'='*50}")
print(f"FINAL VERDICT: {'ALL CHANGES CONFIRMED IN NOTEBOOK ✓' if all_pass else 'SOME ISSUES REMAIN ✗'}")
print(f"{'='*50}")
