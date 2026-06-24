"""
Comprehensive verification script for Full_Pipeline.ipynb
Performs all 12 checks as specified.
"""
import json
import sys
import re
from pathlib import Path

NOTEBOOK_PATH = Path(r"c:\Users\abhis\Desktop\Abhishek Kumar\Ecnomic project\notebooks\Full_Pipeline.ipynb")

results = {}

# ============================================================
# CHECK 1: Valid JSON
# ============================================================
print("=" * 70)
print("CHECK 1: Valid JSON")
print("=" * 70)
try:
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb.get('cells', [])
    cell_count = len(cells)
    print(f"  PASS - Notebook parses as valid JSON. Cell count: {cell_count}")
    results['1_valid_json'] = 'PASS'
except Exception as e:
    print(f"  FAIL - JSON parse error: {e}")
    results['1_valid_json'] = 'FAIL'
    sys.exit(1)

# Helper: get code cells with their index
code_cells = []
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        code_cells.append((i, source))

# ============================================================
# CHECK 3: No src.* imports remain
# ============================================================
print("\n" + "=" * 70)
print("CHECK 3: No src.* imports remain")
print("=" * 70)
src_import_patterns = [
    r'from\s+src\.data_engineering',
    r'from\s+src\.baselines',
    r'import\s+src\.',
]
found_src_imports = []
for idx, source in code_cells:
    for pat in src_import_patterns:
        if re.search(pat, source):
            found_src_imports.append((idx, pat))

if not found_src_imports:
    print("  PASS - No src.* imports found")
    results['3_no_src_imports'] = 'PASS'
else:
    print(f"  FAIL - Found {len(found_src_imports)} src.* imports:")
    for idx, pat in found_src_imports:
        print(f"    Cell {idx}: matched pattern '{pat}'")
    results['3_no_src_imports'] = 'FAIL'

# ============================================================
# CHECK 4: No pyreader typo
# ============================================================
print("\n" + "=" * 70)
print("CHECK 4: No 'pyreader' typo (should be 'pyreadr')")
print("=" * 70)
pyreader_found = []
for idx, source in code_cells:
    # Match 'pyreader' but not 'pyreadr'
    if re.search(r'pyreader(?!r)', source):
        pyreader_found.append(idx)

if not pyreader_found:
    print("  PASS - No 'pyreader' typo found")
    results['4_no_pyreader_typo'] = 'PASS'
else:
    print(f"  FAIL - Found 'pyreader' in cells: {pyreader_found}")
    results['4_no_pyreader_typo'] = 'FAIL'

# ============================================================
# CHECK 5: Consolidated imports cell
# ============================================================
print("\n" + "=" * 70)
print("CHECK 5: Consolidated imports (pandas, numpy, Path in ONE cell)")
print("=" * 70)
import_checks = {
    'import pandas': [],
    'import numpy': [],
    'from pathlib import Path': [],
}
for idx, source in code_cells:
    for imp in import_checks:
        if imp in source:
            import_checks[imp].append(idx)

all_single = True
for imp, locations in import_checks.items():
    if len(locations) == 1:
        print(f"  '{imp}' appears in cell {locations[0]} only - OK")
    elif len(locations) == 0:
        print(f"  '{imp}' NOT FOUND - WARNING")
        all_single = False
    else:
        print(f"  '{imp}' appears in cells {locations} - MULTIPLE!")
        all_single = False

results['5_consolidated_imports'] = 'PASS' if all_single else 'FAIL'

# ============================================================
# CHECK 6: Consolidated directory setup
# ============================================================
print("\n" + "=" * 70)
print("CHECK 6: Consolidated directory setup (one cell with all 4 dirs, before writes)")
print("=" * 70)
dir_strings = ["data/processed", "data/market", "models", "outputs"]
dir_setup_cells = []
for idx, source in code_cells:
    if all(d in source for d in dir_strings):
        dir_setup_cells.append(idx)

first_write_cell = None
for idx, source in code_cells:
    if 'plt.savefig' in source or '.to_parquet' in source:
        first_write_cell = idx
        break

if dir_setup_cells:
    print(f"  Directory setup cell(s): {dir_setup_cells}")
    if first_write_cell is not None:
        print(f"  First write cell: {first_write_cell}")
        if dir_setup_cells[0] < first_write_cell:
            print("  PASS - Directory setup comes before first write")
            results['6_consolidated_dirs'] = 'PASS'
        else:
            print("  FAIL - Directory setup comes AFTER first write")
            results['6_consolidated_dirs'] = 'FAIL'
    else:
        print("  No write cells found - PASS (vacuously true)")
        results['6_consolidated_dirs'] = 'PASS'
else:
    print("  FAIL - No single cell contains all 4 directory strings")
    results['6_consolidated_dirs'] = 'FAIL'

# ============================================================
# CHECK 7: No ad-hoc mkdir in write cells
# ============================================================
print("\n" + "=" * 70)
print("CHECK 7: No ad-hoc mkdir in write cells")
print("=" * 70)
adhoc_mkdir_cells = []
for idx, source in code_cells:
    has_mkdir = '.mkdir' in source
    has_write = 'to_parquet' in source or 'savefig' in source
    if has_mkdir and has_write:
        adhoc_mkdir_cells.append(idx)

if not adhoc_mkdir_cells:
    print("  PASS - No cells have both .mkdir and write operations")
    results['7_no_adhoc_mkdir'] = 'PASS'
else:
    print(f"  FAIL - Cells with both .mkdir and write ops: {adhoc_mkdir_cells}")
    results['7_no_adhoc_mkdir'] = 'FAIL'

# ============================================================
# CHECK 8: GPU Optimization present
# ============================================================
print("\n" + "=" * 70)
print("CHECK 8: GPU Optimization present")
print("=" * 70)
all_source = '\n'.join(s for _, s in code_cells)

gpu_checks = {
    'torch.backends.cuda.matmul.allow_tf32 = True': 'torch.backends.cuda.matmul.allow_tf32 = True' in all_source,
    'torch.backends.cudnn.allow_tf32 = True': 'torch.backends.cudnn.allow_tf32 = True' in all_source,
    'bfloat16 or autocast': 'bfloat16' in all_source or 'autocast' in all_source,
}

# Check batch_size >= 64 in FinBERT context
batch_size_ok = False
for idx, source in code_cells:
    if 'finbert' in source.lower() or 'FinBERT' in source or 'batch' in source.lower():
        matches = re.findall(r'batch_size\s*=\s*(\d+)', source)
        for m in matches:
            if int(m) >= 64:
                batch_size_ok = True
                break

gpu_checks['batch_size >= 64 in FinBERT cells'] = batch_size_ok

all_gpu_pass = True
for check, passed in gpu_checks.items():
    status = "FOUND" if passed else "NOT FOUND"
    print(f"  {check}: {status}")
    if not passed:
        all_gpu_pass = False

results['8_gpu_optimization'] = 'PASS' if all_gpu_pass else 'FAIL'

# ============================================================
# CHECK 9: All inlined functions present
# ============================================================
print("\n" + "=" * 70)
print("CHECK 9: All inlined functions present")
print("=" * 70)
required_functions = [
    'compute_content_hash',
    'filter_corpus',
    'build_document_registry',
    'segment_into_paragraphs',
    'segment_into_windows',
    'build_chunk_registry',
    ('load_yield_data', 'compute_daily_yield_changes'),  # Either one
    'match_labels_to_documents',
    'match_labels_to_chunks',
    'compute_lm_scores',
    'compute_hd_scores',
    'load_finbert',
    'score_batch',
    'bootstrap_correlation',
    'compute_directional_accuracy',
    'evaluate_baseline',
    'aggregate_to_event_level',
    'validate_tick_data',
]

all_funcs_found = True
for func in required_functions:
    if isinstance(func, tuple):
        # Either/or
        found = any(re.search(rf'def\s+{f}\b', all_source) for f in func)
        label = ' or '.join(func)
    else:
        found = bool(re.search(rf'def\s+{func}\b', all_source))
        label = func
    
    status = "FOUND" if found else "MISSING"
    print(f"  def {label}: {status}")
    if not found:
        all_funcs_found = False

results['9_inlined_functions'] = 'PASS' if all_funcs_found else 'FAIL'

# ============================================================
# CHECK 10: No Python syntax errors in cells
# ============================================================
print("\n" + "=" * 70)
print("CHECK 10: No Python syntax errors in code cells")
print("=" * 70)
syntax_errors = []
for idx, source in code_cells:
    # Filter out shell (!) and magic (%) commands including multi-line continuations
    lines = source.split('\n')
    filtered_lines = []
    in_shell_continuation = False
    for line in lines:
        stripped = line.strip()
        if in_shell_continuation:
            # This line is a continuation of a shell/magic command
            in_shell_continuation = stripped.endswith('\\')
            filtered_lines.append('')  # blank line placeholder
            continue
        if stripped.startswith('!') or stripped.startswith('%'):
            in_shell_continuation = stripped.endswith('\\')
            filtered_lines.append('pass  # magic/shell command')
        else:
            filtered_lines.append(line)
    filtered_source = '\n'.join(filtered_lines)
    
    try:
        compile(filtered_source, f'<cell_{idx}>', 'exec')
    except SyntaxError as e:
        syntax_errors.append((idx, str(e)))

if not syntax_errors:
    print("  PASS - No syntax errors in any code cell")
    results['10_no_syntax_errors'] = 'PASS'
else:
    print(f"  FAIL - {len(syntax_errors)} cells with syntax errors:")
    for idx, err in syntax_errors:
        print(f"    Cell {idx}: {err}")
    results['10_no_syntax_errors'] = 'FAIL'

# ============================================================
# CHECK 11: No load_rds_corpus or pyreadr.read_r references
# ============================================================
print("\n" + "=" * 70)
print("CHECK 11: No load_rds_corpus or pyreadr.read_r references")
print("=" * 70)
bad_refs = []
for idx, source in code_cells:
    if 'load_rds_corpus' in source:
        bad_refs.append((idx, 'load_rds_corpus'))
    if 'pyreadr.read_r' in source:
        bad_refs.append((idx, 'pyreadr.read_r'))

if not bad_refs:
    print("  PASS - No load_rds_corpus or pyreadr.read_r found")
    results['11_no_rds_refs'] = 'PASS'
else:
    print(f"  FAIL - Found references:")
    for idx, ref in bad_refs:
        print(f"    Cell {idx}: {ref}")
    results['11_no_rds_refs'] = 'FAIL'

# ============================================================
# CHECK 12: Notebook structure sanity
# ============================================================
print("\n" + "=" * 70)
print("CHECK 12: Notebook structure sanity")
print("=" * 70)
structure_issues = []

# First code cell should be pip install
if code_cells:
    first_idx, first_source = code_cells[0]
    if 'pip install' in first_source or '!pip' in first_source:
        print("  First code cell is pip install - OK")
    else:
        print(f"  First code cell (cell {first_idx}) is NOT pip install - WARNING")
        structure_issues.append("First cell not pip install")

# Early cells should have Drive mount and PROJECT_ROOT
early_cells = code_cells[:10]  # First 10 code cells
has_drive_mount = any('drive.mount' in s or 'PROJECT_ROOT' in s for _, s in early_cells)
if has_drive_mount:
    print("  Drive mount / PROJECT_ROOT in early cells - OK")
else:
    print("  Drive mount / PROJECT_ROOT NOT in early cells - WARNING")
    structure_issues.append("No Drive mount/PROJECT_ROOT in early cells")

# Consolidated imports should come early (first 10 code cells)
import_cell_indices = import_checks.get('import pandas', [])
if import_cell_indices:
    imp_idx = import_cell_indices[0]
    # Find its position among code cells
    code_cell_indices = [i for i, _ in code_cells]
    if imp_idx in code_cell_indices:
        pos = code_cell_indices.index(imp_idx)
        if pos < 10:
            print(f"  Consolidated imports at code cell position {pos} - OK (early)")
        else:
            print(f"  Consolidated imports at code cell position {pos} - WARNING (not early)")
            structure_issues.append("Imports not early enough")

# Directory setup should come before first write
if dir_setup_cells and first_write_cell is not None:
    if dir_setup_cells[0] < first_write_cell:
        print("  Directory setup before first write - OK")
    else:
        print("  Directory setup AFTER first write - ISSUE")
        structure_issues.append("Dir setup after first write")

if not structure_issues:
    results['12_structure_sanity'] = 'PASS'
else:
    results['12_structure_sanity'] = 'FAIL'

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Check':<45} {'Result':<10}")
print("-" * 55)

check_names = {
    '1_valid_json': '1. Valid JSON',
    '2_test_suite': '2. Test suite (run separately)',
    '3_no_src_imports': '3. No src.* imports',
    '4_no_pyreader_typo': '4. No pyreader typo',
    '5_consolidated_imports': '5. Consolidated imports',
    '6_consolidated_dirs': '6. Consolidated directory setup',
    '7_no_adhoc_mkdir': '7. No ad-hoc mkdir in write cells',
    '8_gpu_optimization': '8. GPU optimization present',
    '9_inlined_functions': '9. All inlined functions present',
    '10_no_syntax_errors': '10. No syntax errors',
    '11_no_rds_refs': '11. No load_rds_corpus/pyreadr.read_r',
    '12_structure_sanity': '12. Notebook structure sanity',
}

for key, name in check_names.items():
    status = results.get(key, 'NOT RUN')
    icon = "✓" if status == 'PASS' else ("✗" if status == 'FAIL' else "—")
    print(f"  {icon} {name:<43} {status}")

# Final verdict
all_pass = all(v == 'PASS' for k, v in results.items() if k != '2_test_suite')
print("\n" + "=" * 70)
if all_pass:
    print("OVERALL: ALL STATIC CHECKS PASSED")
else:
    failed = [k for k, v in results.items() if v == 'FAIL']
    print(f"OVERALL: {len(failed)} CHECK(S) FAILED: {failed}")
print("=" * 70)
