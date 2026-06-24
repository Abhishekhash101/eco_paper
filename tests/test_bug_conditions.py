"""
Bug Condition Exploration Tests for Full_Pipeline.ipynb

These tests parse the notebook JSON and verify that seven bug conditions exist
in the UNFIXED notebook. All assertions are written so that they FAIL on the
buggy notebook, confirming the bugs are present.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7**
"""

import json
import re
from pathlib import Path

import pytest

# Load the notebook once for all tests
NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "Full_Pipeline.ipynb"


@pytest.fixture(scope="module")
def notebook():
    """Load the Full_Pipeline.ipynb as parsed JSON."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return nb


@pytest.fixture(scope="module")
def code_cells(notebook):
    """Extract all code cells' source as joined strings."""
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            cells.append(source)
    return cells


@pytest.fixture(scope="module")
def all_source(code_cells):
    """Concatenate all code cell sources for global checks."""
    return "\n".join(code_cells)


class TestBug1_1_WrongPackageName:
    """Bug 1.1: Assert that the install cell installs pyreadr (not pyreader).

    Verify by checking that 'pyreader' does NOT appear in install commands.
    On the unfixed notebook, this test will FAIL because 'pyreader' IS present.
    """

    def test_pyreader_not_in_install_commands(self, code_cells):
        """Assert that no install cell contains the misspelled 'pyreader' package."""
        install_cells = [c for c in code_cells if "pip install" in c or "pip -q install" in c]
        assert len(install_cells) > 0, "No install cells found in notebook"

        # Check that 'pyreader' does NOT appear in any install cell
        # We use a regex that matches 'pyreader' but NOT 'pyreadr'
        for cell in install_cells:
            assert not re.search(r'\bpyreader\b', cell), (
                f"Bug 1.1 CONFIRMED: Found 'pyreader' (wrong package name) in install cell. "
                f"Should be 'pyreadr'."
            )


class TestBug1_2_BrokenSrcImports:
    """Bug 1.2: Assert that NO cell contains 'from src.data_engineering' or
    'from src.baselines' imports.

    On the unfixed notebook, this test will FAIL because src.* imports exist.
    """

    def test_no_src_data_engineering_imports(self, code_cells):
        """Assert that no cell imports from src.data_engineering."""
        for cell in code_cells:
            assert "from src.data_engineering" not in cell, (
                f"Bug 1.2 CONFIRMED: Found 'from src.data_engineering' import. "
                f"These should be inlined for Colab compatibility."
            )

    def test_no_src_baselines_imports(self, code_cells):
        """Assert that no cell imports from src.baselines."""
        for cell in code_cells:
            assert "from src.baselines" not in cell, (
                f"Bug 1.2 CONFIRMED: Found 'from src.baselines' import. "
                f"These should be inlined for Colab compatibility."
            )


class TestBug1_3_RDSCSVMismatch:
    """Bug 1.3: Assert that no cell references load_rds_corpus or pyreadr.read_r()
    for data loading — corpus loading should use pd.read_csv() directly.

    On the unfixed notebook, this test will FAIL because load_rds_corpus is imported.
    """

    def test_no_load_rds_corpus_reference(self, code_cells):
        """Assert that no cell references load_rds_corpus."""
        for cell in code_cells:
            assert "load_rds_corpus" not in cell, (
                f"Bug 1.3 CONFIRMED: Found 'load_rds_corpus' reference. "
                f"The notebook actually uses pd.read_csv() so this is dead code."
            )

    def test_no_pyreadr_read_r_call(self, code_cells):
        """Assert that no cell calls pyreadr.read_r() for data loading."""
        for cell in code_cells:
            assert "pyreadr.read_r" not in cell, (
                f"Bug 1.3 CONFIRMED: Found 'pyreadr.read_r()' call. "
                f"The notebook loads CSV data, not RDS files."
            )


class TestBug1_4_MissingDirectoryCreation:
    """Bug 1.4: Assert that a directory creation cell exists before any
    plt.savefig() or .to_parquet() call, creating data/processed, data/market,
    models, outputs.

    On the unfixed notebook, this test will FAIL because there is no single
    consolidated directory creation cell that creates ALL required directories.
    """

    def test_consolidated_directory_setup_exists(self, code_cells):
        """Assert that a single cell creates all required output directories
        BEFORE any savefig/to_parquet call."""
        required_dirs = ["data/processed", "data/market", "models", "outputs"]

        # Find the first cell that writes output (savefig or to_parquet)
        first_write_idx = None
        for i, cell in enumerate(code_cells):
            if "plt.savefig" in cell or ".to_parquet" in cell:
                first_write_idx = i
                break

        assert first_write_idx is not None, "No savefig or to_parquet calls found"

        # Check if there's a cell BEFORE the first write that creates ALL dirs
        setup_found = False
        for i in range(first_write_idx):
            cell = code_cells[i]
            # Check if this cell creates all required directories
            dirs_created = sum(1 for d in required_dirs if d in cell)
            if dirs_created >= len(required_dirs):
                setup_found = True
                break

        assert setup_found, (
            f"Bug 1.4 CONFIRMED: No consolidated directory creation cell found before "
            f"first file write (cell index {first_write_idx}). Required directories: "
            f"{required_dirs}"
        )


class TestBug1_5_RedundantImports:
    """Bug 1.5: Assert that import pandas, import numpy, from pathlib import Path
    each appear only ONCE in the notebook (in a single consolidated cell).

    On the unfixed notebook, this test will FAIL because these imports are
    duplicated across multiple cells.
    """

    def test_pandas_imported_once(self, code_cells):
        """Assert that 'import pandas' appears in only one cell."""
        cells_with_pandas = [
            c for c in code_cells
            if re.search(r'^\s*import pandas', c, re.MULTILINE)
        ]
        assert len(cells_with_pandas) <= 1, (
            f"Bug 1.5 CONFIRMED: 'import pandas' appears in {len(cells_with_pandas)} "
            f"cells. Should appear in only 1 consolidated imports cell."
        )

    def test_numpy_imported_once(self, code_cells):
        """Assert that 'import numpy' appears in only one cell."""
        cells_with_numpy = [
            c for c in code_cells
            if re.search(r'^\s*import numpy', c, re.MULTILINE)
        ]
        assert len(cells_with_numpy) <= 1, (
            f"Bug 1.5 CONFIRMED: 'import numpy' appears in {len(cells_with_numpy)} "
            f"cells. Should appear in only 1 consolidated imports cell."
        )

    def test_pathlib_imported_once(self, code_cells):
        """Assert that 'from pathlib import Path' appears in only one cell."""
        cells_with_pathlib = [
            c for c in code_cells
            if re.search(r'^\s*from pathlib import Path', c, re.MULTILINE)
        ]
        assert len(cells_with_pathlib) <= 1, (
            f"Bug 1.5 CONFIRMED: 'from pathlib import Path' appears in "
            f"{len(cells_with_pathlib)} cells. Should appear in only 1 consolidated "
            f"imports cell."
        )


class TestBug1_6_SuboptimalGPU:
    """Bug 1.6: Assert that FinBERT scoring uses batch_size >= 64, enables
    bf16 or fp16, and sets torch.backends.cuda.matmul.allow_tf32 = True.

    On the unfixed notebook, this test will FAIL because the GPU configuration
    is not optimized for A100.
    """

    def test_finbert_batch_size_at_least_64(self, code_cells):
        """Assert that FinBERT scoring uses batch_size >= 64."""
        finbert_cells = [c for c in code_cells if "finbert" in c.lower() and "batch_size" in c]
        assert len(finbert_cells) > 0, "No FinBERT cells with batch_size found"

        for cell in finbert_cells:
            # Find batch_size=<number> patterns
            matches = re.findall(r'batch_size\s*=\s*(\d+)', cell)
            for match in matches:
                batch_size = int(match)
                assert batch_size >= 64, (
                    f"Bug 1.6 CONFIRMED: FinBERT uses batch_size={batch_size} "
                    f"which is suboptimal for A100 GPU. Should be >= 64."
                )

    def test_mixed_precision_enabled(self, all_source):
        """Assert that bf16 or fp16 mixed precision is enabled for inference."""
        has_bf16 = "bf16" in all_source or "bfloat16" in all_source
        has_fp16 = "fp16" in all_source
        has_autocast = "autocast" in all_source

        assert has_bf16 or has_fp16 or has_autocast, (
            "Bug 1.6 CONFIRMED: No mixed precision (bf16/fp16/autocast) found in "
            "the notebook. A100 GPU should use mixed precision for optimal performance."
        )

    def test_tf32_enabled(self, all_source):
        """Assert that TF32 is enabled for A100 tensor cores."""
        assert "allow_tf32" in all_source, (
            "Bug 1.6 CONFIRMED: 'torch.backends.cuda.matmul.allow_tf32 = True' "
            "not found. A100 should enable TF32 for tensor core optimization."
        )


class TestBug1_7_InconsistentDirectoryHandling:
    """Bug 1.7: Assert that all output path references use centrally-created
    directories without ad-hoc mkdir scattered across cells.

    On the unfixed notebook, this test will FAIL because some cells that write
    output assume directories exist without creating them, while one ad-hoc cell
    creates data/processed inline. There is no centralized setup for ALL dirs.
    """

    def test_all_output_dirs_created_centrally(self, code_cells):
        """Assert that ALL required output directories (data/processed, data/market,
        models, outputs) are created in a SINGLE early setup cell, not ad-hoc
        in individual write cells or not at all."""
        required_dirs = ["data/processed", "data/market", "models", "outputs"]

        # Find cells that create directories (mkdir calls)
        # Exclude the PROJECT_ROOT.mkdir cell
        dir_creation_cells = []
        for i, cell in enumerate(code_cells):
            if ".mkdir" in cell and "PROJECT_ROOT.mkdir" not in cell:
                dir_creation_cells.append((i, cell))

        # Check: is there a single cell that creates ALL required dirs?
        has_centralized_setup = False
        for idx, cell in dir_creation_cells:
            dirs_covered = sum(1 for d in required_dirs if d in cell)
            if dirs_covered >= len(required_dirs):
                has_centralized_setup = True
                break

        # Also check: are ad-hoc mkdir calls present inline with write operations?
        adhoc_mkdir_with_writes = []
        for i, cell in enumerate(code_cells):
            if ".mkdir" in cell and ("to_parquet" in cell or "savefig" in cell):
                adhoc_mkdir_with_writes.append(i)

        # The test fails if:
        # 1) No centralized setup exists for all dirs, OR
        # 2) Ad-hoc mkdir calls are scattered alongside write operations
        assert has_centralized_setup and len(adhoc_mkdir_with_writes) == 0, (
            f"Bug 1.7 CONFIRMED: Directory handling is inconsistent. "
            f"Centralized setup for all dirs: {has_centralized_setup}. "
            f"Ad-hoc mkdir calls mixed with writes in cells: {adhoc_mkdir_with_writes}. "
            f"Should have one setup cell creating {required_dirs} with no ad-hoc mkdir elsewhere."
        )
