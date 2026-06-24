"""Fix the Full_Pipeline.ipynb to have robust PROJECT_ROOT detection."""
import json

with open('notebooks/Full_Pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix cell 4 — make PROJECT_ROOT detection robust
new_source = [
    'import sys\n',
    'import logging\n',
    'from pathlib import Path\n',
    '\n',
    'import pandas as pd\n',
    'import numpy as np\n',
    'import matplotlib.pyplot as plt\n',
    'import seaborn as sns\n',
    '\n',
    '# Project root detection (works from notebooks/ or project root)\n',
    'cwd = Path(".").resolve()\n',
    'if cwd.name == "notebooks":\n',
    '    PROJECT_ROOT = cwd.parent\n',
    'elif (cwd / "CBS_dataset_v1.0.rds").exists():\n',
    '    PROJECT_ROOT = cwd\n',
    'else:\n',
    '    PROJECT_ROOT = cwd.parent  # fallback\n',
    '\n',
    'sys.path.insert(0, str(PROJECT_ROOT))\n',
    '\n',
    '# Verify CBS file exists\n',
    'cbs_path = PROJECT_ROOT / "CBS_dataset_v1.0.rds"\n',
    'assert cbs_path.exists(), f"CBS dataset not found at {cbs_path}. Place it in project root."\n',
    '\n',
    '# Configure logging\n',
    'logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")\n',
    '\n',
    '# Display settings\n',
    'pd.set_option("display.max_columns", None)\n',
    'pd.set_option("display.max_colwidth", 100)\n',
    '\n',
    'print(f"Project root: {PROJECT_ROOT}")\n',
    'print(f"CBS dataset found: {cbs_path.exists()}")\n',
]

nb['cells'][4]['source'] = new_source

with open('notebooks/Full_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Fixed cell 4: robust PROJECT_ROOT detection + CBS file check')
