# Tech Stack

## Language
- Python (primary)
- R dataset support (`.rds` files — read via `pyreadr` or R interop)

## Core Libraries
| Category | Libraries |
|----------|-----------|
| Data | numpy, pandas, scipy |
| Data Engineering | requests, beautifulsoup4, pdfplumber, PyPDF2, spacy |
| ML / Deep Learning | tensorflow, torch, scikit-learn, statsmodels |
| NLP / Transformers | transformers (Hugging Face), peft (LoRA), accelerate, datasets |
| Visualization | matplotlib, seaborn, plotly, umap-learn |
| Experiment Tracking | mlflow, wandb |
| Data Versioning | dvc |
| Notebooks | jupyter, ipykernel |

## Environment
- Platform: Windows
- Package management: pip with pinned versions in `requirements.txt`
- Note: `bitsandbytes` is disabled due to Windows compatibility issues

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (if needed)
python -m spacy download en_core_web_sm

# Launch Jupyter
jupyter notebook

# MLflow tracking UI
mlflow ui

# DVC operations
dvc pull    # Pull versioned data
dvc push    # Push versioned data
dvc repro   # Reproduce pipeline
```

## Conventions
- Pin all dependency versions exactly (e.g., `numpy==1.23.5`)
- Use DVC for large data files — do not commit data to git
- Track experiments with MLflow or W&B, not ad-hoc logging
