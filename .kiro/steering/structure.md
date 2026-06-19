# Project Structure

```
.
├── .kiro/
│   └── steering/          # AI assistant steering rules
├── CBS_dataset_v1.0.rds   # Primary dataset (R data format)
├── requirements.txt       # Python dependencies (pinned)
├── research_overview.pdf  # Research context document
└── research_roadmap.pdf   # Research planning document
```

## Notes
- The project is in early stage — expect notebooks, scripts, and model directories to be added
- `.rds` files are R serialized data; use `pyreadr` in Python or load in R directly
- PDF documents contain research context and should be referenced for domain understanding
- Large data and model artifacts should be tracked with DVC, not committed to git

## Expected Growth Areas
When expanding, follow this conventional layout:
- `data/` — raw and processed datasets
- `notebooks/` — exploratory Jupyter notebooks
- `src/` — reusable Python modules and pipelines
- `models/` — trained model checkpoints
- `experiments/` — experiment configs and results
- `scripts/` — standalone utility scripts
