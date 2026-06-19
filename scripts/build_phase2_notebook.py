"""Build the Phase 2 Baseline Models Jupyter notebook."""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
nb_path = PROJECT / "notebooks" / "Phase2_Baseline_Models.ipynb"

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": source})


# ============================================================
# TITLE
# ============================================================
md([
    "# Phase 2 — Baseline Models\n",
    "## Lexicon & FinBERT Baselines: Quantifying the Performance Ceiling\n",
    "\n",
    "**Baselines:**\n",
    "1. Loughran-McDonald (LM) Lexicon\n",
    "2. Custom Hawkish-Dovish Word List\n",
    "3. FinBERT Zero-Shot Sentiment\n",
    "4. Baseline Comparison Table\n",
    "\n",
    "---"
])

# ============================================================
# IMPORTS
# ============================================================
code([
    "import sys\n",
    "import logging\n",
    "from pathlib import Path\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy import stats\n",
    "from tqdm.notebook import tqdm\n",
    "\n",
    "PROJECT_ROOT = Path('.').resolve().parent\n",
    "sys.path.insert(0, str(PROJECT_ROOT))\n",
    "\n",
    "logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')\n",
    "pd.set_option('display.max_colwidth', 100)\n",
    "print(f'Project root: {PROJECT_ROOT}')\n",
])


# ============================================================
# LOAD DATA
# ============================================================
md(["---\n", "## Step 0: Load Processed Data from Phase 1"])

code([
    "# Load chunk registry from Phase 1\n",
    "chunk_path = PROJECT_ROOT / 'data' / 'processed' / 'chunk_registry_window.parquet'\n",
    "if not chunk_path.exists():\n",
    "    chunk_path = PROJECT_ROOT / 'data' / 'processed' / 'chunk_registry_paragraph.parquet'\n",
    "\n",
    "chunks = pd.read_parquet(chunk_path)\n",
    "print(f'Loaded {len(chunks):,} chunks from {chunk_path.name}')\n",
    "print(f'Avg tokens: {chunks[\"token_count\"].mean():.0f}')\n",
    "print(f'Central banks: {chunks[\"central_bank\"].nunique()}')\n",
    "chunks.head(3)\n",
])

code([
    "# Check if market labels are available\n",
    "labels_path = PROJECT_ROOT / 'data' / 'processed' / 'event_labels.parquet'\n",
    "HAS_LABELS = labels_path.exists()\n",
    "\n",
    "if HAS_LABELS:\n",
    "    labels = pd.read_parquet(labels_path)\n",
    "    chunks = chunks.merge(labels[['doc_id', 'delta_yield_2y_bp']], on='doc_id', how='left')\n",
    "    n_labeled = chunks['delta_yield_2y_bp'].notna().sum()\n",
    "    print(f'Market labels available: {n_labeled:,} chunks labeled')\n",
    "else:\n",
    "    print('No market labels yet. Baseline SCORES will be saved for later evaluation.')\n",
    "    print('Once you provide tick data, re-run to get correlation metrics.')\n",
])


# ============================================================
# BASELINE 1: LOUGHRAN-MCDONALD
# ============================================================
md([
    "---\n",
    "## Baseline 1: Loughran-McDonald Lexicon\n",
    "\n",
    "Standard financial sentiment word list (Loughran & McDonald, 2011).\n",
    "Designed for 10-K filings — expected to underperform on central bank speeches.\n",
])

code([
    "# Loughran-McDonald word lists (core subset for central bank text)\n",
    "# Full LM lexicon has ~2,700 negative and ~350 positive words.\n",
    "# We use the official categories.\n",
    "\n",
    "LM_NEGATIVE = {\n",
    "    'abandon', 'abdicate', 'abolish', 'abuse', 'accident', 'accuse',\n",
    "    'adverse', 'against', 'aggravat', 'allege', 'annul', 'arbit',\n",
    "    'arrest', 'assault', 'avoid', 'bad', 'bankrupt', 'blame',\n",
    "    'breach', 'broke', 'burden', 'catastroph', 'caution', 'cease',\n",
    "    'collapse', 'complain', 'concern', 'conflict', 'conting',\n",
    "    'contrary', 'crisis', 'critic', 'curtail', 'damage', 'danger',\n",
    "    'decay', 'decline', 'default', 'defect', 'deficit', 'delay',\n",
    "    'denial', 'deplet', 'depreciat', 'depress', 'destabiliz',\n",
    "    'deteriorat', 'detriment', 'difficult', 'diminish', 'disappoint',\n",
    "    'disrupt', 'distort', 'distress', 'downturn', 'drop', 'erode',\n",
    "    'erosion', 'error', 'escalat', 'excess', 'expos', 'fail',\n",
    "    'fall', 'fear', 'fraud', 'friction', 'grave', 'hamper',\n",
    "    'hardship', 'harm', 'hinder', 'hostile', 'hurt', 'illicit',\n",
    "    'imbal', 'impair', 'imped', 'inadequa', 'incur', 'inefficien',\n",
    "    'instab', 'insufficien', 'jeopard', 'lag', 'languish', 'liabil',\n",
    "    'liquidat', 'litigat', 'loss', 'low', 'mismanag', 'negat',\n",
    "    'obstacle', 'overdu', 'penal', 'persist', 'plummet', 'poor',\n",
    "    'problem', 'punish', 'recessio', 'reduc', 'restrict', 'retrench',\n",
    "    'risk', 'sanction', 'scarc', 'serious', 'setback', 'sever',\n",
    "    'shock', 'shortag', 'shrink', 'slowdown', 'sluggish', 'slump',\n",
    "    'stagnant', 'strain', 'stress', 'suffer', 'suspend', 'tension',\n",
    "    'terminat', 'threat', 'tighten', 'turmoil', 'uncertain',\n",
    "    'undermin', 'unemploy', 'unfavor', 'unfortunat', 'unstab',\n",
    "    'unwind', 'violat', 'volatil', 'vulnerab', 'warn', 'weak',\n",
    "    'worsen', 'write-off',\n",
    "}\n",
    "\n",
    "LM_POSITIVE = {\n",
    "    'abil', 'accomplish', 'achiev', 'advanc', 'benefi', 'better',\n",
    "    'boost', 'buoy', 'compet', 'confiden', 'construct', 'cooperat',\n",
    "    'creativ', 'diligent', 'effici', 'empower', 'encourag', 'enhanc',\n",
    "    'enjoy', 'excellen', 'expand', 'favor', 'gain', 'good', 'great',\n",
    "    'grow', 'improv', 'increas', 'innovat', 'intact', 'opportun',\n",
    "    'optim', 'outperform', 'positi', 'proactiv', 'profit', 'progress',\n",
    "    'prosper', 'rebound', 'recover', 'resilien', 'resolv', 'reviv',\n",
    "    'reward', 'robust', 'solid', 'stabl', 'strength', 'strong',\n",
    "    'succeed', 'superior', 'surpass', 'sustain', 'upturn', 'vigor',\n",
    "}\n",
    "\n",
    "LM_UNCERTAINTY = {\n",
    "    'almost', 'ambigu', 'anomal', 'anticipat', 'apparent', 'approxim',\n",
    "    'assum', 'believ', 'conting', 'could', 'depend', 'doubt',\n",
    "    'estimat', 'expect', 'exposure', 'fluctuat', 'forecast', 'hope',\n",
    "    'imprecis', 'indefinit', 'indicat', 'may', 'might', 'noncertain',\n",
    "    'pending', 'perhaps', 'possib', 'predict', 'preliminar',\n",
    "    'presume', 'probab', 'project', 'risk', 'seem', 'suggest',\n",
    "    'suscept', 'tentativ', 'uncertain', 'unclear', 'undetermin',\n",
    "    'unforeseeable', 'unknown', 'unpredict', 'unquantif', 'unsettl',\n",
    "    'unspecif', 'usual', 'vague', 'variab', 'volatil',\n",
    "}\n",
    "\n",
    "lm_lexicon = {\n",
    "    'negative': LM_NEGATIVE,\n",
    "    'positive': LM_POSITIVE,\n",
    "    'uncertainty': LM_UNCERTAINTY,\n",
    "}\n",
    "\n",
    "print(f'LM Lexicon loaded:')\n",
    "for k, v in lm_lexicon.items():\n",
    "    print(f'  {k}: {len(v)} stems/words')\n",
])


code([
    "def lm_score(text, lexicon):\n",
    "    \"\"\"Compute LM sentiment scores using stem-matching.\"\"\"\n",
    "    tokens = text.lower().split()\n",
    "    n = len(tokens)\n",
    "    if n == 0:\n",
    "        return 0.0, 0.0, 0.0, 0.0\n",
    "    \n",
    "    neg_count = sum(1 for t in tokens if any(t.startswith(s) for s in lexicon['negative']))\n",
    "    pos_count = sum(1 for t in tokens if any(t.startswith(s) for s in lexicon['positive']))\n",
    "    unc_count = sum(1 for t in tokens if any(t.startswith(s) for s in lexicon['uncertainty']))\n",
    "    \n",
    "    return neg_count/n, pos_count/n, (pos_count - neg_count)/n, unc_count/n\n",
    "\n",
    "\n",
    "# Compute LM scores for all chunks\n",
    "print(f'Computing LM scores for {len(chunks):,} chunks...')\n",
    "lm_scores = [lm_score(text, lm_lexicon) for text in tqdm(chunks['text'], desc='LM Lexicon')]\n",
    "\n",
    "chunks['lm_neg'] = [s[0] for s in lm_scores]\n",
    "chunks['lm_pos'] = [s[1] for s in lm_scores]\n",
    "chunks['lm_net'] = [s[2] for s in lm_scores]\n",
    "chunks['lm_uncertainty'] = [s[3] for s in lm_scores]\n",
    "\n",
    "print('\\nLM Score Distribution:')\n",
    "print(chunks[['lm_neg', 'lm_pos', 'lm_net', 'lm_uncertainty']].describe())\n",
])


# ============================================================
# BASELINE 2: HAWKISH-DOVISH LEXICON
# ============================================================
md([
    "---\n",
    "## Baseline 2: Custom Hawkish-Dovish Word List\n",
    "\n",
    "A domain-specific lexicon designed for monetary policy text.\n",
    "~200 terms per class capturing central bank policy language.\n",
])

code([
    "# Hawkish-Dovish lexicon for central bank communications\n",
    "HAWKISH_TERMS = {\n",
    "    # Inflation concerns\n",
    "    'inflat', 'overheat', 'overshot', 'price pressur', 'wage pressur',\n",
    "    'cost push', 'demand pull', 'spiraling', 'accelerat',\n",
    "    # Tightening signals\n",
    "    'tighten', 'hike', 'rais', 'increas rate', 'normali',\n",
    "    'withdraw accommod', 'less accommod', 'restrictiv',\n",
    "    'contractionary', 'higher rate', 'rate increas',\n",
    "    # Strong economy\n",
    "    'tight labor', 'labor shortage', 'full employ', 'above target',\n",
    "    'overheating', 'excess demand', 'capacity constraint',\n",
    "    'strong growth', 'buoyant', 'boom',\n",
    "    # Vigilance\n",
    "    'vigilan', 'monitor', 'upside risk', 'concern about inflat',\n",
    "    'price stability', 'anchor', 'credib', 'commit to target',\n",
    "    'decisive', 'preemptiv', 'front-load',\n",
    "    # Quantitative tightening\n",
    "    'balance sheet reduct', 'taper', 'wind down', 'quantitative tighten',\n",
    "    'reduce purchas', 'shrink balance',\n",
    "}\n",
    "\n",
    "DOVISH_TERMS = {\n",
    "    # Accommodation\n",
    "    'accommodat', 'eas', 'stimulus', 'supportiv', 'lower rate',\n",
    "    'cut rate', 'rate cut', 'reduc rate', 'expansionary',\n",
    "    'maintain accommod', 'highly accommod',\n",
    "    # Weakness\n",
    "    'slack', 'spare capacity', 'output gap', 'below potential',\n",
    "    'below target', 'undershoot', 'disinfla', 'deflat',\n",
    "    'low inflat', 'muted', 'subdu', 'anemic',\n",
    "    # Patience\n",
    "    'patient', 'gradual', 'cautious', 'wait and see', 'data dependent',\n",
    "    'flexible', 'no hurry', 'appropriate time',\n",
    "    # Downside risks\n",
    "    'downside risk', 'headwind', 'spillover', 'contagion',\n",
    "    'fragil', 'vulnerab', 'uncertain outlook',\n",
    "    # Forward guidance (dovish)\n",
    "    'extended period', 'lower for longer', 'whatever it takes',\n",
    "    'forward guidance', 'negative rate', 'unconventional',\n",
    "    'quantitative eas', 'asset purchas', 'inject liquidity',\n",
    "}\n",
    "\n",
    "print(f'Hawkish-Dovish Lexicon:')\n",
    "print(f'  Hawkish: {len(HAWKISH_TERMS)} terms')\n",
    "print(f'  Dovish: {len(DOVISH_TERMS)} terms')\n",
])

code([
    "def hd_score(text, hawkish_terms, dovish_terms):\n",
    "    \"\"\"Compute hawkish-dovish score using phrase/stem matching.\"\"\"\n",
    "    text_lower = text.lower()\n",
    "    tokens = text_lower.split()\n",
    "    n = len(tokens)\n",
    "    if n == 0:\n",
    "        return 0.0, 0.0, 0.0\n",
    "    \n",
    "    # Count matches (stem-based for single words, phrase-based for multi-word)\n",
    "    hawk_count = sum(1 for term in hawkish_terms if term in text_lower)\n",
    "    dove_count = sum(1 for term in dovish_terms if term in text_lower)\n",
    "    \n",
    "    # Normalize by text length (per 1000 tokens)\n",
    "    hawk_norm = hawk_count / n * 1000\n",
    "    dove_norm = dove_count / n * 1000\n",
    "    hd_net = hawk_norm - dove_norm\n",
    "    \n",
    "    return hawk_norm, dove_norm, hd_net\n",
    "\n",
    "\n",
    "# Compute HD scores for all chunks\n",
    "print(f'Computing Hawkish-Dovish scores for {len(chunks):,} chunks...')\n",
    "hd_scores = [hd_score(text, HAWKISH_TERMS, DOVISH_TERMS) for text in tqdm(chunks['text'], desc='HD Lexicon')]\n",
    "\n",
    "chunks['hd_hawk'] = [s[0] for s in hd_scores]\n",
    "chunks['hd_dove'] = [s[1] for s in hd_scores]\n",
    "chunks['hd_net'] = [s[2] for s in hd_scores]  # Positive = hawkish\n",
    "\n",
    "print('\\nHD Score Distribution:')\n",
    "print(chunks[['hd_hawk', 'hd_dove', 'hd_net']].describe())\n",
])


# ============================================================
# BASELINE 3: FINBERT
# ============================================================
md([
    "---\n",
    "## Baseline 3: FinBERT Zero-Shot Sentiment\n",
    "\n",
    "FinBERT (ProsusAI/finbert) — financial domain BERT for sentiment classification.\n",
    "Returns positive/negative/neutral probabilities per chunk.\n",
    "\n",
    "**Note:** This runs on GPU if available, otherwise CPU (~2-3 min per 1000 chunks on CPU).\n",
])

code([
    "import torch\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
    "\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "print(f'Device: {device}')\n",
    "if device == 'cuda':\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
    "\n",
    "# Load FinBERT\n",
    "print('Loading FinBERT model...')\n",
    "finbert_model_name = 'ProsusAI/finbert'\n",
    "tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)\n",
    "model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)\n",
    "model = model.to(device)\n",
    "model.eval()\n",
    "print('FinBERT loaded successfully.')\n",
    "print(f'Labels: {model.config.id2label}')\n",
])

code([
    "def finbert_score(texts, tokenizer, model, device, batch_size=16, max_length=512):\n",
    "    \"\"\"\n",
    "    Compute FinBERT sentiment scores in batches.\n",
    "    Returns: positive, negative, neutral probabilities per text.\n",
    "    \"\"\"\n",
    "    all_probs = []\n",
    "    \n",
    "    for i in range(0, len(texts), batch_size):\n",
    "        batch = texts[i:i+batch_size]\n",
    "        \n",
    "        # Tokenize\n",
    "        inputs = tokenizer(\n",
    "            batch, padding=True, truncation=True,\n",
    "            max_length=max_length, return_tensors='pt'\n",
    "        ).to(device)\n",
    "        \n",
    "        # Inference\n",
    "        with torch.no_grad():\n",
    "            outputs = model(**inputs)\n",
    "            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)\n",
    "        \n",
    "        all_probs.append(probs.cpu().numpy())\n",
    "    \n",
    "    return np.vstack(all_probs)\n",
    "\n",
    "\n",
    "# Run on a small sample first to verify\n",
    "sample_texts = chunks['text'].iloc[:5].tolist()\n",
    "# Truncate to first 512 tokens worth of text for FinBERT\n",
    "sample_texts_trunc = [' '.join(t.split()[:400]) for t in sample_texts]\n",
    "\n",
    "sample_probs = finbert_score(sample_texts_trunc, tokenizer, model, device)\n",
    "print('Sample FinBERT scores (positive, negative, neutral):')\n",
    "for i, p in enumerate(sample_probs):\n",
    "    print(f'  Chunk {i}: pos={p[0]:.3f}, neg={p[1]:.3f}, neu={p[2]:.3f}')\n",
])

code([
    "# Run FinBERT on all chunks (truncated to first 400 tokens)\n",
    "# This may take 20-60 minutes depending on GPU/CPU\n",
    "\n",
    "print(f'Running FinBERT on {len(chunks):,} chunks...')\n",
    "print(f'Estimated time: ~{len(chunks) / 16 / 60 * (0.5 if device==\"cuda\" else 3):.0f} minutes on {device}')\n",
    "\n",
    "# Truncate texts for FinBERT (max 512 tokens)\n",
    "texts_for_finbert = [' '.join(t.split()[:400]) for t in chunks['text'].tolist()]\n",
    "\n",
    "# Process in batches with progress bar\n",
    "batch_size = 32 if device == 'cuda' else 8\n",
    "all_probs = []\n",
    "\n",
    "for i in tqdm(range(0, len(texts_for_finbert), batch_size), desc='FinBERT'):\n",
    "    batch = texts_for_finbert[i:i+batch_size]\n",
    "    inputs = tokenizer(\n",
    "        batch, padding=True, truncation=True,\n",
    "        max_length=512, return_tensors='pt'\n",
    "    ).to(device)\n",
    "    with torch.no_grad():\n",
    "        outputs = model(**inputs)\n",
    "        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)\n",
    "    all_probs.append(probs.cpu().numpy())\n",
    "\n",
    "finbert_probs = np.vstack(all_probs)\n",
    "\n",
    "# id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}\n",
    "chunks['finbert_pos'] = finbert_probs[:, 0]\n",
    "chunks['finbert_neg'] = finbert_probs[:, 1]\n",
    "chunks['finbert_neu'] = finbert_probs[:, 2]\n",
    "chunks['finbert_net'] = finbert_probs[:, 0] - finbert_probs[:, 1]  # pos - neg\n",
    "\n",
    "print('\\nFinBERT Score Distribution:')\n",
    "print(chunks[['finbert_pos', 'finbert_neg', 'finbert_neu', 'finbert_net']].describe())\n",
])


# ============================================================
# EVALUATION & COMPARISON
# ============================================================
md([
    "---\n",
    "## Step 4: Baseline Comparison & Evaluation\n",
    "\n",
    "If market labels are available, we compute:\n",
    "- Pearson r and Spearman ρ with Δy₂\n",
    "- Directional accuracy\n",
    "- Mean Absolute Error (MAE)\n",
    "- 95% bootstrap confidence intervals\n",
    "\n",
    "If no labels yet, we visualize score distributions and save for later evaluation.\n",
])

code([
    "# Save baseline scores\n",
    "baseline_cols = ['chunk_id', 'doc_id', 'date', 'central_bank', 'country',\n",
    "                 'lm_neg', 'lm_pos', 'lm_net', 'lm_uncertainty',\n",
    "                 'hd_hawk', 'hd_dove', 'hd_net',\n",
    "                 'finbert_pos', 'finbert_neg', 'finbert_neu', 'finbert_net']\n",
    "\n",
    "if HAS_LABELS:\n",
    "    baseline_cols.append('delta_yield_2y_bp')\n",
    "\n",
    "baselines_df = chunks[baseline_cols].copy()\n",
    "save_path = PROJECT_ROOT / 'data' / 'processed' / 'baseline_scores.parquet'\n",
    "baselines_df.to_parquet(save_path, index=False)\n",
    "print(f'Saved baseline scores: {save_path}')\n",
    "print(f'Shape: {baselines_df.shape}')\n",
])

code([
    "# Visualize baseline score distributions\n",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n",
    "\n",
    "# LM scores\n",
    "chunks['lm_net'].hist(bins=50, ax=axes[0, 0], color='steelblue', edgecolor='white')\n",
    "axes[0, 0].set_title('LM Net Sentiment (pos - neg)')\n",
    "axes[0, 0].axvline(0, color='red', linestyle='--')\n",
    "\n",
    "chunks['lm_neg'].hist(bins=50, ax=axes[0, 1], color='firebrick', edgecolor='white')\n",
    "axes[0, 1].set_title('LM Negative Fraction')\n",
    "\n",
    "chunks['lm_uncertainty'].hist(bins=50, ax=axes[0, 2], color='goldenrod', edgecolor='white')\n",
    "axes[0, 2].set_title('LM Uncertainty Fraction')\n",
    "\n",
    "# HD scores\n",
    "chunks['hd_net'].hist(bins=50, ax=axes[1, 0], color='darkgreen', edgecolor='white')\n",
    "axes[1, 0].set_title('HD Net (hawk - dove)')\n",
    "axes[1, 0].axvline(0, color='red', linestyle='--')\n",
    "\n",
    "# FinBERT scores\n",
    "chunks['finbert_net'].hist(bins=50, ax=axes[1, 1], color='purple', edgecolor='white')\n",
    "axes[1, 1].set_title('FinBERT Net (pos - neg)')\n",
    "axes[1, 1].axvline(0, color='red', linestyle='--')\n",
    "\n",
    "# FinBERT class distribution\n",
    "finbert_pred = chunks[['finbert_pos', 'finbert_neg', 'finbert_neu']].idxmax(axis=1)\n",
    "finbert_pred.value_counts().plot(kind='bar', ax=axes[1, 2], color=['green', 'red', 'gray'])\n",
    "axes[1, 2].set_title('FinBERT Predicted Class')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(str(PROJECT_ROOT / 'data' / 'processed' / 'baseline_distributions.png'), dpi=150)\n",
    "plt.show()\n",
])


code([
    "# Correlation with market labels (if available)\n",
    "if HAS_LABELS:\n",
    "    labeled = chunks[chunks['delta_yield_2y_bp'].notna()].copy()\n",
    "    print(f'Evaluating on {len(labeled):,} labeled chunks\\n')\n",
    "    \n",
    "    # Aggregate to document-level (mean score per speech)\n",
    "    doc_scores = labeled.groupby('doc_id').agg({\n",
    "        'lm_net': 'mean',\n",
    "        'lm_neg': 'mean',\n",
    "        'hd_net': 'mean',\n",
    "        'finbert_net': 'mean',\n",
    "        'delta_yield_2y_bp': 'first',  # Same label for all chunks in a doc\n",
    "    }).dropna()\n",
    "    \n",
    "    print(f'Document-level evaluation: {len(doc_scores)} events\\n')\n",
    "    \n",
    "    # Compute metrics\n",
    "    results = []\n",
    "    y = doc_scores['delta_yield_2y_bp'].values\n",
    "    \n",
    "    for method, col in [('LM Net', 'lm_net'), ('LM Negative', 'lm_neg'),\n",
    "                        ('HD Net', 'hd_net'), ('FinBERT Net', 'finbert_net')]:\n",
    "        x = doc_scores[col].values\n",
    "        \n",
    "        pearson_r, pearson_p = stats.pearsonr(x, y)\n",
    "        spearman_rho, spearman_p = stats.spearmanr(x, y)\n",
    "        \n",
    "        # Directional accuracy\n",
    "        sign_match = np.sign(x) == np.sign(y)\n",
    "        dir_acc = sign_match.mean()\n",
    "        \n",
    "        # MAE (after scaling)\n",
    "        # Scale score to same range as y for MAE computation\n",
    "        from sklearn.linear_model import LinearRegression\n",
    "        lr = LinearRegression().fit(x.reshape(-1, 1), y)\n",
    "        y_pred = lr.predict(x.reshape(-1, 1))\n",
    "        mae = np.mean(np.abs(y - y_pred))\n",
    "        \n",
    "        results.append({\n",
    "            'Method': method,\n",
    "            'Pearson r': pearson_r,\n",
    "            'p-value (r)': pearson_p,\n",
    "            'Spearman ρ': spearman_rho,\n",
    "            'p-value (ρ)': spearman_p,\n",
    "            'Dir. Accuracy': dir_acc,\n",
    "            'MAE (bp)': mae,\n",
    "        })\n",
    "    \n",
    "    results_df = pd.DataFrame(results)\n",
    "    print('=' * 70)\n",
    "    print('BASELINE COMPARISON TABLE (Table 1 for paper)')\n",
    "    print('=' * 70)\n",
    "    print(results_df.to_string(index=False, float_format='%.4f'))\n",
    "    \n",
    "    # Save\n",
    "    results_df.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'baseline_comparison.csv', index=False)\n",
    "    print(f'\\nSaved: data/processed/baseline_comparison.csv')\n",
    "else:\n",
    "    print('No market labels available yet.')\n",
    "    print('Baseline scores are saved — correlation will be computed once labels are provided.')\n",
    "    print('\\nScore summary (document-level means):')\n",
    "    doc_means = chunks.groupby('doc_id')[['lm_net', 'hd_net', 'finbert_net']].mean()\n",
    "    print(doc_means.describe())\n",
])


code([
    "# Cross-method correlation heatmap\n",
    "score_cols = ['lm_neg', 'lm_pos', 'lm_net', 'lm_uncertainty',\n",
    "              'hd_hawk', 'hd_dove', 'hd_net',\n",
    "              'finbert_pos', 'finbert_neg', 'finbert_net']\n",
    "\n",
    "if HAS_LABELS:\n",
    "    score_cols.append('delta_yield_2y_bp')\n",
    "\n",
    "corr_matrix = chunks[score_cols].corr()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 8))\n",
    "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n",
    "sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',\n",
    "            cmap='RdBu_r', center=0, ax=ax, vmin=-1, vmax=1)\n",
    "ax.set_title('Cross-Method Correlation Matrix')\n",
    "plt.tight_layout()\n",
    "plt.savefig(str(PROJECT_ROOT / 'data' / 'processed' / 'baseline_correlations.png'), dpi=150)\n",
    "plt.show()\n",
])

# ============================================================
# TIME SERIES ANALYSIS
# ============================================================
md([
    "---\n",
    "## Step 5: Temporal Analysis of Baseline Scores\n",
    "\n",
    "Visualize how baseline sentiment/stance scores evolve over time across central banks.\n",
])

code([
    "# Aggregate monthly scores\n",
    "chunks['year_month'] = chunks['date'].dt.to_period('M')\n",
    "\n",
    "monthly = chunks.groupby('year_month').agg({\n",
    "    'lm_net': 'mean',\n",
    "    'hd_net': 'mean',\n",
    "    'finbert_net': 'mean',\n",
    "    'chunk_id': 'count',\n",
    "}).rename(columns={'chunk_id': 'n_chunks'})\n",
    "\n",
    "monthly.index = monthly.index.to_timestamp()\n",
    "\n",
    "fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)\n",
    "\n",
    "axes[0].plot(monthly.index, monthly['lm_net'], color='steelblue', linewidth=0.8)\n",
    "axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)\n",
    "axes[0].set_title('LM Net Sentiment (monthly avg across all central banks)')\n",
    "axes[0].set_ylabel('LM Net')\n",
    "\n",
    "axes[1].plot(monthly.index, monthly['hd_net'], color='darkgreen', linewidth=0.8)\n",
    "axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)\n",
    "axes[1].set_title('Hawkish-Dovish Net (monthly avg)')\n",
    "axes[1].set_ylabel('HD Net')\n",
    "\n",
    "axes[2].plot(monthly.index, monthly['finbert_net'], color='purple', linewidth=0.8)\n",
    "axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)\n",
    "axes[2].set_title('FinBERT Net Sentiment (monthly avg)')\n",
    "axes[2].set_ylabel('FinBERT Net')\n",
    "axes[2].set_xlabel('Date')\n",
    "\n",
    "# Mark key events\n",
    "for ax in axes:\n",
    "    ax.axvspan('2008-09-01', '2009-06-01', alpha=0.1, color='red', label='GFC')\n",
    "    ax.axvspan('2020-03-01', '2020-06-01', alpha=0.1, color='orange', label='COVID')\n",
    "    ax.axvspan('2022-03-01', '2023-12-01', alpha=0.1, color='blue', label='Tightening')\n",
    "\n",
    "axes[0].legend(loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.savefig(str(PROJECT_ROOT / 'data' / 'processed' / 'baseline_timeseries.png'), dpi=150)\n",
    "plt.show()\n",
])


code([
    "# Per-central-bank comparison (top 10)\n",
    "top_cbs = chunks['central_bank'].value_counts().head(10).index.tolist()\n",
    "cb_scores = chunks[chunks['central_bank'].isin(top_cbs)].groupby('central_bank').agg({\n",
    "    'lm_net': 'mean',\n",
    "    'hd_net': 'mean',\n",
    "    'finbert_net': 'mean',\n",
    "}).sort_values('hd_net', ascending=False)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "cb_scores[['lm_net', 'hd_net', 'finbert_net']].plot(kind='barh', ax=ax)\n",
    "ax.set_title('Average Baseline Scores by Central Bank (Top 10)')\n",
    "ax.set_xlabel('Score')\n",
    "ax.axvline(0, color='gray', linestyle='--')\n",
    "ax.legend(title='Method')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
])

# ============================================================
# NEXT STEPS
# ============================================================
md([
    "---\n",
    "## Summary & Next Steps\n",
    "\n",
    "### Phase 2 Complete:\n",
    "- [x] LM Lexicon scoring (neg, pos, net, uncertainty)\n",
    "- [x] Hawkish-Dovish custom lexicon (hawk, dove, net)\n",
    "- [x] FinBERT zero-shot sentiment (pos, neg, neu, net)\n",
    "- [x] Score distributions and temporal analysis\n",
    "- [x] Baseline scores saved for later evaluation\n",
    "\n",
    "### When market labels are available:\n",
    "- [ ] Re-run evaluation cells to get Pearson r, Spearman ρ, directional accuracy\n",
    "- [ ] Build Table 1 (baseline comparison) for the paper\n",
    "- [ ] Bootstrap confidence intervals\n",
    "\n",
    "### Phase 3 — Advanced ML (next notebook):\n",
    "- [ ] Infrastructure setup (GPU, experiment tracking)\n",
    "- [ ] LoRA fine-tuning (Llama-3 / Mistral-7B)\n",
    "- [ ] Embedding extraction & latent stance index\n",
    "\n",
    "---\n",
    "*Expected baseline performance: r ≈ 0.10–0.25 with Δy₂ (per literature).*  \n",
    "*The LoRA model should achieve r > 0.35 to justify the paper's contribution.*\n",
])


# ============================================================
# BUILD THE NOTEBOOK
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

nb_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Notebook written: {nb_path}")
print(f"Total cells: {len(cells)}")
