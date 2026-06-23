"""
Phase 3 — Advanced ML: LoRA Fine-Tuning (Colab A100/T4)
=========================================================
Copy each section into separate Colab cells.
Upload your data files to Colab first:
  - data/processed/labeled_chunk_registry.parquet
  - data/processed/labeled_document_registry.parquet

Architecture: Mistral-7B-Instruct-v0.3 + LoRA + Regression Head
Objective: Predict Δy₂ (basis-point yield change) from text chunks
"""

# ============================================================================
# CELL 1: Install & Imports
# ============================================================================
"""
!pip install -q transformers peft accelerate datasets bitsandbytes
!pip install -q scipy scikit-learn umap-learn mlflow
!pip install -q pyarrow fastparquet
"""

# ============================================================================
# CELL 2: Imports & Config
# ============================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from scipy import stats
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Config
SEED = 42
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8  # effective batch = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LABEL_COL = "delta_US_2Y_bp"

# Temporal split (strict — no leakage)
TRAIN_END = "2018-12-31"
VAL_START = "2019-01-01"
VAL_END = "2020-12-31"
TEST_START = "2021-01-01"

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")
print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")


# ============================================================================
# CELL 3: Load & Prepare Data
# ============================================================================

# Upload your parquet files to Colab, or mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Load labeled chunks
chunks = pd.read_parquet("labeled_chunk_registry.parquet")
docs = pd.read_parquet("labeled_document_registry.parquet")

# Filter to Fed-only (US 2Y yield responds to Fed)
fed_docs = docs[docs["central_bank"] == "Board of Governors of the Federal Reserve"]
fed_doc_ids = set(fed_docs["doc_id"].values)

# Filter chunks
fed_chunks = chunks[chunks["doc_id"].isin(fed_doc_ids)].copy()
fed_chunks = fed_chunks[fed_chunks[LABEL_COL].notna()].copy()
fed_chunks["date"] = pd.to_datetime(fed_chunks["date"])

print(f"Fed chunks with labels: {len(fed_chunks)}")
print(f"Label stats: mean={fed_chunks[LABEL_COL].mean():.2f}, std={fed_chunks[LABEL_COL].std():.2f}")

# Temporal split
train = fed_chunks[fed_chunks["date"] <= TRAIN_END]
val = fed_chunks[(fed_chunks["date"] >= VAL_START) & (fed_chunks["date"] <= VAL_END)]
test = fed_chunks[fed_chunks["date"] >= TEST_START]

print(f"\nTemporal Split:")
print(f"  Train: {len(train)} chunks (up to {TRAIN_END})")
print(f"  Val:   {len(val)} chunks ({VAL_START} to {VAL_END})")
print(f"  Test:  {len(test)} chunks ({TEST_START} onward)")


# ============================================================================
# CELL 4: Dataset & DataLoader
# ============================================================================

class PolicyStanceDataset(Dataset):
    """Dataset for text → yield change regression."""
    
    def __init__(self, df, tokenizer, max_length=512, label_col="delta_US_2Y_bp"):
        self.texts = df["text"].tolist()
        self.labels = df[label_col].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])[:5000]  # truncate very long texts
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


# ============================================================================
# CELL 5: Load Model with LoRA
# ============================================================================

# Quantization config (4-bit for memory efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("Loading base model (this takes 2-3 minutes)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
base_model.config.use_cache = False

# LoRA config
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


# ============================================================================
# CELL 6: Regression Head (appended to LLM)
# ============================================================================

class StanceRegressionModel(nn.Module):
    """LLM + LoRA + Linear Regression Head for predicting Δy₂."""
    
    def __init__(self, base_model, hidden_size=4096):
        super().__init__()
        self.base_model = base_model
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get hidden states from the LLM
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Use last hidden state, mean-pool over non-padding tokens
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        
        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Regression prediction
        prediction = self.regression_head(pooled).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(prediction, labels)
        
        return {"loss": loss, "predictions": prediction, "embeddings": pooled}


# Initialize
hidden_size = base_model.config.hidden_size
stance_model = StanceRegressionModel(model, hidden_size=hidden_size)
print(f"Hidden size: {hidden_size}")
print(f"Regression head parameters: {sum(p.numel() for p in stance_model.regression_head.parameters()):,}")


# ============================================================================
# CELL 7: Training Loop
# ============================================================================

# Create datasets
train_dataset = PolicyStanceDataset(train, tokenizer, MAX_LENGTH, LABEL_COL)
val_dataset = PolicyStanceDataset(val, tokenizer, MAX_LENGTH, LABEL_COL)
test_dataset = PolicyStanceDataset(test, tokenizer, MAX_LENGTH, LABEL_COL)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Optimizer (only LoRA + regression head parameters)
trainable_params = [p for p in stance_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

# Learning rate scheduler
total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION
warmup_steps = total_steps // 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Total steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")


# ============================================================================
# CELL 8: Training Execution
# ============================================================================

def evaluate(model, dataloader, device):
    """Evaluate model on a dataloader."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs["loss"].item()
            all_preds.extend(outputs["predictions"].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    r = stats.pearsonr(preds, labels)[0] if len(preds) > 2 else 0
    mae = mean_absolute_error(labels, preds)
    dir_acc = np.mean(np.sign(preds) == np.sign(labels))
    
    return {
        "loss": total_loss / len(dataloader),
        "pearson_r": r,
        "mae_bp": mae,
        "dir_accuracy": dir_acc,
    }


# Training loop
best_val_r = -1
train_losses = []
val_metrics_history = []

print("Starting training...")
print(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<10}{'Val r':<10}{'Val MAE':<10}{'Val Dir.Acc':<12}")
print("-" * 62)

for epoch in range(NUM_EPOCHS):
    stance_model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = stance_model(input_ids, attention_mask, labels)
        loss = outputs["loss"] / GRADIENT_ACCUMULATION
        loss.backward()
        epoch_loss += outputs["loss"].item()
        
        if (step + 1) % GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    # Evaluate
    val_metrics = evaluate(stance_model, val_loader, device)
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    val_metrics_history.append(val_metrics)
    
    print(f"{epoch+1:<8}{avg_train_loss:<12.4f}{val_metrics['loss']:<10.4f}"
          f"{val_metrics['pearson_r']:<10.4f}{val_metrics['mae_bp']:<10.2f}"
          f"{val_metrics['dir_accuracy']:<12.3f}")
    
    # Save best checkpoint
    if val_metrics["pearson_r"] > best_val_r:
        best_val_r = val_metrics["pearson_r"]
        torch.save(stance_model.state_dict(), "best_stance_model.pt")
        print(f"  → New best model (r = {best_val_r:.4f})")
    
    # Early stopping (patience = 3)
    if epoch >= 3:
        recent = [m["pearson_r"] for m in val_metrics_history[-3:]]
        if all(r <= best_val_r - 0.01 for r in recent):
            print("Early stopping triggered.")
            break

print(f"\nBest validation Pearson r: {best_val_r:.4f}")


# ============================================================================
# CELL 9: Final Test Evaluation (run ONCE only)
# ============================================================================

# Load best model
stance_model.load_state_dict(torch.load("best_stance_model.pt"))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_metrics = evaluate(stance_model, test_loader, device)
print("=" * 50)
print("FINAL TEST SET EVALUATION (2021-present)")
print("=" * 50)
print(f"  Pearson r:  {test_metrics['pearson_r']:.4f}")
print(f"  MAE (bp):   {test_metrics['mae_bp']:.2f}")
print(f"  Dir. Acc:   {test_metrics['dir_accuracy']:.3f}")
print(f"  Loss (MSE): {test_metrics['loss']:.4f}")
print("=" * 50)


# ============================================================================
# CELL 10: Embedding Extraction (M6)
# ============================================================================

def extract_embeddings(model, dataloader, device):
    """Extract embeddings from the fine-tuned model's final layer."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(input_ids, attention_mask)
            embeddings = outputs["embeddings"].cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_labels.extend(labels.numpy())
    
    return np.vstack(all_embeddings), np.array(all_labels)


print("Extracting embeddings from all splits...")
# Use all data for embedding extraction
all_dataset = PolicyStanceDataset(fed_chunks, tokenizer, MAX_LENGTH, LABEL_COL)
all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=False)

embeddings, labels = extract_embeddings(stance_model, all_loader, device)
print(f"Embedding matrix: {embeddings.shape}")  # (N, 4096)
print(f"Labels: {labels.shape}")

# Save embeddings
np.save("stance_embeddings.npy", embeddings)
np.save("stance_labels.npy", labels)
print("Saved: stance_embeddings.npy, stance_labels.npy")


# ============================================================================
# CELL 11: PCA & Latent Stance Space (M6)
# ============================================================================

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# PCA fitted on TRAINING split only (no leakage!)
train_mask = fed_chunks["date"] <= TRAIN_END
train_emb = embeddings[train_mask.values]
train_lbl = labels[train_mask.values]

# Standardize
scaler = StandardScaler()
train_emb_scaled = scaler.fit_transform(train_emb)

# PCA
pca = PCA(n_components=50)
pca.fit(train_emb_scaled)

# Transform all embeddings using training-fitted PCA
all_emb_scaled = scaler.transform(embeddings)
all_pca = pca.transform(all_emb_scaled)

# PC1 = latent stance score
stance_scores = all_pca[:, 0]

# Check: does PC1 correlate with Δy₂?
r_pc1, p_pc1 = stats.pearsonr(stance_scores[~np.isnan(labels)], 
                                labels[~np.isnan(labels)])
print(f"PC1 explained variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"PC1 correlation with Δy₂: r = {r_pc1:.4f}, p = {p_pc1:.4e}")
print(f"Top 5 PCs explain: {pca.explained_variance_ratio_[:5].sum()*100:.1f}%")

# If PC1 is negatively correlated, flip sign for interpretability
if r_pc1 < 0:
    stance_scores = -stance_scores
    r_pc1 = -r_pc1
    print("(Flipped PC1 sign for hawkish=positive convention)")

print(f"\nLatent stance score: r = {r_pc1:.4f} with Δy₂")
print("→ Positive scores = hawkish, Negative = dovish")


# ============================================================================
# CELL 12: UMAP Visualization
# ============================================================================

import umap

# UMAP on the PCA-reduced embeddings (50-dim → 2-dim)
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, 
                     metric="cosine", random_state=SEED)
umap_emb = reducer.fit_transform(all_pca)

# Plot colored by Δy₂
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color by label
sc = axes[0].scatter(umap_emb[:, 0], umap_emb[:, 1], 
                      c=labels, cmap="RdBu_r", s=5, alpha=0.6,
                      vmin=-10, vmax=10)
axes[0].set_title("UMAP colored by Δy₂ (bp)")
plt.colorbar(sc, ax=axes[0], label="Δy₂ (bp)")

# Color by year
years = fed_chunks["date"].dt.year.values
sc2 = axes[1].scatter(umap_emb[:, 0], umap_emb[:, 1],
                       c=years, cmap="viridis", s=5, alpha=0.6)
axes[1].set_title("UMAP colored by Year")
plt.colorbar(sc2, ax=axes[1], label="Year")

plt.tight_layout()
plt.savefig("umap_stance_space.png", dpi=150)
plt.show()

print("✅ Latent stance space constructed and visualized.")


# ============================================================================
# CELL 13: Face Validity Check — Top Hawkish/Dovish Chunks
# ============================================================================

# Top 10 most hawkish chunks (by stance score)
top_hawk_idx = np.argsort(stance_scores)[-10:][::-1]
top_dove_idx = np.argsort(stance_scores)[:10]

print("=" * 70)
print("TOP 10 MOST HAWKISH CHUNKS (by latent stance score)")
print("=" * 70)
for i, idx in enumerate(top_hawk_idx):
    text = fed_chunks.iloc[idx]["text"][:200]
    score = stance_scores[idx]
    print(f"\n[{i+1}] Score: {score:.3f}")
    print(f"    {text}...")

print("\n" + "=" * 70)
print("TOP 10 MOST DOVISH CHUNKS (by latent stance score)")
print("=" * 70)
for i, idx in enumerate(top_dove_idx):
    text = fed_chunks.iloc[idx]["text"][:200]
    score = stance_scores[idx]
    print(f"\n[{i+1}] Score: {score:.3f}")
    print(f"    {text}...")


# ============================================================================
# CELL 14: Save All Results
# ============================================================================

# Save stance scores alongside chunk metadata
results_df = fed_chunks[["doc_id", "chunk_id", "date", "text"]].copy()
results_df["stance_score"] = stance_scores
results_df["delta_y2_bp"] = labels
results_df.to_parquet("stance_scores_all_chunks.parquet", index=False)

# Save training history
history = pd.DataFrame({
    "epoch": range(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "val_loss": [m["loss"] for m in val_metrics_history],
    "val_pearson_r": [m["pearson_r"] for m in val_metrics_history],
    "val_mae_bp": [m["mae_bp"] for m in val_metrics_history],
    "val_dir_accuracy": [m["dir_accuracy"] for m in val_metrics_history],
})
history.to_csv("training_history.csv", index=False)

# Summary
print("=" * 70)
print("PHASE 3 COMPLETE — FILES SAVED")
print("=" * 70)
print("  ✓ best_stance_model.pt (model weights)")
print("  ✓ stance_embeddings.npy (N×4096 embedding matrix)")
print("  ✓ stance_labels.npy (ground truth labels)")
print("  ✓ stance_scores_all_chunks.parquet (scores + metadata)")
print("  ✓ training_history.csv")
print("  ✓ umap_stance_space.png")
print()
print(f"  Best validation r: {best_val_r:.4f}")
print(f"  Test set r: {test_metrics['pearson_r']:.4f}")
print(f"  PC1 stance correlation: {r_pc1:.4f}")
print()
print("  Next: Download these files and proceed to Phase 4 (Robustness)")

# ============================================================================
# END OF PHASE 3
# ============================================================================
