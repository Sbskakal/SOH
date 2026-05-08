import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.patchtst import PatchTST
from utils.dataset   import normalize_per_battery, BatteryDataset

# ── Config ────────────────────────────────────────────────────
TRAIN_PATH = 'dataset/Battery Data/train_dataset_4.csv'
TEST_PATH  = 'dataset/Battery Data/test_dataset_4.csv'
SEQ_LEN    = 100
PRED_LEN   = 50
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-4

# ── Chargement des splits déjà préparés ───────────────────────
train_df = pd.read_csv(TRAIN_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
test_df  = pd.read_csv(TEST_PATH).sort_values(['cell','cycle']).reset_index(drop=True)

print(f"✅ Train : {train_df.shape} | Batteries : {train_df['cell'].nunique()}")
print(f"✅ Test  : {test_df.shape}  | Batteries : {test_df['cell'].nunique()}")
print(f"   Train batteries : {sorted(train_df['cell'].unique())}")
print(f"   Test  batteries : {sorted(test_df['cell'].unique())}")

# ── Normalisation ─────────────────────────────────────────────
# Important : on normalise train et test séparément
# (chaque batterie normalisée sur ses propres min/max)
train_norm = normalize_per_battery(train_df)
test_norm  = normalize_per_battery(test_df)

# ── Datasets & Loaders ────────────────────────────────────────
train_dataset = BatteryDataset(train_norm, SEQ_LEN, PRED_LEN)
test_dataset  = BatteryDataset(test_norm,  SEQ_LEN, PRED_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Modèle ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device : {device}")

model = PatchTST(
    seq_len    = SEQ_LEN,
    pred_len   = PRED_LEN,
    patch_len  = 16,
    stride     = 8,
    n_features = 8,
    d_model    = 128,
    n_heads    = 8,
    n_layers   = 3,
    dropout    = 0.1
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"📊 Paramètres : {n_params:,}\n")

# ── Optimizer / Loss ──────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)
criterion = nn.MSELoss()

# ── Entraînement ──────────────────────────────────────────────
history  = {'train': [], 'val': []}
best_val = float('inf')

print("Epoch | Train Loss | Val Loss  |")
print("-" * 38)

for epoch in range(EPOCHS):

    # --- Train ---
    model.train()
    train_losses = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # --- Validation ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_losses.append(criterion(pred, y).item())

    tl = np.mean(train_losses)
    vl = np.mean(val_losses)
    history['train'].append(tl)
    history['val'].append(vl)
    scheduler.step(vl)

    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), 'best_patchtst_v2.pth')
        saved = " ✅"
    else:
        saved = ""

    if (epoch + 1) % 10 == 0:
        print(f"  {epoch+1:3d}  |  {tl:.5f}   |  {vl:.5f}  {saved}")

print(f"\n🏆 Meilleur Val Loss : {best_val:.5f}")

# ── Courbe de loss ────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(history['train'], label='Train Loss', color='steelblue')
plt.plot(history['val'],   label='Val Loss',   color='tomato')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('PatchTST v2  — Courbe apprentissage')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve_v2.png', dpi=150)
plt.show()