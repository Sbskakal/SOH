import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.rnn_models import LSTMModel, GRUModel, GRULSTMModel, Conv1DLSTMModel
from utils.dataset     import normalize_per_battery, BatteryDataset

# ── Config (identique à PatchTST) ────────────────────────────
TRAIN_PATH = 'dataset/Battery Data/train_dataset_4.csv'
TEST_PATH  = 'dataset/Battery Data/test_dataset_4.csv'
SEQ_LEN    = 100
PRED_LEN   = 50
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-4

# ── Données ───────────────────────────────────────────────────
train_df   = pd.read_csv(TRAIN_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
test_df    = pd.read_csv(TEST_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
train_norm = normalize_per_battery(train_df)
test_norm  = normalize_per_battery(test_df)

train_dataset = BatteryDataset(train_norm, SEQ_LEN, PRED_LEN)
test_dataset  = BatteryDataset(test_norm,  SEQ_LEN, PRED_LEN)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device : {device}\n")

# ── Définition des modèles ────────────────────────────────────
MODELS = {
    'LSTM'        : LSTMModel(n_features=8, hidden_size=128, n_layers=3, pred_len=PRED_LEN),
    'GRU'         : GRUModel(n_features=8,  hidden_size=128, n_layers=3, pred_len=PRED_LEN),
    'GRU-LSTM'    : GRULSTMModel(n_features=8, hidden_size=128, n_layers=2, pred_len=PRED_LEN),
    'Conv1DLSTM'  : Conv1DLSTMModel(n_features=8, hidden_size=128, n_layers=2, pred_len=PRED_LEN),
}

all_histories = {}

# ── Entraînement de chaque modèle ─────────────────────────────
for model_name, model in MODELS.items():
    print(f"{'='*50}")
    print(f"🚀 Entraînement : {model_name}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Paramètres : {n_params:,}")
    print(f"{'='*50}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()
    history   = {'train': [], 'val': []}
    best_val  = float('inf')

    for epoch in range(EPOCHS):

        # Train
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

        # Validation
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
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            saved = "✅"
        else:
            saved = ""

        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch+1:3d}/{EPOCHS} | Train: {tl:.5f} | Val: {vl:.5f} {saved}")

    print(f"\n🏆 {model_name} — Meilleur Val Loss : {best_val:.5f}\n")
    all_histories[model_name] = history

# ── Courbes de loss comparatives ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
colors = {'train': 'steelblue', 'val': 'tomato'}

for i, (name, hist) in enumerate(all_histories.items()):
    axes[i].plot(hist['train'], color=colors['train'], label='Train Loss', linewidth=2)
    axes[i].plot(hist['val'],   color=colors['val'],   label='Val Loss',   linewidth=2)
    axes[i].set_title(name, fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('MSE Loss')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Courbes d\'apprentissage — Modèles RNN', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('loss_curves_rnn.png', dpi=150)
plt.show()
print("💾 Courbes sauvegardées : loss_curves_rnn.png")