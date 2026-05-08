import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.patchtst   import PatchTST
from models.itransformers import iTransformer
from utils.dataset     import normalize_per_battery, FEATURES, TARGET

# ── Config ────────────────────────────────────────────────────
TRAIN_PATH = 'dataset/Battery Data/train_dataset_4.csv'
TEST_PATH  = 'dataset/Battery Data/test_dataset_4.csv'
PRED_LEN   = 50
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-4

# ← Le cœur de l'Early Prediction
SEQ_LENS   = [30, 50, 70, 100]


# ── Dataset adapté pour Early Prediction ──────────────────────
class EarlyBatteryDataset(Dataset):
    """
    Même logique que BatteryDataset mais avec seq_len variable.
    Permet de tester différents historiques (30, 50, 70, 100 cycles).
    """
    def __init__(self, df, seq_len, pred_len):
        self.samples = []

        for cell_id in df['cell'].unique():
            cell_df = df[df['cell'] == cell_id].reset_index(drop=True)
            n = len(cell_df)

            if n < seq_len + pred_len:
                continue

            X = cell_df[FEATURES].values.astype(np.float32)
            y = cell_df[TARGET].values.astype(np.float32)

            for i in range(n - seq_len - pred_len + 1):
                self.samples.append((
                    X[i : i + seq_len],
                    y[i + seq_len : i + seq_len + pred_len]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


# ── Chargement des données ─────────────────────────────────────
train_df   = pd.read_csv(TRAIN_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
test_df    = pd.read_csv(TEST_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
train_norm = normalize_per_battery(train_df)
test_norm  = normalize_per_battery(test_df)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device : {device}\n")


def build_model(model_name, seq_len, pred_len=50, n_features=8):
    """Construit le modèle selon le nom et le seq_len."""
    if model_name == 'PatchTST':
        # patch_len adapté selon seq_len
        patch_len = min(16, seq_len // 4)
        stride    = patch_len // 2
        return PatchTST(
            seq_len    = seq_len,
            pred_len   = pred_len,
            patch_len  = patch_len,
            stride     = stride,
            n_features = n_features,
            d_model    = 128,
            n_heads    = 8,
            n_layers   = 3,
            dropout    = 0.1
        )
    elif model_name == 'iTransformer':
        return iTransformer(
            seq_len    = seq_len,
            pred_len   = pred_len,
            n_features = n_features,
            d_model    = 128,
            n_heads    = 8,
            n_layers   = 3,
            dropout    = 0.1
        )


def train_one_model(model, train_loader, test_loader, model_name, seq_len):
    """Entraîne un modèle et retourne le meilleur val loss."""
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()
    best_val  = float('inf')
    history   = {'train': [], 'val': []}

    for epoch in range(EPOCHS):

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

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(criterion(model(x), y).item())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        history['train'].append(tl)
        history['val'].append(vl)
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            save_name = f'best_{model_name}_seq{seq_len}.pth'
            torch.save(model.state_dict(), save_name)

        if (epoch + 1) % 20 == 0:
            print(f"    Ep {epoch+1:3d} | Train: {tl:.5f} | Val: {vl:.5f}")

    return best_val, history


# ── Boucle principale : tous modèles × tous seq_len ───────────
MODEL_NAMES = ['PatchTST', 'iTransformer']
all_histories = {}   # histories[model_name][seq_len]
summary       = []   # pour le tableau récapitulatif

for seq_len in SEQ_LENS:
    print(f"\n{'='*60}")
    print(f"📏 SEQ_LEN = {seq_len} cycles ({seq_len/530*100:.0f}% historique moyen)")
    print(f"{'='*60}")

    train_dataset = EarlyBatteryDataset(train_norm, seq_len, PRED_LEN)
    test_dataset  = EarlyBatteryDataset(test_norm,  seq_len, PRED_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    print(f"   Train windows : {len(train_dataset)} | Test windows : {len(test_dataset)}")

    for model_name in MODEL_NAMES:
        print(f"\n🚀 {model_name} — seq_len={seq_len}")
        model    = build_model(model_name, seq_len, PRED_LEN)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Paramètres : {n_params:,}")

        best_val, history = train_one_model(
            model, train_loader, test_loader, model_name, seq_len
        )

        key = f"{model_name}_seq{seq_len}"
        all_histories[key] = history
        summary.append({
            'Model'      : model_name,
            'seq_len'    : seq_len,
            'Best_ValLoss': round(best_val, 6)
        })
        print(f"   ✅ Best Val Loss : {best_val:.6f}")

# ── Tableau récapitulatif ──────────────────────────────────────
print("\n" + "="*60)
print("RÉSUMÉ ENTRAÎNEMENT")
print("="*60)
df_summary = pd.DataFrame(summary)
df_pivot   = df_summary.pivot(index='seq_len', columns='Model', values='Best_ValLoss')
print(df_pivot.to_string())
df_pivot.to_csv('training_summary_online.csv')

# ── Courbes de loss ────────────────────────────────────────────
fig, axes = plt.subplots(len(SEQ_LENS), len(MODEL_NAMES),
                         figsize=(14, 4 * len(SEQ_LENS)))

for i, seq_len in enumerate(SEQ_LENS):
    for j, model_name in enumerate(MODEL_NAMES):
        key  = f"{model_name}_seq{seq_len}"
        hist = all_histories[key]
        ax   = axes[i][j]
        ax.plot(hist['train'], color='steelblue', label='Train', linewidth=2)
        ax.plot(hist['val'],   color='tomato',    label='Val',   linewidth=2)
        ax.set_title(f'{model_name} | seq_len={seq_len}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.suptitle('Early Prediction — Courbes d\'apprentissage', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('loss_curves_online.png', dpi=150)
plt.show()
print("\n💾 loss_curves_online.png sauvegardé")