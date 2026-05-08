"""
models_rnn.py — LSTM, GRU, GRU-LSTM, Conv1D-LSTM
Run: python models_rnn.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random, os
from utils.dataset   import normalize_per_battery, FEATURES, TARGET

# ═══════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
TRAIN_PATH  = 'dataset/Battery Data/train_dataset_4.csv'
TEST_PATH   = 'dataset/Battery Data/test_dataset_4.csv'

SEQ_LEN     = 100
PRED_LEN    = 50
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
PATIENCE    = 15        # early stopping




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}\n")


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════
class BatteryDataset(Dataset):
    def __init__(self, df, seq_len, pred_len):
        self.samples = []
        for cell_id, cell_df in df.groupby('cell'):
            cell_df = cell_df.sort_values('cycle').reset_index(drop=True)
            X = cell_df[FEATURES].values.astype(np.float32)
            y = cell_df[TARGET].values.astype(np.float32)
            n = len(cell_df)
            for i in range(0, n - seq_len - pred_len + 1, pred_len):
                self.samples.append((
                    X[i : i + seq_len],
                    y[i + seq_len : i + seq_len + pred_len]
                ))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


# ═══════════════════════════════════════════════════════════════
#  NORMALISATION  (per-battery min-max on train, applied to test)
# ═══════════════════════════════════════════════════════════════
def fit_normalize(train_df):
    stats = {}
    for col in FEATURES + [TARGET]:
        mn = train_df.groupby('cell')[col].min()
        mx = train_df.groupby('cell')[col].max()
        stats[col] = {'min': mn, 'max': mx}
    return stats

def apply_normalize(df, stats):
    df = df.copy()
    for col in FEATURES + [TARGET]:
        mn = stats[col]['min']
        mx = stats[col]['max']
        # use per-cell stats when available, else global fallback
        def norm_cell(grp):
            c = grp.name
            lo = mn.get(c, mn.mean())
            hi = mx.get(c, mx.mean())
            denom = hi - lo if hi != lo else 1.0
            grp[col] = (grp[col] - lo) / denom
            return grp
        df = df.groupby('cell', group_keys=False).apply(norm_cell)
    return df


# ═══════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════
N_FEATURES = len(FEATURES)

# ── 1. LSTM ───────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden=128, layers=2,
                 dropout=0.2, pred_len=PRED_LEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, pred_len)
        )

    def forward(self, x):                   # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])     # last timestep → pred_len


# ── 2. GRU ────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden=128, layers=2,
                 dropout=0.2, pred_len=PRED_LEN):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden, layers,
                           batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, pred_len)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# ── 3. GRU-LSTM ───────────────────────────────────────────────
class GRULSTMModel(nn.Module):
    """GRU extracts local patterns → LSTM models long-range dependencies."""
    def __init__(self, input_size=N_FEATURES, hidden=128, dropout=0.2,
                 pred_len=PRED_LEN):
        super().__init__()
        self.gru  = nn.GRU( input_size, hidden, num_layers=1,
                            batch_first=True)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1,
                            batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, pred_len)
        )

    def forward(self, x):
        g, _  = self.gru(x)
        g      = self.drop(g)
        l, _  = self.lstm(g)
        return self.head(l[:, -1, :])


# ── 4. Conv1D-LSTM ────────────────────────────────────────────
class Conv1DLSTMModel(nn.Module):
    """
    Conv1D extracts local feature patterns across time,
    then LSTM models temporal dependencies on the compressed sequence.
    """
    def __init__(self, input_size=N_FEATURES, hidden=128, dropout=0.2,
                 pred_len=PRED_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(64, hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, pred_len)
        )

    def forward(self, x):                   # x: (B, T, F)
        c = self.conv(x.permute(0, 2, 1))   # → (B, 64, T)
        c = c.permute(0, 2, 1)              # → (B, T, 64)
        l, _ = self.lstm(c)
        return self.head(l[:, -1, :])


# ═══════════════════════════════════════════════════════════════
#  SHARED TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader, name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=0.01)

    best_val, no_improve = float('inf'), 0
    history = {'train': [], 'val': []}

    for epoch in range(EPOCHS):
        # ── train ──
        model.train()
        tl = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # light jitter augmentation
            x = x + torch.randn_like(x) * 0.002
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl.append(loss.item())

        # ── val ──
        model.eval()
        vl = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vl.append(criterion(model(x), y).item())

        tl_m, vl_m = np.mean(tl), np.mean(vl)
        history['train'].append(tl_m)
        history['val'].append(vl_m)
        scheduler.step()

        if vl_m < best_val:
            best_val = vl_m
            no_improve = 0
            torch.save(model.state_dict(), f'best_{name}.pth')
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"  [{name}] epoch {epoch+1:3d} | train {tl_m:.5f} | val {vl_m:.5f}")

        if no_improve >= PATIENCE:
            print(f"  [{name}] early stop at epoch {epoch+1}")
            break

    # reload best weights
    model.load_state_dict(torch.load(f'best_{name}.pth', map_location=device))
    return history


def evaluate_model(model, test_df_norm, name):
    model.eval()
    all_preds, all_trues = [], []
    covered = set()

    for cell_id, cell_df in test_df_norm.groupby('cell'):
        cell_df = cell_df.sort_values('cycle').reset_index(drop=True)
        X = cell_df[FEATURES].values.astype(np.float32)
        y = cell_df[TARGET].values.astype(np.float32)
        n = len(cell_df)

        for start in range(0, n - SEQ_LEN - PRED_LEN + 1, PRED_LEN):
            x_t = torch.tensor(X[start:start+SEQ_LEN]).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_t).squeeze().cpu().numpy()

            true = y[start+SEQ_LEN : start+SEQ_LEN+PRED_LEN]

            for i in range(PRED_LEN):
                key = (cell_id, start + SEQ_LEN + i)
                if key not in covered:
                    all_preds.append(pred[i])
                    all_trues.append(true[i])
                    covered.add(key)

    y_p = np.array(all_preds) * 100
    y_t = np.array(all_trues) * 100

    mse  = mean_squared_error(y_t, y_p)
    mae  = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100
    r2   = r2_score(y_t, y_p)

    return {
        'MSE':  round(mse,  4),
        'MAE':  round(mae,  4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4),
        'R²':   round(r2,   4),
    }, y_p, y_t


def plot_loss(histories, model_names):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = [('steelblue','tomato'), ('orchid','coral'),
              ('seagreen','gold'), ('slateblue','orange')]
    for ax, (name, h), (c1, c2) in zip(axes.flat, histories.items(), colors):
        ax.plot(h['train'], color=c1, lw=2, label='Train')
        ax.plot(h['val'],   color=c2, lw=2, label='Val')
        ax.set_title(name); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale('log')
    plt.suptitle('Loss curves — all models', fontsize=14)
    plt.tight_layout()
    plt.savefig('loss_curves_all.png', dpi=150)
    plt.show()


def plot_predictions(predictions, model_names, test_df_norm):
    """One figure per model showing predicted vs real SOH for all test cells."""
    test_cells = test_df_norm['cell'].unique()
    n_cells    = len(test_cells)

    for name in model_names:
        fig, axes = plt.subplots(1, n_cells, figsize=(5*n_cells, 4), sharey=True)
        if n_cells == 1: axes = [axes]

        model = MODEL_REGISTRY[name]()
        model.load_state_dict(torch.load(f'best_{name}.pth', map_location=device))
        model.eval().to(device)

        for ax, cell_id in zip(axes, test_cells):
            cell_df = test_df_norm[test_df_norm['cell']==cell_id].sort_values('cycle').reset_index(drop=True)
            X = cell_df[FEATURES].values.astype(np.float32)
            y = cell_df[TARGET].values.astype(np.float32)
            n = len(cell_df)

            preds, trues, pred_cycles = [], [], []
            for start in range(0, n - SEQ_LEN - PRED_LEN + 1, PRED_LEN):
                x_t = torch.tensor(X[start:start+SEQ_LEN]).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = model(x_t).squeeze().cpu().numpy()
                preds.extend(p * 100)
                trues.extend(y[start+SEQ_LEN:start+SEQ_LEN+PRED_LEN] * 100)
                pred_cycles.extend(range(start+SEQ_LEN+1, start+SEQ_LEN+PRED_LEN+1))

            ax.plot(range(1, SEQ_LEN+1), y[:SEQ_LEN]*100,
                    color='orchid', lw=2, label='History')
            ax.plot(pred_cycles, trues, color='black', lw=2, label='Real SOH')
            ax.plot(pred_cycles, preds, color='tomato', lw=2, ls='--', label='Predicted')
            ax.axhline(80, color='orange', ls='--', lw=1, label='EOL 80%')
            ax.set_title(cell_id); ax.set_xlabel('Cycle'); ax.grid(True, alpha=0.3)
            if ax == axes[0]: ax.set_ylabel('SOH (%)')
            ax.legend(fontsize=8)

        fig.suptitle(f'{name} — SOH predictions', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'predictions_{name}.png', dpi=150)
        plt.show()
        print(f"📊 Plot saved: predictions_{name}.png")


# ═══════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    'LSTM':        lambda: LSTMModel().to(device),
    'GRU':         lambda: GRUModel().to(device),
    'GRU-LSTM':    lambda: GRULSTMModel().to(device),
    'Conv1D-LSTM': lambda: Conv1DLSTMModel().to(device),
}

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── Load data ─────────────────────────────────────────────
    print("📂 Loading data...")
    train_df = pd.read_csv(TRAIN_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
    test_df  = pd.read_csv(TEST_PATH ).sort_values(['cell','cycle']).reset_index(drop=True)
    print(f"   Train: {len(train_df):,} rows | {train_df['cell'].nunique()} batteries")
    print(f"   Test : {len(test_df):,} rows  | {test_df['cell'].nunique()} batteries\n")

    # ── Normalise ─────────────────────────────────────────────
    print("📊 Normalising (per-battery min-max)...")
    stats        = fit_normalize(train_df)
    train_norm   = apply_normalize(train_df, stats)
    test_norm    = apply_normalize(test_df,  stats)

    # ── SOH range diagnostic ──────────────────────────────────
    print("\n🔍 SOH range per test cell:")
    for cell, g in test_df.groupby('cell'):
        print(f"   {cell}: {g[TARGET].min():.2f}% → {g[TARGET].max():.2f}%  ({len(g)} cycles)")

    # ── Datasets / loaders ────────────────────────────────────
    train_dataset = BatteryDataset(train_norm, SEQ_LEN, PRED_LEN)
    test_dataset  = BatteryDataset(test_norm,  SEQ_LEN, PRED_LEN)
    train_loader  = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(test_dataset,  BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"\n✅ Train samples: {len(train_dataset)} | Val samples: {len(test_dataset)}\n")

    # ── Train & evaluate all models ───────────────────────────
    all_results  = {}
    all_histories = {}

    for model_name, model_fn in MODEL_REGISTRY.items():
        print(f"\n{'═'*50}")
        print(f"  Training {model_name}")
        print(f"{'═'*50}")

        model   = model_fn()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        history = train_model(model, train_loader, val_loader, model_name)
        all_histories[model_name] = history

        metrics, y_pred, y_true = evaluate_model(model, test_norm, model_name)
        all_results[model_name] = metrics
        print(f"  ✅ {model_name} — R²: {metrics['R²']:.4f} | RMSE: {metrics['RMSE']:.4f}%")

    # ── Results table ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  FINAL RESULTS")
    print(f"{'═'*60}")
    results_df = pd.DataFrame(all_results).T
    print(results_df.to_string())
    results_df.to_csv('./results/results_all_models.csv')
    print("\n💾 Saved: results_all_models.csv")

    # ── Plots ─────────────────────────────────────────────────
    plot_loss(all_histories, list(MODEL_REGISTRY.keys()))
    plot_predictions(all_results, list(MODEL_REGISTRY.keys()), test_norm)

    print("\n✅ Done. Check results_all_models.csv and the .png files.")