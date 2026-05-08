import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.patchtst     import PatchTST
from models.itransformers import iTransformer
from utils.dataset       import normalize_per_battery, FEATURES, TARGET

TEST_PATH  = 'dataset/Battery Data/test_dataset_4.csv'
TEST_CELLS = ['B2C20', 'B2C22', 'B2C28', 'B2C34', 'B2C35', 'B2C46']
PRED_LEN   = 50
SEQ_LENS   = [30, 50, 70, 100]
MODEL_NAMES = ['PatchTST', 'iTransformer']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_name, seq_len, pred_len=50, n_features=8):
    if model_name == 'PatchTST':
        patch_len = min(16, seq_len // 4)
        stride    = patch_len // 2
        model = PatchTST(
            seq_len=seq_len, pred_len=pred_len,
            patch_len=patch_len, stride=stride,
            n_features=n_features, d_model=128,
            n_heads=8, n_layers=3, dropout=0.1
        )
    else:
        model = iTransformer(
            seq_len=seq_len, pred_len=pred_len,
            n_features=n_features, d_model=128,
            n_heads=8, n_layers=3, dropout=0.1
        )
    path = f'best_{model_name}_seq{seq_len}.pth'
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_recursive(model, X_all, y_all, seq_len, pred_len):
    """Prédiction récursive sur toute la vie de la batterie."""
    all_preds = []
    all_trues = []
    current   = 0
    n_cycles  = len(X_all)

    while current + seq_len + pred_len <= n_cycles:
        x_t = torch.tensor(
            X_all[current : current + seq_len]
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(x_t).squeeze().cpu().numpy()

        y_true = y_all[current + seq_len : current + seq_len + pred_len]
        all_preds.extend(y_pred.tolist())
        all_trues.extend(y_true.tolist())
        current += pred_len

    return np.array(all_preds) * 100, np.array(all_trues) * 100


def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return {'MSE': round(mse,4), 'MAE': round(mae,4),
            'RMSE': round(rmse,4), 'MAPE': round(mape,4), 'R²': round(r2,4)}


# ── Données ───────────────────────────────────────────────────
test_df   = pd.read_csv(TEST_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
test_norm = normalize_per_battery(test_df)

# ── Évaluation ────────────────────────────────────────────────
# all_results[model_name][seq_len][cell_id] = métriques
all_results = {m: {s: {} for s in SEQ_LENS} for m in MODEL_NAMES}

for seq_len in SEQ_LENS:
    print(f"\n{'='*55}")
    print(f"📏 Évaluation seq_len = {seq_len}")
    print(f"{'='*55}")

    for model_name in MODEL_NAMES:
        model = load_model(model_name, seq_len)
        print(f"\n  🔍 {model_name}")

        for cell_id in TEST_CELLS:
            cell_df  = test_norm[test_norm['cell'] == cell_id].reset_index(drop=True)
            n_cycles = len(cell_df)

            if n_cycles < seq_len + PRED_LEN:
                print(f"    ⚠️  {cell_id} ignorée")
                continue

            X_all = cell_df[FEATURES].values.astype('float32')
            y_all = cell_df[TARGET].values.astype('float32')

            y_pred_pct, y_true_pct = evaluate_recursive(
                model, X_all, y_all, seq_len, PRED_LEN
            )
            metrics = compute_metrics(y_true_pct, y_pred_pct)
            all_results[model_name][seq_len][cell_id] = metrics
            print(f"    {cell_id} | R²={metrics['R²']:.4f} | RMSE={metrics['RMSE']:.4f}")

# ── Tableaux de résultats ──────────────────────────────────────
metrics_list = ['MSE', 'MAE', 'RMSE', 'MAPE', 'R²']
records      = []

for model_name in MODEL_NAMES:
    for seq_len in SEQ_LENS:
        for cell_id in TEST_CELLS:
            if cell_id in all_results[model_name][seq_len]:
                rec = {
                    'Model'  : model_name,
                    'seq_len': seq_len,
                    'Battery': cell_id
                }
                rec.update(all_results[model_name][seq_len][cell_id])
                records.append(rec)

df_all = pd.DataFrame(records)
df_all.to_csv('results_online_all.csv', index=False)

# ── Tableau principal : R² par seq_len ────────────────────────
print("\n" + "="*65)
print("R² — Impact de l'historique (Early Prediction)")
print("="*65)
for model_name in MODEL_NAMES:
    print(f"\n── {model_name}")
    rows = {}
    for seq_len in SEQ_LENS:
        row = {}
        for cell_id in TEST_CELLS:
            val = all_results[model_name][seq_len].get(cell_id, {}).get('R²', None)
            row[cell_id] = val
        rows[f'seq={seq_len}'] = row
    df_r2 = pd.DataFrame(rows).T
    df_r2['Moyenne'] = df_r2.mean(axis=1).round(4)
    print(df_r2.to_string())

# ── Plot 1 : R² vs seq_len pour chaque modèle ─────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors  = {'PatchTST': 'steelblue', 'iTransformer': 'tomato'}

for model_name in MODEL_NAMES:
    mean_r2 = []
    for seq_len in SEQ_LENS:
        vals = [
            all_results[model_name][seq_len][c]['R²']
            for c in TEST_CELLS
            if c in all_results[model_name][seq_len]
        ]
        mean_r2.append(np.mean(vals))

    ax.plot(SEQ_LENS, mean_r2, marker='o', linewidth=2.5, markersize=8,
            color=colors[model_name], label=model_name)
    for x, y in zip(SEQ_LENS, mean_r2):
        ax.annotate(f'{y:.4f}', (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color=colors[model_name])

ax.set_xlabel('Historique disponible (seq_len en cycles)', fontsize=12)
ax.set_ylabel('R² moyen', fontsize=12)
ax.set_title('Early Prediction — R² vs Historique\nPatchTST vs iTransformer', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(SEQ_LENS)
ax.set_xticklabels([f'{s} cycles\n({s/530*100:.0f}%)' for s in SEQ_LENS])
plt.tight_layout()
plt.savefig('r2_vs_seqlen.png', dpi=150)
plt.show()

# ── Plot 2 : Courbes SOH par batterie pour seq_len=30 et 100 ──
COLORS_MODEL = {'PatchTST': 'steelblue', 'iTransformer': 'tomato'}

for cell_id in TEST_CELLS:
    cell_df  = test_norm[test_norm['cell'] == cell_id].reset_index(drop=True)
    n_cycles = len(cell_df)
    y_all    = cell_df[TARGET].values.astype('float32')

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax_idx, seq_len in enumerate([30, 100]):
        ax = axes[ax_idx]

        # Historique réel
        hist_soh    = y_all[:seq_len] * 100
        hist_cycles = range(1, seq_len + 1)
        ax.plot(hist_cycles, hist_soh,
                color='orchid', linewidth=2.5,
                label=f'Historique ({seq_len} cycles)')

        # SOH réel futur (calculé une fois)
        X_all   = cell_df[FEATURES].values.astype('float32')
        current = 0
        trues   = []
        while current + seq_len + PRED_LEN <= n_cycles:
            trues.extend(y_all[current+seq_len : current+seq_len+PRED_LEN].tolist())
            current += PRED_LEN

        pred_cycles = range(seq_len + 1, seq_len + 1 + len(trues))
        ax.plot(pred_cycles, np.array(trues) * 100,
                color='black', linewidth=2.5, label='SOH Réel')

        # Prédictions de chaque modèle
        for model_name in MODEL_NAMES:
            if n_cycles < seq_len + PRED_LEN:
                continue
            model   = load_model(model_name, seq_len)
            y_pred_pct, _ = evaluate_recursive(
                model, X_all, y_all, seq_len, PRED_LEN
            )
            r2 = all_results[model_name][seq_len].get(cell_id, {}).get('R²', 0)
            ax.plot(pred_cycles, y_pred_pct,
                    color=COLORS_MODEL[model_name], linewidth=2,
                    linestyle='--',
                    label=f'{model_name} (R²={r2:.4f})')

        ax.axvline(seq_len, color='gray',   linestyle=':', linewidth=1.5)
        ax.axhline(80,      color='orange', linestyle='--', linewidth=1,
                   label='EOL (80%)')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('SOH (%)')
        ax.set_title(f'seq_len = {seq_len} cycles', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{cell_id} — Early Prediction : 30 vs 100 cycles d\'historique',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'online_{cell_id}.png', dpi=150)
    plt.show()
    print(f"✅ Plot sauvegardé : online_{cell_id}.png")

print("\n✅ Évaluation terminée — résultats dans results_online_all.csv")