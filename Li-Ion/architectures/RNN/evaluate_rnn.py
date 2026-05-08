import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.rnn_models import LSTMModel, GRUModel, GRULSTMModel, Conv1DLSTMModel
from utils.dataset     import normalize_per_battery, FEATURES, TARGET

TEST_PATH  = 'dataset/Battery Data/B2C08_data.csv'
SEQ_LEN    = 100
PRED_LEN   = 50
TEST_CELLS = ['B2C08']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Chargement des modèles ────────────────────────────────────
MODELS = {
    'LSTM'       : LSTMModel(n_features=8,       pred_len=PRED_LEN),
    'GRU'        : GRUModel(n_features=8,         pred_len=PRED_LEN),
    'GRU-LSTM'   : GRULSTMModel(n_features=8,     pred_len=PRED_LEN),
    'Conv1DLSTM' : Conv1DLSTMModel(n_features=8,  pred_len=PRED_LEN),
}

for name, model in MODELS.items():
    model.load_state_dict(torch.load(f'best_{name}.pth', map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ {name} chargé")

# ── Données ───────────────────────────────────────────────────
test_df   = pd.read_csv(TEST_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
test_norm = normalize_per_battery(test_df)

# ── Évaluation récursive ──────────────────────────────────────
# Structure : results[model_name][cell_id] = dict métriques
all_results = {name: {} for name in MODELS}

for cell_id in TEST_CELLS:
    cell_df  = test_norm[test_norm['cell'] == cell_id].reset_index(drop=True)
    n_cycles = len(cell_df)

    if n_cycles < SEQ_LEN + PRED_LEN:
        print(f"⚠️  {cell_id} ignorée ({n_cycles} cycles)")
        continue

    X_all = cell_df[FEATURES].values.astype('float32')
    y_all = cell_df[TARGET].values.astype('float32')

    for model_name, model in MODELS.items():
        all_preds = []
        all_trues = []
        current   = 0

        while current + SEQ_LEN + PRED_LEN <= n_cycles:
            x_t = torch.tensor(
                X_all[current : current + SEQ_LEN]
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(x_t).squeeze().cpu().numpy()

            y_true = y_all[current + SEQ_LEN : current + SEQ_LEN + PRED_LEN]
            all_preds.extend(y_pred.tolist())
            all_trues.extend(y_true.tolist())
            current += PRED_LEN

        y_pred_pct = np.array(all_preds) * 100
        y_true_pct = np.array(all_trues) * 100

        mse  = mean_squared_error(y_true_pct, y_pred_pct)
        mae  = mean_absolute_error(y_true_pct, y_pred_pct)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true_pct - y_pred_pct) / (y_true_pct + 1e-8))) * 100
        r2   = r2_score(y_true_pct, y_pred_pct)

        all_results[model_name][cell_id] = {
            'MSE': round(mse, 4), 'MAE': round(mae, 4),
            'RMSE': round(rmse, 4), 'MAPE': round(mape, 4), 'R²': round(r2, 4)
        }

# ── Plots par batterie (tous modèles superposés) ──────────────
COLORS = {
    'LSTM': 'steelblue', 'GRU': 'green',
    'GRU-LSTM': 'orange', 'Conv1DLSTM': 'red'
}

for cell_id in TEST_CELLS:
    cell_df  = test_norm[test_norm['cell'] == cell_id].reset_index(drop=True)
    n_cycles = len(cell_df)
    if n_cycles < SEQ_LEN + PRED_LEN:
        continue

    X_all = cell_df[FEATURES].values.astype('float32')
    y_all = cell_df[TARGET].values.astype('float32')

    hist_cycles = range(1, SEQ_LEN + 1)
    hist_soh    = y_all[:SEQ_LEN] * 100

    plt.figure(figsize=(13, 5))
    plt.plot(hist_cycles, hist_soh, color='orchid', linewidth=2.5,
             label='Historique (100 cycles)')

    # SOH réel (calculé une fois)
    current   = 0
    all_trues = []
    while current + SEQ_LEN + PRED_LEN <= n_cycles:
        y_true = y_all[current + SEQ_LEN : current + SEQ_LEN + PRED_LEN]
        all_trues.extend(y_true.tolist())
        current += PRED_LEN

    pred_cycles = range(SEQ_LEN + 1, SEQ_LEN + 1 + len(all_trues))
    plt.plot(pred_cycles, np.array(all_trues) * 100,
             color='black', linewidth=2.5, label='SOH Réel')

    # Prédiction de chaque modèle
    for model_name, model in MODELS.items():
        current   = 0
        all_preds = []
        while current + SEQ_LEN + PRED_LEN <= n_cycles:
            x_t = torch.tensor(
                X_all[current : current + SEQ_LEN]
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                y_pred = model(x_t).squeeze().cpu().numpy()
            all_preds.extend(y_pred.tolist())
            current += PRED_LEN

        r2 = all_results[model_name].get(cell_id, {}).get('R²', 0)
        plt.plot(pred_cycles, np.array(all_preds) * 100,
                 color=COLORS[model_name], linewidth=1.8,
                 linestyle='--', label=f'{model_name} (R²={r2:.4f})')

    plt.axvline(SEQ_LEN, color='gray',   linestyle=':',  linewidth=1.5)
    plt.axhline(80,       color='orange', linestyle='--', linewidth=1,
                label='Seuil EOL (80%)')
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH (%)')
    plt.title(f'{cell_id} — Comparaison LSTM / GRU / GRU-LSTM / Conv1DLSTM')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'compare_rnn_{cell_id}.png', dpi=150)
    plt.show()
    print(f"✅ Plot sauvegardé : compare_rnn_{cell_id}.png")

# ── Tableaux récapitulatifs — un par métrique ─────────────────
metrics = ['MSE', 'MAE', 'RMSE', 'MAPE', 'R²']

print("\n" + "="*70)
print("RÉSULTATS COMPLETS — FORMAT IDENTIQUE À LA PRÉSENTATION DE HIND")
print("="*70)

for metric in metrics:
    print(f"\n── {metric} ──────────────────────────────")
    rows = {}
    for model_name in MODELS:
        row = {}
        for cell_id in TEST_CELLS:
            val = all_results[model_name].get(cell_id, {}).get(metric, None)
            row[cell_id] = val
        rows[model_name] = row
    df_metric = pd.DataFrame(rows).T
    df_metric['Moyenne'] = df_metric.mean(axis=1).round(4)
    print(df_metric.to_string())

# ── Export CSV global ─────────────────────────────────────────
records = []
for model_name in MODELS:
    for cell_id in TEST_CELLS:
        if cell_id in all_results[model_name]:
            rec = {'Model': model_name, 'Battery': cell_id}
            rec.update(all_results[model_name][cell_id])
            records.append(rec)

df_final = pd.DataFrame(records)
df_final.to_csv('results_rnn_all.csv', index=False)
print("\n💾 Résultats exportés : results_rnn_all.csv")

# ── Graphique comparatif des moyennes ─────────────────────────
df_mean = df_final.groupby('Model')[metrics].mean().round(4)
print("\n── Moyennes par modèle ──────────────────────")
print(df_mean.to_string())

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, metric in enumerate(metrics):
    vals   = df_mean[metric]
    colors = ['steelblue', 'green', 'orange', 'red']
    bars   = axes[i].bar(vals.index, vals.values, color=colors, width=0.5)
    axes[i].set_title(metric, fontweight='bold')
    axes[i].set_xticklabels(vals.index, rotation=30, ha='right', fontsize=9)
    axes[i].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals.values):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.001,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Comparaison moyenne — LSTM / GRU / GRU-LSTM / Conv1DLSTM',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('compare_rnn_metrics.png', dpi=150)
plt.show()
print("💾 Graphique comparatif : compare_rnn_metrics.png")