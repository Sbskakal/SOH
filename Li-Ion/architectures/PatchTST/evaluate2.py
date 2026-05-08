import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.patchtst import PatchTST
from utils.dataset   import normalize_per_battery, FEATURES, TARGET

TEST_PATH  = 'dataset/Battery Data/B2C08_data.csv'
SEQ_LEN    = 100
PRED_LEN   = 50

# Batteries test de Hind (mêmes que dans la présentation)
TEST_CELLS = ['B2C08']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Chargement modèle ─────────────────────────────────────────
model = PatchTST(seq_len=SEQ_LEN, pred_len=PRED_LEN).to(device)
model.load_state_dict(torch.load('best_patchtst_v2.pth', map_location=device))
model.eval()
print("✅ Modèle v2 chargé\n")

# ── Données test ──────────────────────────────────────────────
test_df   = pd.read_csv(TEST_PATH).sort_values(['cell','cycle']).reset_index(drop=True)
test_norm = normalize_per_battery(test_df)

results = {}

for cell_id in TEST_CELLS:
    cell_df  = test_norm[test_norm['cell'] == cell_id].reset_index(drop=True)
    n_cycles = len(cell_df)

    if n_cycles < SEQ_LEN + PRED_LEN:
        print(f"⚠️  {cell_id} : seulement {n_cycles} cycles, ignorée")
        continue

    X_all = cell_df[FEATURES].values.astype(np.float32)
    y_all = cell_df[TARGET].values.astype(np.float32)

    # ── Prédiction récursive sur toute la vie ─────────────────
    all_preds = []
    all_trues = []
    current   = 0

    while current + SEQ_LEN + PRED_LEN <= n_cycles:
        x_tensor = torch.tensor(
            X_all[current : current + SEQ_LEN]
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(x_tensor).squeeze().cpu().numpy()

        y_true = y_all[current + SEQ_LEN : current + SEQ_LEN + PRED_LEN]
        all_preds.extend(y_pred.tolist())
        all_trues.extend(y_true.tolist())
        current += PRED_LEN

    # Reconvertir en %
    y_pred_pct = np.array(all_preds) * 100
    y_true_pct = np.array(all_trues) * 100

    # ── Métriques ─────────────────────────────────────────────
    mse  = mean_squared_error(y_true_pct, y_pred_pct)
    mae  = mean_absolute_error(y_true_pct, y_pred_pct)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_pct - y_pred_pct) / (y_true_pct + 1e-8))) * 100
    r2   = r2_score(y_true_pct, y_pred_pct)

    # Pourcentage de cycles utilisés pour l'historique
    pct_history = round(SEQ_LEN / n_cycles * 100, 2)

    results[cell_id] = {
        'Cycle Life'   : n_cycles,
        '100 Cycles %' : pct_history,
        'MSE'          : round(mse,  4),
        'MAE'          : round(mae,  4),
        'RMSE'         : round(rmse, 4),
        'MAPE'         : round(mape, 4),
        'R²'           : round(r2,   4)
    }

    # ── Plot ──────────────────────────────────────────────────
    hist_soh    = y_all[:SEQ_LEN] * 100
    hist_cycles = range(1, SEQ_LEN + 1)
    pred_cycles = range(SEQ_LEN + 1, SEQ_LEN + 1 + len(y_pred_pct))

    plt.figure(figsize=(12, 5))
    plt.plot(hist_cycles,  hist_soh,   color='orchid',    linewidth=2,
             label=f'Historique ({SEQ_LEN} cycles donnés)')
    plt.plot(pred_cycles,  y_true_pct, color='black',     linewidth=2,
             label='SOH Réel')
    plt.plot(pred_cycles,  y_pred_pct, color='tomato',    linewidth=2,
             linestyle='--', label='SOH Prédit (PatchTST)')
    plt.axvline(SEQ_LEN, color='gray',   linestyle=':',  linewidth=1.5,
                label='Début prédiction')
    plt.axhline(80,       color='orange', linestyle='--', linewidth=1,
                label='Seuil EOL (80%)')
    plt.xlabel('Cycle Number')
    plt.ylabel('SOH (%)')
    plt.title(f'{cell_id} | Cycle Life: {n_cycles} | {pct_history}% history\n'
              f'R²={r2:.4f} | RMSE={rmse:.4f}% | MAPE={mape:.4f}%')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results_v2_{cell_id}.png', dpi=150)
    plt.show()
    print(f"✅ {cell_id} | Cycles: {n_cycles} | RMSE: {rmse:.4f}% | R²: {r2:.4f}")

# ── Tableau final — même format que la présentation de Hind ───
print("\n══════ Résultats PatchTST v2 (split officiel Hind) ══════")
results_df = pd.DataFrame(results).T
print(results_df.to_string())
results_df.to_csv('results_patchtst_v2.csv')
print("\n💾 Sauvegardé dans results_patchtst_v2.csv")

# ── Moyenne globale ───────────────────────────────────────────
print("\n══════ Moyennes ══════")
numeric_cols = ['MSE', 'MAE', 'RMSE', 'MAPE', 'R²']
print(results_df[numeric_cols].mean().round(4))