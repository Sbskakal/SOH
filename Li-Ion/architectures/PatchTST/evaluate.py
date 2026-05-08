# evaluate.py — Protocol matching the reference table
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.patchtst import PatchTST
from utils.dataset import load_data, normalize_per_battery, FEATURES, TARGET

SEQ_LEN    = 100
PRED_LEN   = 50
CSV_PATH   = 'dataset/Battery Data/cleaned_dataset_with_dqdv_peak_final_final.csv'
TEST_CELLS = ['B2C15', 'B2C35', 'B2C42', 'B2C46', 'B2C08']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = PatchTST(seq_len=SEQ_LEN, pred_len=PRED_LEN).to(device)
# In evaluate.py — replace your model = PatchTST(...) block with this:

model = PatchTST(
    seq_len    = SEQ_LEN,
    pred_len   = PRED_LEN,
    patch_len  = 16,
    stride     = 8,
    n_features = 8,
    d_model    = 256,   # ← was 128
    n_heads    = 8,
    n_layers   = 4,     # ← was 3
    dropout    = 0.15   # ← was 0.1 (doesn't affect weights, but keep consistent)
).to(device)
model.load_state_dict(torch.load('best_patchtst.pth', map_location=device))
model.eval()
print("✅ Modèle chargé\n")

df      = load_data(CSV_PATH)
df_norm = normalize_per_battery(df)
results = {}

for cell_id in TEST_CELLS:
    cell_df  = df_norm[df_norm['cell'] == cell_id].reset_index(drop=True)
    n_cycles = len(cell_df)

    if n_cycles < SEQ_LEN + PRED_LEN:
        print(f"⚠️  {cell_id} ignorée (trop courte)")
        continue

    X_all = cell_df[FEATURES].values.astype(np.float32)
    y_all = cell_df[TARGET].values.astype(np.float32)

    # ── PROTOCOL: Give first 100 real cycles, predict ALL remaining ──
    # Use only real features (no recursive injection of predicted SOH)
    # Slide window but always use REAL X features, predict SOH only
    
    all_preds = []
    all_trues = []
    covered   = set()

    for start in range(0, n_cycles - SEQ_LEN - PRED_LEN + 1, PRED_LEN):
        end_input = start + SEQ_LEN
        end_pred  = end_input + PRED_LEN

        x_window = X_all[start:end_input]           # real features only
        y_true   = y_all[end_input:end_pred] * 100

        x_tensor = torch.tensor(x_window).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(x_tensor).squeeze().cpu().numpy() * 100

        # Only add non-overlapping predictions
        for i, cycle_idx in enumerate(range(end_input, end_pred)):
            if cycle_idx not in covered:
                all_preds.append(y_pred[i])
                all_trues.append(y_true[i])
                covered.add(cycle_idx)

    y_pred_arr = np.array(all_preds)
    y_true_arr = np.array(all_trues)

    mse  = mean_squared_error(y_true_arr, y_pred_arr)
    mae  = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / (y_true_arr + 1e-8))) * 100
    r2   = r2_score(y_true_arr, y_pred_arr)

    results[cell_id] = {
        'Cycle Life': n_cycles,
        'MSE' : round(mse,  4),
        'MAE' : round(mae,  4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4),
        'R²'  : round(r2,   4)
    }

    # Plot
    hist_cycles = range(1, SEQ_LEN + 1)
    hist_soh    = y_all[:SEQ_LEN] * 100
    pred_cycles = range(SEQ_LEN + 1, SEQ_LEN + 1 + len(y_pred_arr))

    plt.figure(figsize=(12, 5))
    plt.plot(hist_cycles, hist_soh,    color='orchid', lw=2, label='Input (100 cycles)')
    plt.plot(pred_cycles, y_true_arr,  color='black',  lw=2, label='SOH Réel')
    plt.plot(pred_cycles, y_pred_arr,  color='tomato', lw=2, ls='--', label='SOH Prédit')
    plt.axvline(SEQ_LEN, color='gray', ls=':', lw=1.5, label='Début prédiction')
    plt.axhline(80, color='orange', ls='--', lw=1, label='EOL (80%)')
    plt.xlabel('Cycle'); plt.ylabel('SOH (%)')
    plt.title(f'{cell_id} — R²={r2:.4f} | RMSE={rmse:.4f}% | {n_cycles} cycles')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f'results_{cell_id}.png', dpi=150)
    plt.close()
    print(f"✅ {cell_id} — RMSE: {rmse:.4f}% | R²: {r2:.4f} | points évalués: {len(y_pred_arr)}")

print("\n══════ Résultats PatchTST ══════")
results_df = pd.DataFrame(results).T
print(results_df.to_string())
results_df.to_csv('results_patchtst.csv')
print("\n💾 Sauvegardé dans results_patchtst.csv")