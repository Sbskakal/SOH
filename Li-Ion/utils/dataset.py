import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Features retenues d'après l'analyse de corrélation de la présentation
FEATURES = ['QC', 'QD', 'IR', 'Tavg', 'Ctime', 'V', 'I', 'dqdv_peak']
TARGET   = 'SOH_C'

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(['cell', 'cycle']).reset_index(drop=True)
    print(f"✅ Données chargées : {len(df)} lignes, {df['cell'].nunique()} batteries")
    return df

def normalize_per_battery(df):
    """
    Normalise chaque feature entre 0 et 1 PAR batterie.
    Le SOH est ramené entre 0 et 1 (divisé par 100).
    """
    df_norm = df.copy()
    for cell_id in df['cell'].unique():
        mask = df['cell'] == cell_id
        for col in FEATURES:
            col_min = df.loc[mask, col].min()
            col_max = df.loc[mask, col].max()
            denom   = col_max - col_min
            if denom < 1e-8:
                denom = 1.0
            df_norm.loc[mask, col] = (df.loc[mask, col] - col_min) / denom
    
    # SOH : 73-99% → on divise par 100 pour avoir 0.73-0.99
    df_norm[TARGET] = df_norm[TARGET] / 100.0
    return df_norm

class BatteryDataset(Dataset):
    """
    Fenêtre glissante par batterie.
    Entrée  : seq_len cycles de features  → (seq_len, 8)
    Sortie  : pred_len cycles de SOH_C   → (pred_len,)
    """
    def __init__(self, df, seq_len=100, pred_len=50):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.samples  = []

        for cell_id in df['cell'].unique():
            cell_df = df[df['cell'] == cell_id].reset_index(drop=True)
            n = len(cell_df)

            if n < seq_len + pred_len:
                print(f"⚠️  {cell_id} ignorée ({n} cycles < {seq_len+pred_len} requis)")
                continue

            X = cell_df[FEATURES].values.astype(np.float32)
            y = cell_df[TARGET].values.astype(np.float32)

            for i in range(n - seq_len - pred_len + 1):
                self.samples.append((
                    X[i : i + seq_len],
                    y[i + seq_len : i + seq_len + pred_len]
                ))

        print(f"✅ {len(self.samples)} fenêtres créées")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)