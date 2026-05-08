import torch
import torch.nn as nn

# ════════════════════════════════════════════════════════
# 1. LSTM
# ════════════════════════════════════════════════════════
class LSTMModel(nn.Module):
    def __init__(self, n_features=8, hidden_size=128, n_layers=3,
                 pred_len=50, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout,
            batch_first = True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x : (batch, seq_len, n_features)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        out = out[:, -1, :]            # dernier timestep → (batch, hidden)
        return self.head(out)          # (batch, pred_len)


# ════════════════════════════════════════════════════════
# 2. GRU
# ════════════════════════════════════════════════════════
class GRUModel(nn.Module):
    def __init__(self, n_features=8, hidden_size=128, n_layers=3,
                 pred_len=50, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout,
            batch_first = True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.head(out)


# ════════════════════════════════════════════════════════
# 3. GRU-LSTM (GRU en entrée → LSTM en sortie)
# ════════════════════════════════════════════════════════
class GRULSTMModel(nn.Module):
    def __init__(self, n_features=8, hidden_size=128, n_layers=2,
                 pred_len=50, dropout=0.1):
        super().__init__()
        # Bloc GRU : extrait les features locales
        self.gru = nn.GRU(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout,
            batch_first = True
        )
        # Bloc LSTM : capte les dépendances longue-terme
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout,
            batch_first = True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _  = self.gru(x)          # (batch, seq_len, hidden)
        out, _  = self.lstm(out)        # (batch, seq_len, hidden)
        out = out[:, -1, :]
        return self.head(out)


# ════════════════════════════════════════════════════════
# 4. Conv1D-LSTM
# ════════════════════════════════════════════════════════
class Conv1DLSTMModel(nn.Module):
    def __init__(self, n_features=8, hidden_size=128, n_layers=2,
                 pred_len=50, dropout=0.1):
        super().__init__()
        # Bloc Conv1D : détecte les patterns locaux
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Bloc LSTM : séquence temporelle sur les features extraites
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            dropout     = dropout,
            batch_first = True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Conv1D attend (batch, channels, seq_len)
        x = x.permute(0, 2, 1)         # (batch, n_features, seq_len)
        x = self.conv(x)               # (batch, hidden, seq_len)
        x = x.permute(0, 2, 1)         # (batch, seq_len, hidden)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out)
    