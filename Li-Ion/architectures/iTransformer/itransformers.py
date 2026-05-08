import torch
import torch.nn as nn
import torch.nn.functional as F


class iTransformer(nn.Module):
    """
    iTransformer : l'attention est appliquée sur les FEATURES
    au lieu des timesteps.
    
    Idée : chaque feature (QC, IR, dQdV...) devient un token.
    Le modèle apprend les relations entre features plutôt que
    les relations temporelles.
    
    Référence : Liu et al. 2024 - "iTransformer: Inverted 
    Transformers Are Effective for Time Series Forecasting"
    """
    def __init__(self,
                 seq_len    = 100,
                 pred_len   = 50,
                 n_features = 8,
                 d_model    = 128,
                 n_heads    = 8,
                 n_layers   = 3,
                 dropout    = 0.1):
        super().__init__()

        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.n_features = n_features

        # ── Embedding : chaque feature (seq_len valeurs) → d_model
        # Contrairement à PatchTST qui embed des patches de timesteps,
        # ici on embed chaque feature entière comme un token
        self.feature_embedding = nn.Linear(seq_len, d_model)

        # ── Positional encoding sur les features (pas sur le temps)
        self.pos_emb = nn.Parameter(
            torch.randn(1, n_features, d_model) * 0.02
        )

        # ── Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True    # Pre-LN : plus stable à l'entraînement
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # ── Tête de prédiction : pour chaque feature → pred_len valeurs
        # puis on agrège pour prédire le SOH
        self.head = nn.Sequential(
            nn.Flatten(),                              # (batch, n_features * d_model)
            nn.Linear(n_features * d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, pred_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x : (batch, seq_len, n_features)

        # ── Inversion : (batch, n_features, seq_len)
        # Chaque feature devient un token de dimension seq_len
        x = x.permute(0, 2, 1)

        # ── Embedding de chaque feature : seq_len → d_model
        x = self.feature_embedding(x)    # (batch, n_features, d_model)
        x = x + self.pos_emb             # positional encoding

        # ── Attention sur les features (pas sur le temps)
        x = self.transformer(x)          # (batch, n_features, d_model)

        # ── Prédiction
        x = self.head(x)                 # (batch, pred_len)
        return x