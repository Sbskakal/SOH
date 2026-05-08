import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Découpe la série temporelle en patches qui se chevauchent
    et les projette dans un espace vectoriel de dimension d_model.
    """
    def __init__(self, seq_len, patch_len, stride, n_features, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Chaque patch (patch_len × n_features) → vecteur d_model
        self.projection = nn.Linear(patch_len * n_features, d_model)

        # Position apprise (pas fixe comme dans le Transformer original)
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )

    def forward(self, x):
        # x : (batch, seq_len, n_features)
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end   = start + self.patch_len
            p = x[:, start:end, :]           # (batch, patch_len, n_features)
            p = p.reshape(p.size(0), -1)      # (batch, patch_len * n_features)
            patches.append(p)

        x = torch.stack(patches, dim=1)       # (batch, n_patches, patch_len*n_features)
        x = self.projection(x)                # (batch, n_patches, d_model)
        x = x + self.pos_emb
        return x


class PatchTST(nn.Module):
    def __init__(self,
                 seq_len    = 100,
                 pred_len   = 50,
                 patch_len  = 16,
                 stride     = 8,
                 n_features = 8,
                 d_model    = 128,
                 n_heads    = 8,
                 n_layers   = 3,
                 dropout    = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            seq_len, patch_len, stride, n_features, d_model
        )
        n_patches = self.patch_embed.n_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,   # 512
            dropout         = dropout,
            batch_first     = True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Tête de régression → pred_len valeurs SOH entre 0 et 1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patches * d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len),
            nn.Sigmoid()   # SOH normalisé entre 0 et 1
        )

    def forward(self, x):
        # x : (batch, seq_len, n_features)
        x = self.patch_embed(x)    # (batch, n_patches, d_model)
        x = self.transformer(x)    # (batch, n_patches, d_model)
        x = self.head(x)           # (batch, pred_len)
        return x