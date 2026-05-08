import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, random, math

from models.patchtst import PatchTST
from utils.dataset   import load_data, normalize_per_battery, BatteryDataset

# ═══════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
CSV_PATH   = 'dataset/Battery Data/cleaned_dataset_with_dqdv_peak_final_final.csv'
SEQ_LEN    = 100
PRED_LEN   = 50
BATCH_SIZE = 32
EPOCHS     = 150          # more epochs with early stopping
LR         = 3e-4         # slightly higher for faster convergence
PATIENCE   = 20           # early stopping patience
GRAD_CLIP  = 1.0

TRAIN_CELLS = [
    'B2C00', 'B2C04', 'B2C11', 'B2C13', 'B2C14',
    'B2C16', 'B2C17', 'B2C18', 'B2C19', 'B2C20',
    'B2C22', 'B2C24', 'B2C26', 'B2C27', 'B2C28',
    'B2C29', 'B2C30', 'B2C31', 'B2C32', 'B2C33',
    'B2C34', 'B2C37', 'B2C38', 'B2C43', 'B2C45'
]
TEST_CELLS = ['B2C15', 'B2C35', 'B2C42', 'B2C46', 'B2C08']

# ═══════════════════════════════════════════════════════════════
#  IMPROVED LOSS — Huber + SOH-drop penalty
# ═══════════════════════════════════════════════════════════════
class WeightedSOHLoss(nn.Module):
    """
    Huber loss base + extra penalty on the degradation region (SOH < 0.90).
    Forces the model to learn the critical drop, not just the plateau.
    """
    def __init__(self, delta=0.01, degradation_weight=3.0, threshold=0.90):
        super().__init__()
        self.huber     = nn.HuberLoss(delta=delta)
        self.dw        = degradation_weight
        self.threshold = threshold

    def forward(self, pred, target):
        base_loss = self.huber(pred, target)

        # Extra weight where SOH is falling fast (below threshold)
        mask   = (target < self.threshold).float()
        if mask.sum() > 0:
            deg_loss = self.huber(pred * mask, target * mask)
            return base_loss + self.dw * deg_loss
        return base_loss


# ═══════════════════════════════════════════════════════════════
#  DATA AUGMENTATION — jitter + time-warp (on-the-fly)
# ═══════════════════════════════════════════════════════════════
def augment_batch(x: torch.Tensor, noise_std=0.002) -> torch.Tensor:
    """
    Light Gaussian jitter on input features.
    Keeps the SOH signal meaningful while helping generalisation.
    """
    noise = torch.randn_like(x) * noise_std
    return x + noise


# ═══════════════════════════════════════════════════════════════
#  LOAD & NORMALIZE
# ═══════════════════════════════════════════════════════════════
df      = load_data(CSV_PATH)
df_norm = normalize_per_battery(df)

# ── Diagnostic: check SOH range per test cell ─────────────────
from utils.dataset import TARGET
print("\n📊 Diagnostic SOH range (test cells):")
for cell in TEST_CELLS:
    sub = df[df['cell'] == cell][TARGET]
    print(f"   {cell}: SOH [{sub.min():.2f}% → {sub.max():.2f}%], cycles = {len(sub)}")
print()

train_df = df_norm[df_norm['cell'].isin(TRAIN_CELLS)]
test_df  = df_norm[df_norm['cell'].isin(TEST_CELLS)]

train_dataset = BatteryDataset(train_df, SEQ_LEN, PRED_LEN)
test_dataset  = BatteryDataset(test_df,  SEQ_LEN, PRED_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"✅ Train samples : {len(train_dataset)}")
print(f"✅ Test  samples : {len(test_dataset)}")

# ═══════════════════════════════════════════════════════════════
#  MODEL — slightly larger for better capacity
# ═══════════════════════════════════════════════════════════════
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device : {device}")

model = PatchTST(
    seq_len    = SEQ_LEN,
    pred_len   = PRED_LEN,
    patch_len  = 16,
    stride     = 8,
    n_features = 8,
    d_model    = 256,       # increased from 128 → richer representations
    n_heads    = 8,
    n_layers   = 4,         # one extra transformer layer
    dropout    = 0.15       # slightly more dropout to fight overfitting
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"📊 Paramètres : {n_params:,}\n")

# ═══════════════════════════════════════════════════════════════
#  OPTIMIZER / SCHEDULER / LOSS
# ═══════════════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(          # AdamW > Adam for transformers
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# Cosine annealing: smoothly decays LR, avoids sharp drops
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

criterion = WeightedSOHLoss(delta=0.01, degradation_weight=3.0, threshold=0.90)

# ═══════════════════════════════════════════════════════════════
#  WARMUP WRAPPER (first 10 epochs: linear LR warmup)
# ═══════════════════════════════════════════════════════════════
WARMUP_EPOCHS = 10

def get_lr_scale(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return 1.0   # cosine scheduler takes over after warmup

# ═══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
history      = {'train': [], 'val': [], 'lr': []}
best_val     = float('inf')
no_improve   = 0

print("Epoch | Train Loss | Val Loss  |  LR        | Status")
print("─" * 60)

for epoch in range(EPOCHS):

    # ── LR warmup ──────────────────────────────────────────────
    if epoch < WARMUP_EPOCHS:
        scale = get_lr_scale(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = LR * scale

    # ── Train ──────────────────────────────────────────────────
    model.train()
    train_losses = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # On-the-fly augmentation (only during training)
        x = augment_batch(x, noise_std=0.002)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_losses.append(loss.item())

    # ── Validation ─────────────────────────────────────────────
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_losses.append(criterion(pred, y).item())

    tl = np.mean(train_losses)
    vl = np.mean(val_losses)
    current_lr = optimizer.param_groups[0]['lr']

    history['train'].append(tl)
    history['val'].append(vl)
    history['lr'].append(current_lr)

    # Step cosine scheduler (after warmup)
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

    # ── Checkpoint ─────────────────────────────────────────────
    if vl < best_val:
        best_val   = vl
        no_improve = 0
        torch.save(model.state_dict(), 'best_patchtst.pth')
        status = "✅ saved"
    else:
        no_improve += 1
        status = f"({no_improve}/{PATIENCE})"

    if (epoch + 1) % 10 == 0 or no_improve == 0:
        print(f"  {epoch+1:3d}  |  {tl:.5f}   |  {vl:.5f}  |  {current_lr:.2e}  | {status}")

    # ── Early stopping ─────────────────────────────────────────
    if no_improve >= PATIENCE:
        print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

print(f"\n🏆 Meilleur Val Loss : {best_val:.6f}")

# ═══════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Loss curve
axes[0].plot(history['train'], label='Train Loss', color='steelblue', lw=2)
axes[0].plot(history['val'],   label='Val Loss',   color='tomato',    lw=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title("PatchTST — Loss Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')   # log scale reveals fine convergence

# LR schedule
axes[1].plot(history['lr'], color='orchid', lw=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title("Learning Rate Schedule (Warmup + Cosine)")
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.show()
print("📈 Courbe sauvegardée dans loss_curve.png")