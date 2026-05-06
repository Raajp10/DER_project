"""
TCN (Temporal Convolutional Network) Autoencoder for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/sequence_windows_all.npz
Writes: D:/updated_dataset/models/weights/tcn_autoencoder/tcn_best.pt
        D:/updated_dataset/models/weights/tcn_autoencoder/tcn_config.json
        (appends to model_scores_all.csv, model_metrics_all.csv)

Causal dilated 1D convolutions with residual connections.
Encoder: stacked TCN blocks reducing time resolution.
Decoder: mirrored ConvTranspose1d blocks reconstructing sequence.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    SEQ_ALL_NPZ, RESULTS_DIR, WEIGHTS_DIR, ensure_all_dirs,
)
from model_utils import choose_threshold, compute_metrics, save_scores, save_metrics_row
from torch_utils import make_seq_dataloaders, train_autoencoder, score_autoencoder

MODEL_NAME = "tcn_autoencoder"
MAX_EPOCHS = 100
PATIENCE = 10


class CausalConv1d(nn.Module):
    """Causal convolution with dilation (no future leakage)."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        # Remove right padding to maintain causal property
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """TCN residual block: 2 causal dilated convolutions + skip connection."""
    def __init__(self, n_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(n_channels, n_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_channels, n_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(n_channels)
        self.norm2 = nn.BatchNorm1d(n_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dropout(F.relu(self.norm1(self.conv1(x))))
        out = self.dropout(F.relu(self.norm2(self.conv2(out))))
        return F.relu(out + residual)


class TCNAutoencoder(nn.Module):
    """TCN-based sequence autoencoder using causal dilated convolutions.

    Encoder: project + stacked TCN blocks (operates on full time axis).
    Decoder: mirrored ConvTranspose1d blocks.
    Input/output: (B, T, F) — internally (B, F, T).
    """
    def __init__(self, n_features: int, channels: int = 64,
                 kernel_size: int = 3, n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, channels, kernel_size=1)

        # Encoder: TCN blocks with exponentially growing dilation
        enc_blocks = []
        for i in range(n_blocks):
            dilation = 2 ** i
            enc_blocks.append(TCNBlock(channels, kernel_size, dilation, dropout))
        self.encoder = nn.Sequential(*enc_blocks)

        # Bottleneck channel compression
        self.bottleneck_down = nn.Conv1d(channels, channels // 2, kernel_size=1)
        self.bottleneck_up = nn.Conv1d(channels // 2, channels, kernel_size=1)

        # Decoder: mirrored TCN blocks (non-causal for reconstruction)
        dec_blocks = []
        for i in range(n_blocks - 1, -1, -1):
            dilation = 2 ** i
            dec_blocks.append(TCNBlock(channels, kernel_size, dilation, dropout))
        self.decoder = nn.Sequential(*dec_blocks)

        self.output_proj = nn.Conv1d(channels, n_features, kernel_size=1)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        z = self.encoder(x)
        z = F.relu(self.bottleneck_down(z))
        z = F.relu(self.bottleneck_up(z))
        out = self.decoder(z)
        out = self.output_proj(out)
        return out.transpose(1, 2)  # (B, T, F)


def train_tcn():
    ensure_all_dirs()
    t0 = time.time()
    weights_dir = WEIGHTS_DIR / "tcn_autoencoder"

    npz = SEQ_ALL_NPZ
    if not npz.exists():
        print("  TCN: sequence_windows_all.npz not found — SKIP")
        return {"model_name": MODEL_NAME, "status": "NOT_RUN", "reason": "npz not found"}

    tr_loader, va_loader, te_loader, n_time, n_feat, mean, std, wids_test = \
        make_seq_dataloaders(npz, batch_size=128)

    print(f"  TCN: n_time={n_time}, n_feat={n_feat}, "
          f"train={len(tr_loader.dataset)}, val={len(va_loader.dataset)}")

    model = TCNAutoencoder(n_feat, channels=64, kernel_size=3, n_blocks=4, dropout=0.1)
    tr_losses, va_losses, best_ep = train_autoencoder(
        model, tr_loader, va_loader, max_epochs=MAX_EPOCHS,
        lr=1e-3, patience=PATIENCE)

    scores_va, y_va = score_autoencoder(model, va_loader)
    best_thr, _ = choose_threshold(scores_va, y_va)

    scores_te, y_te = score_autoencoder(model, te_loader)
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)
    elapsed = time.time() - t0

    model_path = weights_dir / "tcn_best.pt"
    torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                "threshold": best_thr, "n_time": n_time, "n_feat": n_feat}, model_path)
    pd.DataFrame({"train_loss": tr_losses, "val_loss": va_losses}).to_csv(
        weights_dir / "tcn_loss_curve.csv", index=False)

    config = {"model_name": MODEL_NAME, "status": "PASS",
              "threshold": best_thr, "best_epoch": best_ep,
              "test_metrics": metrics, "elapsed_s": round(elapsed, 2),
              "artifact_path": str(model_path)}
    with open(weights_dir / "tcn_config.json", "w") as f:
        json.dump(config, f, indent=2)

    save_scores(wids_test, y_te, y_pred_te, scores_te, MODEL_NAME, "primary", RESULTS_DIR)
    save_metrics_row(metrics, MODEL_NAME, "primary", "PASS",
                     {"feature_count": n_feat, "train_window_count": len(tr_loader.dataset),
                      "val_window_count": len(va_loader.dataset),
                      "test_window_count": len(te_loader.dataset),
                      "threshold": best_thr, "runtime_seconds": round(elapsed, 2),
                      "artifact_path": str(model_path)},
                     RESULTS_DIR)

    print(f"  TCN: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    return config


def main():
    print("Training TCN Autoencoder...")
    train_tcn()
    print("TCN Autoencoder complete.")


if __name__ == "__main__":
    main()
