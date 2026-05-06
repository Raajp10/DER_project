"""
CNN Autoencoder (1D) for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/sequence_windows_all.npz
Writes: D:/updated_dataset/models/weights/cnn_autoencoder/cnn_best.pt
        D:/updated_dataset/models/weights/cnn_autoencoder/cnn_config.json
        (appends to model_scores_all.csv, model_metrics_all.csv)

Uses 1D convolutional encoder over time dimension.
Trains on normal sequence windows only.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    SEQ_ALL_NPZ, RESULTS_DIR, WEIGHTS_DIR, WINDOWS_ALL_PARQUET, ensure_all_dirs,
)
from feature_config import N_RAW_FEATURES
from model_utils import choose_threshold, compute_metrics, save_scores, save_metrics_row
from torch_utils import make_seq_dataloaders, train_autoencoder, score_autoencoder

MODEL_NAME = "cnn_autoencoder"
MAX_EPOCHS = 100
PATIENCE = 10


class CNN1DAutoencoder(nn.Module):
    """1D CNN Autoencoder.
    Encoder: Conv1d layers reduce time dimension.
    Decoder: ConvTranspose1d layers reconstruct.
    Input/output: (batch, features, time) — we transpose inside forward.
    """
    def __init__(self, n_features: int, n_time: int):
        super().__init__()
        # Encoder: input (B, n_features, n_time)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, n_features, kernel_size=5, padding=2),
        )

    def forward(self, x):
        # x: (B, T, F) → transpose to (B, F, T)
        x = x.transpose(1, 2)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat.transpose(1, 2)  # back to (B, T, F)


def train_cnn():
    ensure_all_dirs()
    t0 = time.time()
    weights_dir = WEIGHTS_DIR / "cnn_autoencoder"

    npz = SEQ_ALL_NPZ
    if not npz.exists():
        print("  CNN: sequence_windows_all.npz not found — SKIP")
        return {"model_name": MODEL_NAME, "status": "NOT_RUN", "reason": "npz not found"}

    tr_loader, va_loader, te_loader, n_time, n_feat, mean, std, wids_test = \
        make_seq_dataloaders(npz, batch_size=128)

    print(f"  CNN: n_time={n_time}, n_feat={n_feat}, "
          f"train={len(tr_loader.dataset)}, val={len(va_loader.dataset)}")

    model = CNN1DAutoencoder(n_feat, n_time)
    tr_losses, va_losses, best_ep = train_autoencoder(
        model, tr_loader, va_loader, max_epochs=MAX_EPOCHS,
        lr=1e-3, patience=PATIENCE)

    # Score validation for threshold
    scores_va, y_va = score_autoencoder(model, va_loader)
    best_thr, _ = choose_threshold(scores_va, y_va)

    # Score test
    scores_te, y_te = score_autoencoder(model, te_loader)
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)
    elapsed = time.time() - t0

    # Save
    model_path = weights_dir / "cnn_best.pt"
    torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                "threshold": best_thr, "n_time": n_time, "n_feat": n_feat}, model_path)
    pd.DataFrame({"train_loss": tr_losses, "val_loss": va_losses}).to_csv(
        weights_dir / "cnn_loss_curve.csv", index=False)

    config = {"model_name": MODEL_NAME, "status": "PASS",
              "threshold": best_thr, "best_epoch": best_ep,
              "test_metrics": metrics, "elapsed_s": round(elapsed, 2),
              "artifact_path": str(model_path)}
    with open(weights_dir / "cnn_config.json", "w") as f:
        json.dump(config, f, indent=2)

    save_scores(wids_test, y_te, y_pred_te, scores_te, MODEL_NAME, "primary", RESULTS_DIR)
    save_metrics_row(metrics, MODEL_NAME, "primary", "PASS",
                     {"feature_count": n_feat, "train_window_count": len(tr_loader.dataset),
                      "val_window_count": len(va_loader.dataset),
                      "test_window_count": len(te_loader.dataset),
                      "threshold": best_thr, "runtime_seconds": round(elapsed, 2),
                      "artifact_path": str(model_path)},
                     RESULTS_DIR)

    print(f"  CNN: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    return config


def main():
    print("Training CNN Autoencoder...")
    train_cnn()
    print("CNN Autoencoder complete.")


if __name__ == "__main__":
    main()
