"""
GRU Autoencoder for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/sequence_windows_all.npz
Writes: D:/updated_dataset/models/weights/gru_autoencoder/gru_best.pt
        D:/updated_dataset/models/weights/gru_autoencoder/gru_config.json
        (appends to model_scores_all.csv, model_metrics_all.csv)

GRU variant of LSTM autoencoder — same architecture, GRU cells.
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
    SEQ_ALL_NPZ, RESULTS_DIR, WEIGHTS_DIR, ensure_all_dirs,
)
from model_utils import choose_threshold, compute_metrics, save_scores, save_metrics_row
from torch_utils import make_seq_dataloaders, train_autoencoder, score_autoencoder

MODEL_NAME = "gru_autoencoder"
MAX_EPOCHS = 100
PATIENCE = 10


class GRUAutoencoder(nn.Module):
    """GRU sequence autoencoder.
    Encoder: stacked GRU. Decoder: stacked GRU reconstructing sequence.
    """
    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # x: (B, T, F)
        B, T, _ = x.shape
        _, h_n = self.encoder(x)           # h_n: (num_layers, B, hidden_size)
        context = h_n[-1].unsqueeze(1).repeat(1, T, 1)  # (B, T, hidden_size)
        dec_out, _ = self.decoder(context, h_n)
        return self.output_proj(dec_out)   # (B, T, F)


def train_gru():
    ensure_all_dirs()
    t0 = time.time()
    weights_dir = WEIGHTS_DIR / "gru_autoencoder"

    npz = SEQ_ALL_NPZ
    if not npz.exists():
        print("  GRU: sequence_windows_all.npz not found — SKIP")
        return {"model_name": MODEL_NAME, "status": "NOT_RUN", "reason": "npz not found"}

    tr_loader, va_loader, te_loader, n_time, n_feat, mean, std, wids_test = \
        make_seq_dataloaders(npz, batch_size=128)

    print(f"  GRU: n_time={n_time}, n_feat={n_feat}, "
          f"train={len(tr_loader.dataset)}, val={len(va_loader.dataset)}")

    model = GRUAutoencoder(n_feat, hidden_size=64, num_layers=2, dropout=0.1)
    tr_losses, va_losses, best_ep = train_autoencoder(
        model, tr_loader, va_loader, max_epochs=MAX_EPOCHS,
        lr=1e-3, patience=PATIENCE)

    scores_va, y_va = score_autoencoder(model, va_loader)
    best_thr, _ = choose_threshold(scores_va, y_va)

    scores_te, y_te = score_autoencoder(model, te_loader)
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)
    elapsed = time.time() - t0

    model_path = weights_dir / "gru_best.pt"
    torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                "threshold": best_thr, "n_time": n_time, "n_feat": n_feat}, model_path)
    pd.DataFrame({"train_loss": tr_losses, "val_loss": va_losses}).to_csv(
        weights_dir / "gru_loss_curve.csv", index=False)

    config = {"model_name": MODEL_NAME, "status": "PASS",
              "threshold": best_thr, "best_epoch": best_ep,
              "test_metrics": metrics, "elapsed_s": round(elapsed, 2),
              "artifact_path": str(model_path)}
    with open(weights_dir / "gru_config.json", "w") as f:
        json.dump(config, f, indent=2)

    save_scores(wids_test, y_te, y_pred_te, scores_te, MODEL_NAME, "primary", RESULTS_DIR)
    save_metrics_row(metrics, MODEL_NAME, "primary", "PASS",
                     {"feature_count": n_feat, "train_window_count": len(tr_loader.dataset),
                      "val_window_count": len(va_loader.dataset),
                      "test_window_count": len(te_loader.dataset),
                      "threshold": best_thr, "runtime_seconds": round(elapsed, 2),
                      "artifact_path": str(model_path)},
                     RESULTS_DIR)

    print(f"  GRU: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    return config


def main():
    print("Training GRU Autoencoder...")
    train_gru()
    print("GRU Autoencoder complete.")


if __name__ == "__main__":
    main()
