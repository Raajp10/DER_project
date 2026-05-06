"""
Transformer Autoencoder for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/sequence_windows_all.npz
Writes: D:/updated_dataset/models/weights/transformer_autoencoder/transformer_best.pt
        D:/updated_dataset/models/weights/transformer_autoencoder/transformer_config.json
        (appends to model_scores_all.csv, model_metrics_all.csv)

Encoder: positional encoding + TransformerEncoder.
Decoder: TransformerDecoder reconstructing sequence from pooled latent.
"""
import sys
import json
import math
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

MODEL_NAME = "transformer_autoencoder"
MAX_EPOCHS = 100
PATIENCE = 10


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerAutoencoder(nn.Module):
    """Transformer-based sequence autoencoder.

    Projects input to d_model, encodes with TransformerEncoder,
    decodes with TransformerDecoder using learned query tokens.
    """
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(d_model, n_features)
        self.d_model = d_model

    def forward(self, x):
        # x: (B, T, F)
        x_proj = self.pos_enc(self.input_proj(x))        # (B, T, d_model)
        memory = self.encoder(x_proj)                    # (B, T, d_model)
        # Use pos-encoded input as query (reconstruction target)
        dec_out = self.decoder(x_proj, memory)           # (B, T, d_model)
        return self.output_proj(dec_out)                 # (B, T, F)


def train_transformer():
    ensure_all_dirs()
    t0 = time.time()
    weights_dir = WEIGHTS_DIR / "transformer_autoencoder"

    npz = SEQ_ALL_NPZ
    if not npz.exists():
        print("  Transformer: sequence_windows_all.npz not found — SKIP")
        return {"model_name": MODEL_NAME, "status": "NOT_RUN", "reason": "npz not found"}

    tr_loader, va_loader, te_loader, n_time, n_feat, mean, std, wids_test = \
        make_seq_dataloaders(npz, batch_size=128)

    print(f"  Transformer: n_time={n_time}, n_feat={n_feat}, "
          f"train={len(tr_loader.dataset)}, val={len(va_loader.dataset)}")

    model = TransformerAutoencoder(
        n_features=n_feat, d_model=64, nhead=4,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=256, dropout=0.1)
    tr_losses, va_losses, best_ep = train_autoencoder(
        model, tr_loader, va_loader, max_epochs=MAX_EPOCHS,
        lr=5e-4, patience=PATIENCE)

    scores_va, y_va = score_autoencoder(model, va_loader)
    best_thr, _ = choose_threshold(scores_va, y_va)

    scores_te, y_te = score_autoencoder(model, te_loader)
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)
    elapsed = time.time() - t0

    model_path = weights_dir / "transformer_best.pt"
    torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                "threshold": best_thr, "n_time": n_time, "n_feat": n_feat}, model_path)
    pd.DataFrame({"train_loss": tr_losses, "val_loss": va_losses}).to_csv(
        weights_dir / "transformer_loss_curve.csv", index=False)

    config = {"model_name": MODEL_NAME, "status": "PASS",
              "threshold": best_thr, "best_epoch": best_ep,
              "test_metrics": metrics, "elapsed_s": round(elapsed, 2),
              "artifact_path": str(model_path)}
    with open(weights_dir / "transformer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    save_scores(wids_test, y_te, y_pred_te, scores_te, MODEL_NAME, "primary", RESULTS_DIR)
    save_metrics_row(metrics, MODEL_NAME, "primary", "PASS",
                     {"feature_count": n_feat, "train_window_count": len(tr_loader.dataset),
                      "val_window_count": len(va_loader.dataset),
                      "test_window_count": len(te_loader.dataset),
                      "threshold": best_thr, "runtime_seconds": round(elapsed, 2),
                      "artifact_path": str(model_path)},
                     RESULTS_DIR)

    print(f"  Transformer: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    return config


def main():
    print("Training Transformer Autoencoder...")
    train_transformer()
    print("Transformer Autoencoder complete.")


if __name__ == "__main__":
    main()
