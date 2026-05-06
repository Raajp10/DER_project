"""
MLP Autoencoder for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/windows_all.parquet
Writes: D:/updated_dataset/models/weights/mlp_autoencoder/mlp_best_*.pt
        D:/updated_dataset/models/weights/mlp_autoencoder/mlp_config.json
        D:/updated_dataset/models/weights/mlp_autoencoder/mlp_loss_curve.csv
        D:/updated_dataset/models/figures/training_loss_curves.png (partial)
        (appends to model_scores_all.csv, model_metrics_all.csv)

Trains on flat feature windows (220 features).
Uses early stopping. Secondary sweep on 30s and 120s windows.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    WINDOWS_ALL_PARQUET, RESULTS_DIR, WEIGHTS_DIR, ensure_all_dirs,
)
from feature_config import FLAT_FEATURE_NAMES
from model_utils import (
    load_all_windows, choose_threshold, compute_metrics,
    save_scores, save_metrics_row, standardize,
)

MODEL_NAME = "mlp_autoencoder"

CONFIGS = [
    {"hidden": [256, 128, 64], "dropout": 0.1, "lr": 1e-3, "batch_size": 256},
    {"hidden": [512, 256, 128], "dropout": 0.1, "lr": 1e-3, "batch_size": 256},
    {"hidden": [256, 128, 64, 32], "dropout": 0.2, "lr": 5e-4, "batch_size": 128},
]
MAX_EPOCHS = 100
PATIENCE = 10


class MLPAutoencoder(nn.Module):
    """MLP Autoencoder: encoder-bottleneck-decoder with ReLU activations."""
    def __init__(self, input_dim: int, hidden_sizes: list, dropout: float = 0.1):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_sizes:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        # Decoder (mirror)
        dec_layers = []
        for h in reversed(hidden_sizes[:-1]):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _make_flat_loaders(X_tr_n, X_va, y_va, X_te, y_te, batch_size):
    tr_ds = TensorDataset(torch.from_numpy(X_tr_n))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.float32)))
    te_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te.astype(np.float32)))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader


def train_mlp(window_config: str = "primary"):
    ensure_all_dirs()
    t0 = time.time()
    weights_dir = WEIGHTS_DIR / "mlp_autoencoder"

    parquet = WINDOWS_ALL_PARQUET
    if window_config != "primary":
        parquet = parquet.parent / f"windows_all_{window_config}.parquet"

    data = load_all_windows(parquet, FLAT_FEATURE_NAMES)
    X_tr_n = data["X_tr_normal"]
    X_va, y_va = data["X_va"], data["y_va"]
    X_te, y_te = data["X_te"], data["y_te"]
    input_dim = X_tr_n.shape[1]

    X_tr_s, X_va_s, mean, std = standardize(X_tr_n, X_va)
    _, X_te_s, _, _ = standardize(X_tr_n, X_te)

    print(f"  MLP [{window_config}]: input_dim={input_dim}, train={len(X_tr_n)}")

    best_val_loss = float("inf")
    best_cfg = None
    best_model = None
    best_train_losses = []
    best_val_losses = []

    for cfg in CONFIGS:
        model = MLPAutoencoder(input_dim, cfg["hidden"], cfg["dropout"])
        tr_loader, va_loader, _ = _make_flat_loaders(
            X_tr_s, X_va_s, y_va, X_te_s, y_te, cfg["batch_size"])
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        stopper_best = float("inf")
        stopper_count = 0
        tr_losses, va_losses = [], []
        best_state = None

        for epoch in range(MAX_EPOCHS):
            model.train()
            batch_l = []
            for (x,) in tr_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x), x)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batch_l.append(loss.item())
            tr_l = float(np.mean(batch_l))
            model.eval()
            va_l_list = []
            with torch.no_grad():
                for (x, _) in va_loader:
                    va_l_list.append(loss_fn(model(x), x).item())
            va_l = float(np.mean(va_l_list))
            tr_losses.append(tr_l)
            va_losses.append(va_l)
            if va_l < stopper_best - 1e-6:
                stopper_best = va_l
                stopper_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stopper_count += 1
            if stopper_count >= PATIENCE:
                break
        if best_state:
            model.load_state_dict(best_state)
        print(f"    hidden={cfg['hidden']} dr={cfg['dropout']} "
              f"lr={cfg['lr']}: final_val={stopper_best:.6f}")
        if stopper_best < best_val_loss:
            best_val_loss = stopper_best
            best_cfg = cfg
            best_model = model
            best_train_losses = tr_losses
            best_val_losses = va_losses

    # Score validation for threshold
    best_model.eval()
    scores_va_list = []
    with torch.no_grad():
        for (x, _) in _make_flat_loaders(
                X_tr_s, X_va_s, y_va, X_te_s, y_te, 256)[1]:
            x_hat = best_model(x)
            err = nn.functional.mse_loss(x_hat, x, reduction="none").mean(dim=1)
            scores_va_list.append(err.numpy())
    scores_va = np.concatenate(scores_va_list)
    best_thr, _ = choose_threshold(scores_va, y_va)

    # Score test
    scores_te_list = []
    with torch.no_grad():
        for (x, _) in _make_flat_loaders(
                X_tr_s, X_va_s, y_va, X_te_s, y_te, 256)[2]:
            x_hat = best_model(x)
            err = nn.functional.mse_loss(x_hat, x, reduction="none").mean(dim=1)
            scores_te_list.append(err.numpy())
    scores_te = np.concatenate(scores_te_list)
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)
    elapsed = time.time() - t0

    # Save
    model_path = weights_dir / f"mlp_best_{window_config}.pt"
    torch.save({"model_state": best_model.state_dict(), "mean": mean, "std": std,
                "threshold": best_thr, "config": best_cfg}, model_path)

    # Save loss curve
    pd.DataFrame({"train_loss": best_train_losses, "val_loss": best_val_losses}).to_csv(
        weights_dir / f"mlp_loss_curve_{window_config}.csv", index=False)

    config = {
        "model_name": MODEL_NAME, "window_config": window_config,
        "best_config": best_cfg, "threshold": best_thr,
        "test_metrics": metrics, "elapsed_s": round(elapsed, 2),
        "artifact_path": str(model_path),
    }
    with open(weights_dir / f"mlp_config_{window_config}.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    if window_config == "primary":
        with open(weights_dir / "mlp_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    meta_te = data["m_te"]
    wids = meta_te["window_id"].values if "window_id" in meta_te.columns else np.arange(len(y_te))
    if window_config == "primary":
        save_scores(wids, y_te, y_pred_te, scores_te, MODEL_NAME, window_config, RESULTS_DIR)
    save_metrics_row(metrics, MODEL_NAME, window_config, "PASS",
                     {"feature_count": input_dim,
                      "train_window_count": len(X_tr_n),
                      "val_window_count": len(X_va),
                      "test_window_count": len(X_te),
                      "threshold": best_thr,
                      "runtime_seconds": round(elapsed, 2),
                      "artifact_path": str(model_path)},
                     RESULTS_DIR)

    print(f"  MLP [{window_config}]: F1={metrics['f1']:.4f} "
          f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
          f"ROC-AUC={metrics['roc_auc']:.4f}")
    return config


def main():
    print("Training MLP Autoencoder...")
    train_mlp("primary")
    for suffix in ["30s", "120s"]:
        p = WINDOWS_ALL_PARQUET.parent / f"windows_all_{suffix}.parquet"
        if p.exists():
            train_mlp(suffix)
    print("MLP Autoencoder complete.")


if __name__ == "__main__":
    main()
