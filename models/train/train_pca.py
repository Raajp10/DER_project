"""
PCA Reconstruction anomaly detection for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/windows_all.parquet
Writes: D:/updated_dataset/models/weights/pca/pca_model_*.joblib
        D:/updated_dataset/models/weights/pca/pca_best_config.json
        (appends to model_scores_all.csv, model_metrics_all.csv)

Trains PCA on normal training windows.
Anomaly score = reconstruction error (MSE between original and projected-back).
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.decomposition import PCA

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

MODEL_NAME = "pca_reconstruction"
# Variance retained sweep (0-1) + fixed component counts
VARIANCE_CONFIGS = [0.90, 0.95, 0.99]
FIXED_CONFIGS = [5, 10, 20]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def pca_reconstruction_score(X: np.ndarray, pca: PCA) -> np.ndarray:
    """MSE reconstruction error through PCA projection."""
    X_proj = pca.inverse_transform(pca.transform(X))
    return np.mean((X - X_proj) ** 2, axis=1)


def train_pca(window_config: str = "primary"):
    ensure_all_dirs()
    t0 = time.time()
    weights_dir = WEIGHTS_DIR / "pca"

    parquet = WINDOWS_ALL_PARQUET
    if window_config != "primary":
        parquet = parquet.parent / f"windows_all_{window_config}.parquet"

    data = load_all_windows(parquet, FLAT_FEATURE_NAMES)
    X_tr_n = data["X_tr_normal"]
    X_va, y_va = data["X_va"], data["y_va"]
    X_te, y_te = data["X_te"], data["y_te"]

    X_tr_s, X_va_s, mean, std = standardize(X_tr_n, X_va)
    _, X_te_s, _, _ = standardize(X_tr_n, X_te)

    print(f"  PCA [{window_config}]: train={len(X_tr_n)}, val={len(X_va)}, test={len(X_te)}")

    best_f1 = -1
    best_pca = None
    best_thr = None
    best_n = None
    sweep_results = []

    # Variance-based configs
    for var in VARIANCE_CONFIGS:
        pca = PCA(n_components=var, random_state=42)
        pca.fit(X_tr_s)
        n_comp = pca.n_components_
        scores_va = pca_reconstruction_score(X_va_s, pca)
        thr, _ = choose_threshold(scores_va, y_va)
        y_pred_va = (scores_va >= thr).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_va, y_pred_va, zero_division=0)
        sweep_results.append({"type": "variance", "param": var, "n_components": n_comp, "val_f1": f1})
        print(f"    var={var}: n_components={n_comp}, val_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; best_pca = pca; best_thr = thr; best_n = n_comp

    # Fixed component configs
    max_comp = min(X_tr_s.shape[1], X_tr_s.shape[0] - 1)
    for n in FIXED_CONFIGS:
        if n >= max_comp:
            continue
        pca = PCA(n_components=n, random_state=42)
        pca.fit(X_tr_s)
        scores_va = pca_reconstruction_score(X_va_s, pca)
        thr, _ = choose_threshold(scores_va, y_va)
        y_pred_va = (scores_va >= thr).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_va, y_pred_va, zero_division=0)
        sweep_results.append({"type": "fixed", "param": n, "n_components": n, "val_f1": f1})
        print(f"    n={n}: val_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; best_pca = pca; best_thr = thr; best_n = n

    # Final test evaluation
    scores_te = pca_reconstruction_score(X_te_s, best_pca)
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)
    elapsed = time.time() - t0

    # Save model
    model_path = weights_dir / f"pca_model_{window_config}.joblib"
    joblib.dump({"pca": best_pca, "mean": mean, "std": std, "threshold": best_thr}, model_path)

    config = {
        "model_name": MODEL_NAME, "window_config": window_config,
        "best_n_components": best_n, "threshold": best_thr,
        "val_f1": best_f1, "sweep": sweep_results,
        "test_metrics": metrics, "elapsed_s": round(elapsed, 2),
        "artifact_path": str(model_path),
    }
    cfg_path = weights_dir / f"pca_best_config_{window_config}.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2, default=_json_default)
    if window_config == "primary":
        with open(weights_dir / "pca_best_config.json", "w") as f:
            json.dump(config, f, indent=2, default=_json_default)

    meta_te = data["m_te"]
    wids = meta_te["window_id"].values if "window_id" in meta_te.columns else np.arange(len(y_te))
    if window_config == "primary":
        save_scores(wids, y_te, y_pred_te, scores_te, MODEL_NAME, window_config, RESULTS_DIR)
    save_metrics_row(metrics, MODEL_NAME, window_config, "PASS",
                     {"feature_count": len(FLAT_FEATURE_NAMES),
                      "train_window_count": len(X_tr_n),
                      "val_window_count": len(X_va),
                      "test_window_count": len(X_te),
                      "threshold": best_thr,
                      "runtime_seconds": round(elapsed, 2),
                      "artifact_path": str(cfg_path)},
                     RESULTS_DIR)

    print(f"  PCA [{window_config}]: F1={metrics['f1']:.4f} "
          f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
          f"ROC-AUC={metrics['roc_auc']:.4f}")
    return config


def main():
    print("Training PCA reconstruction...")
    train_pca("primary")
    for suffix in ["30s", "120s"]:
        p = WINDOWS_ALL_PARQUET.parent / f"windows_all_{suffix}.parquet"
        if p.exists():
            train_pca(suffix)
    print("PCA reconstruction complete.")


if __name__ == "__main__":
    main()
