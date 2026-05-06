"""
Threshold baseline model for Phase 1 anomaly detection.

Reads:  D:/updated_dataset/models/windows/windows_all.parquet
Writes: D:/updated_dataset/models/weights/threshold/threshold_config.json
        D:/updated_dataset/models/results/threshold_sweep.csv
        (appends to model_scores_all.csv and model_metrics_all.csv)

Trains on normal training windows only.
Scores based on robust z-score of per-window features.
Threshold chosen on validation split only.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    WINDOWS_ALL_PARQUET, RESULTS_DIR, WEIGHTS_DIR, ensure_all_dirs,
)
from feature_config import FLAT_FEATURE_NAMES
from model_utils import (
    load_all_windows, choose_threshold, compute_metrics,
    save_scores, save_metrics_row,
)

MODEL_NAME = "threshold_baseline"
PERCENTILES = [90, 95, 97.5, 99, 99.5, 99.9]


def robust_zscore_anomaly_score(X: np.ndarray, median: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """Compute max robust z-score across all features.

    score = max_i( |X_i - median_i| / (1.4826 * MAD_i + eps) )
    High score = anomalous.
    """
    eps = 1e-8
    z = np.abs(X - median) / (1.4826 * mad + eps)
    return z.max(axis=1)


def train_threshold(window_config: str = "primary"):
    ensure_all_dirs()
    t0 = time.time()

    parquet = WINDOWS_ALL_PARQUET
    if window_config != "primary":
        suffix = window_config
        parquet = parquet.parent / f"windows_all_{suffix}.parquet"

    data = load_all_windows(parquet, FLAT_FEATURE_NAMES)
    X_tr_n = data["X_tr_normal"]  # normal training only
    X_va, y_va = data["X_va"], data["y_va"]
    X_te, y_te = data["X_te"], data["y_te"]

    print(f"  Threshold [{window_config}]: train_normal={len(X_tr_n)}, "
          f"val={len(X_va)}, test={len(X_te)}")

    # Fit: compute per-feature median and MAD on normal training windows
    median = np.median(X_tr_n, axis=0)
    mad = np.median(np.abs(X_tr_n - median), axis=0)

    # Score validation and test
    scores_va = robust_zscore_anomaly_score(X_va, median, mad)
    scores_te = robust_zscore_anomaly_score(X_te, median, mad)

    # Choose threshold on validation split
    best_thr, sweep = choose_threshold(scores_va, y_va, PERCENTILES)

    # Evaluate on test split
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)

    elapsed = time.time() - t0

    # Save model config
    config = {
        "model_name": MODEL_NAME,
        "window_config": window_config,
        "threshold": best_thr,
        "median": median.tolist(),
        "mad": mad.tolist(),
        "percentile_sweep": sweep,
        "test_metrics": metrics,
        "elapsed_s": round(elapsed, 2),
    }
    cfg_path = WEIGHTS_DIR / "threshold" / f"threshold_config_{window_config}.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    # Also save primary as the canonical name
    if window_config == "primary":
        with open(WEIGHTS_DIR / "threshold" / "threshold_config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Save sweep CSV
    sweep_df = pd.DataFrame(sweep)
    sweep_df["window_config"] = window_config
    sweep_path = RESULTS_DIR / "threshold_sweep.csv"
    mode = "a" if sweep_path.exists() else "w"
    sweep_df.to_csv(sweep_path, mode=mode, header=not sweep_path.exists(), index=False)

    # Save scores and metrics
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

    print(f"  Threshold [{window_config}]: F1={metrics['f1']:.4f} "
          f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
          f"ROC-AUC={metrics['roc_auc']:.4f} thr={best_thr:.4f}")
    return config


def main():
    print("Training threshold baseline...")
    # Primary
    train_threshold("primary")
    # Secondary sweep
    for suffix in ["30s", "120s"]:
        p = WINDOWS_ALL_PARQUET.parent / f"windows_all_{suffix}.parquet"
        if p.exists():
            train_threshold(suffix)
    print("Threshold baseline complete.")


if __name__ == "__main__":
    main()
