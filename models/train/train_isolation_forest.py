"""
Isolation Forest anomaly detection for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/windows_all.parquet
Writes: D:/updated_dataset/models/weights/isolation_forest/if_model_*.joblib
        D:/updated_dataset/models/weights/isolation_forest/if_best_config.json
        (appends to model_scores_all.csv, model_metrics_all.csv)

Trains on normal windows only. Hyperparameter sweep on validation F1.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

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

MODEL_NAME = "isolation_forest"
PARAM_GRID = [
    {"n_estimators": 100, "contamination": 0.01, "max_samples": "auto"},
    {"n_estimators": 100, "contamination": 0.05, "max_samples": "auto"},
    {"n_estimators": 300, "contamination": 0.02, "max_samples": "auto"},
    {"n_estimators": 300, "contamination": 0.05, "max_samples": "auto"},
    {"n_estimators": 300, "contamination": 0.10, "max_samples": "auto"},
    {"n_estimators": 500, "contamination": 0.02, "max_samples": "auto"},
]


def train_isolation_forest(window_config: str = "primary"):
    ensure_all_dirs()
    t0 = time.time()

    parquet = WINDOWS_ALL_PARQUET
    if window_config != "primary":
        parquet = parquet.parent / f"windows_all_{window_config}.parquet"

    data = load_all_windows(parquet, FLAT_FEATURE_NAMES)
    X_tr_n = data["X_tr_normal"]
    X_va, y_va = data["X_va"], data["y_va"]
    X_te, y_te = data["X_te"], data["y_te"]

    # Standardize using training normal statistics
    X_tr_s, X_va_s, _, _ = standardize(X_tr_n, X_va)
    _, X_te_s, _, _ = standardize(X_tr_n, X_te)

    print(f"  IF [{window_config}]: train_normal={len(X_tr_n)}, val={len(X_va)}, test={len(X_te)}")

    best_f1 = -1
    best_cfg = None
    best_model = None
    best_thr = None
    all_results = []

    for cfg in PARAM_GRID:
        t_fit = time.time()
        model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            max_samples=cfg["max_samples"],
            random_state=42, n_jobs=-1,
        )
        model.fit(X_tr_s)
        # decision_function: higher = more normal; negate for anomaly score
        scores_va = -model.decision_function(X_va_s)
        thr, sweep = choose_threshold(scores_va, y_va)
        y_pred_va = (scores_va >= thr).astype(int)
        f1 = float(np.mean(y_pred_va == y_va))  # rough accuracy for sweep
        from sklearn.metrics import f1_score
        f1 = f1_score(y_va, y_pred_va, zero_division=0)
        all_results.append({"config": cfg, "val_f1": f1, "threshold": thr})
        print(f"    n_est={cfg['n_estimators']} cont={cfg['contamination']}: "
              f"val_F1={f1:.4f} ({time.time()-t_fit:.1f}s)")
        if f1 > best_f1:
            best_f1 = f1
            best_cfg = cfg
            best_model = model
            best_thr = thr

    # Final evaluation on test
    scores_te = -best_model.decision_function(X_te_s)
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)
    elapsed = time.time() - t0

    # Save model
    model_path = WEIGHTS_DIR / "isolation_forest" / f"if_model_{window_config}.joblib"
    joblib.dump(best_model, model_path)

    config = {
        "model_name": MODEL_NAME, "window_config": window_config,
        "best_params": best_cfg, "threshold": best_thr,
        "val_f1": best_f1, "test_metrics": metrics,
        "sweep": all_results, "elapsed_s": round(elapsed, 2),
        "artifact_path": str(model_path),
    }
    cfg_path = WEIGHTS_DIR / "isolation_forest" / f"if_best_config_{window_config}.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    if window_config == "primary":
        with open(WEIGHTS_DIR / "isolation_forest" / "if_best_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

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

    print(f"  IF [{window_config}]: F1={metrics['f1']:.4f} "
          f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
          f"ROC-AUC={metrics['roc_auc']:.4f}")
    return config


def main():
    print("Training Isolation Forest...")
    train_isolation_forest("primary")
    for suffix in ["30s", "120s"]:
        p = WINDOWS_ALL_PARQUET.parent / f"windows_all_{suffix}.parquet"
        if p.exists():
            train_isolation_forest(suffix)
    print("Isolation Forest complete.")


if __name__ == "__main__":
    main()
