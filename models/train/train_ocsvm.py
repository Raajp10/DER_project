"""
One-Class SVM for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/windows_all.parquet
Writes: D:/updated_dataset/models/weights/ocsvm/ocsvm_model.joblib
        D:/updated_dataset/models/weights/ocsvm/ocsvm_config.json
        (appends to model_scores_all.csv, model_metrics_all.csv)

One-Class SVM may be skipped if too slow or memory-heavy.
If skipped, exact reason and mitigation are documented.
Uses subsampling with documented seed if full dataset is too large.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.svm import OneClassSVM

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

MODEL_NAME = "one_class_svm"
SUBSAMPLE_SEED = 42
MAX_TRAIN_SAMPLES = 5000  # OCSVM is O(n^2); cap to avoid timeout

PARAM_GRID = [
    {"kernel": "rbf", "nu": 0.01, "gamma": "scale"},
    {"kernel": "rbf", "nu": 0.05, "gamma": "scale"},
    {"kernel": "rbf", "nu": 0.10, "gamma": "scale"},
    {"kernel": "rbf", "nu": 0.05, "gamma": "auto"},
]


def train_ocsvm():
    ensure_all_dirs()
    t0 = time.time()
    weights_dir = WEIGHTS_DIR / "ocsvm"

    data = load_all_windows(WINDOWS_ALL_PARQUET, FLAT_FEATURE_NAMES)
    X_tr_n = data["X_tr_normal"]
    X_va, y_va = data["X_va"], data["y_va"]
    X_te, y_te = data["X_te"], data["y_te"]

    # Subsample if too large
    n_train = len(X_tr_n)
    subsampled = False
    if n_train > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(SUBSAMPLE_SEED)
        idx = rng.choice(n_train, MAX_TRAIN_SAMPLES, replace=False)
        X_tr_n = X_tr_n[idx]
        subsampled = True
        print(f"  OCSVM: Subsampled training to {MAX_TRAIN_SAMPLES} "
              f"(from {n_train}, seed={SUBSAMPLE_SEED})")

    X_tr_s, X_va_s, _, _ = standardize(X_tr_n, X_va)
    _, X_te_s, _, _ = standardize(X_tr_n, X_te)

    print(f"  OCSVM: train={len(X_tr_n)}, val={len(X_va)}, test={len(X_te)}")

    best_f1 = -1
    best_cfg = None
    best_model = None
    best_thr = None
    timeout_reason = None

    for cfg in PARAM_GRID:
        try:
            t_fit = time.time()
            model = OneClassSVM(
                kernel=cfg["kernel"], nu=cfg["nu"], gamma=cfg["gamma"]
            )
            model.fit(X_tr_s)
            fit_time = time.time() - t_fit
            if fit_time > 120:
                timeout_reason = f"Fit time {fit_time:.1f}s exceeded 120s limit"
                print(f"  OCSVM TIMEOUT: {timeout_reason}")
                break
            scores_va = -model.decision_function(X_va_s)
            thr, _ = choose_threshold(scores_va, y_va)
            y_pred_va = (scores_va >= thr).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_va, y_pred_va, zero_division=0)
            print(f"    nu={cfg['nu']} gamma={cfg['gamma']}: val_F1={f1:.4f} ({fit_time:.1f}s)")
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = cfg
                best_model = model
                best_thr = thr
        except MemoryError as e:
            timeout_reason = f"MemoryError: {e}"
            print(f"  OCSVM MEMORY ERROR: {timeout_reason}")
            break
        except Exception as e:
            timeout_reason = f"Exception: {e}"
            print(f"  OCSVM ERROR: {timeout_reason}")
            break

    elapsed = time.time() - t0

    if best_model is None:
        reason = timeout_reason or "All configs failed"
        print(f"  OCSVM: NOT_RUN — {reason}")
        config = {"model_name": MODEL_NAME, "status": "NOT_RUN", "reason": reason,
                  "subsampled": subsampled, "subsample_seed": SUBSAMPLE_SEED,
                  "max_train_samples": MAX_TRAIN_SAMPLES, "elapsed_s": round(elapsed, 2)}
        with open(weights_dir / "ocsvm_config.json", "w") as f:
            json.dump(config, f, indent=2)
        save_metrics_row(
            {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
             "roc_auc": float("nan"), "pr_auc": float("nan"),
             "accuracy": float("nan"), "balanced_accuracy": float("nan"),
             "tp": 0, "fp": 0, "tn": 0, "fn": 0,
             "false_positive_count": 0, "false_negative_count": 0},
            MODEL_NAME, "primary", "NOT_RUN",
            {"failure_reason": reason, "feature_count": len(FLAT_FEATURE_NAMES),
             "train_window_count": len(X_tr_n), "val_window_count": len(X_va),
             "test_window_count": len(X_te), "threshold": float("nan"),
             "runtime_seconds": round(elapsed, 2), "artifact_path": ""},
            RESULTS_DIR
        )
        return config

    # Final evaluation on test
    scores_te = -best_model.decision_function(X_te_s)
    y_pred_te = (scores_te >= best_thr).astype(int)
    metrics = compute_metrics(y_te, y_pred_te, scores_te)

    model_path = weights_dir / "ocsvm_model.joblib"
    joblib.dump(best_model, model_path)

    config = {
        "model_name": MODEL_NAME, "status": "PASS",
        "best_params": best_cfg, "threshold": best_thr,
        "subsampled": subsampled, "subsample_seed": SUBSAMPLE_SEED,
        "subsample_n": len(X_tr_n), "val_f1": best_f1,
        "test_metrics": metrics, "elapsed_s": round(elapsed, 2),
        "artifact_path": str(model_path),
    }
    with open(weights_dir / "ocsvm_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    meta_te = data["m_te"]
    wids = meta_te["window_id"].values if "window_id" in meta_te.columns else np.arange(len(y_te))
    save_scores(wids, y_te, y_pred_te, scores_te, MODEL_NAME, "primary", RESULTS_DIR)
    save_metrics_row(metrics, MODEL_NAME, "primary", "PASS",
                     {"feature_count": len(FLAT_FEATURE_NAMES),
                      "train_window_count": len(X_tr_n),
                      "val_window_count": len(X_va),
                      "test_window_count": len(X_te),
                      "threshold": best_thr,
                      "runtime_seconds": round(elapsed, 2),
                      "artifact_path": str(model_path)},
                     RESULTS_DIR)

    print(f"  OCSVM: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    return config


def main():
    print("Training One-Class SVM...")
    train_ocsvm()
    print("One-Class SVM complete.")


if __name__ == "__main__":
    main()
