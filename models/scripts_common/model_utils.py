"""Shared utilities for all Phase 1 model training scripts.

Provides data loading, scoring, threshold selection, and metric computation.
All functions operate on pre-built window parquet files.
"""
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix,
)


def load_windows(parquet_path: Path, feature_names: list,
                 split: Optional[str] = None) -> tuple:
    """Load flat feature windows from parquet.

    Returns (X, y, meta_df) where:
      X: np.ndarray (n, n_flat_features) float32
      y: np.ndarray (n,) int
      meta: pd.DataFrame with window metadata
    """
    df = pd.read_parquet(parquet_path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    available = [c for c in feature_names if c in df.columns]
    X = df[available].values.astype(np.float32)
    y = df["y_anomaly"].values.astype(int)
    meta_cols = [c for c in df.columns if c not in feature_names]
    return X, y, df[meta_cols]


def load_sequence_windows(npz_path: Path, split: Optional[str] = None) -> tuple:
    """Load sequence tensors from npz.

    Returns (sequences, y, window_ids) where:
      sequences: np.ndarray (n, window_s, n_features) float32
      y: np.ndarray (n,) int
      window_ids: np.ndarray (n,) str
    """
    data = np.load(str(npz_path), allow_pickle=True)
    seqs = data["sequences"]
    y = data["y_anomaly"].astype(int)
    wids = data["window_ids"]
    splits = data["splits"]

    if split is not None:
        mask = splits == split
        seqs = seqs[mask]
        y = y[mask]
        wids = wids[mask]

    return seqs, y, wids


def load_all_windows(windows_all_parquet: Path, feature_names: list) -> dict:
    """Load all windows split by train/val/test.

    Training filter: source_dataset == 'clean' AND y_anomaly == 0.
    This prevents contamination from attacked-source windows in the training set.
    Raises RuntimeError if source_dataset column is missing from the parquet file.
    """
    X_tr, y_tr, m_tr = load_windows(windows_all_parquet, feature_names, "train")
    X_va, y_va, m_va = load_windows(windows_all_parquet, feature_names, "val")
    X_te, y_te, m_te = load_windows(windows_all_parquet, feature_names, "test")

    # Require source_dataset column — fail clearly if missing
    if "source_dataset" not in m_tr.columns:
        raise RuntimeError(
            "source_dataset column is missing from the parquet file. "
            "Re-run 00_build_model_windows.py to regenerate windows with "
            "the source_dataset column (added in the current patch)."
        )

    # Training: clean-source normal windows only
    clean_normal_mask = (y_tr == 0) & (m_tr["source_dataset"].values == "clean")
    n_clean_normal = int(clean_normal_mask.sum())
    n_total_train = len(y_tr)
    n_normal_all = int((y_tr == 0).sum())
    if n_clean_normal == 0:
        raise RuntimeError(
            f"No clean-source normal training windows found. "
            f"Train split has {n_total_train} windows, {n_normal_all} normal, "
            f"but none with source_dataset='clean'. "
            f"Check window builder output."
        )
    X_train_normal = X_tr[clean_normal_mask]

    return {
        "X_tr": X_tr, "y_tr": y_tr, "m_tr": m_tr,
        "X_tr_normal": X_train_normal,
        "n_clean_normal_train": n_clean_normal,
        "X_va": X_va, "y_va": y_va, "m_va": m_va,
        "X_te": X_te, "y_te": y_te, "m_te": m_te,
    }


def choose_threshold(scores_val: np.ndarray, y_val: np.ndarray,
                     percentiles: list = None) -> tuple:
    """Choose detection threshold from validation set by maximizing F1.

    Returns (best_threshold, sweep_results)
    If no anomalies in val, falls back to 95th percentile of normal scores.
    """
    if percentiles is None:
        percentiles = [90, 95, 97.5, 99, 99.5, 99.9]

    sweep = []
    if y_val.sum() == 0:
        # No anomalies in validation — use percentile of normal scores
        for p in percentiles:
            thr = float(np.percentile(scores_val, p))
            sweep.append({"percentile": p, "threshold": thr, "f1": None, "note": "no_anomalies_in_val"})
        best_thr = float(np.percentile(scores_val, 95))
        return best_thr, sweep

    for p in percentiles:
        thr = float(np.percentile(scores_val, p))
        y_pred = (scores_val >= thr).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        sweep.append({"percentile": p, "threshold": thr, "f1": f1})

    best = max(sweep, key=lambda x: x.get("f1") or -1)
    return best["threshold"], sweep


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    scores: Optional[np.ndarray] = None) -> dict:
    """Compute full set of binary classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
    }

    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        except Exception:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def save_scores(window_ids, y_true, y_pred, scores, model_name: str,
                config_id: str, results_dir: Path):
    """Append model scores to the shared model_scores_all.csv file."""
    df = pd.DataFrame({
        "window_id": window_ids,
        "model_name": model_name,
        "config_id": config_id,
        "y_true": y_true.astype(int),
        "y_pred": y_pred.astype(int),
        # Preserve full score precision so downstream metric recomputation
        # exactly matches the saved artifact outputs.
        "score": scores.astype(np.float64),
    })
    out = results_dir / "model_scores_all.csv"
    mode = "a" if out.exists() else "w"
    header = not out.exists()
    df.to_csv(out, mode=mode, header=header, index=False)


def save_metrics_row(metrics: dict, model_name: str, config_id: str,
                     status: str, extra: dict, results_dir: Path):
    """Append a metrics row to model_metrics_all.csv."""
    row = {
        "model_name": model_name,
        "config_id": config_id,
        "status": status,
        **metrics,
        **extra,
    }
    out = results_dir / "model_metrics_all.csv"
    df = pd.DataFrame([row])
    mode = "a" if out.exists() else "w"
    header = not out.exists()
    df.to_csv(out, mode=mode, header=header, index=False)


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Z-score standardize X_test using X_train statistics."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std
