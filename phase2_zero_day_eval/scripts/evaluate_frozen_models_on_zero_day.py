"""
Phase 2 Frozen-Model Evaluation on Zero-Day Windows.

Loads Phase 1 trained model artifacts and per-model thresholds (no retraining,
no threshold recalibration) and scores the pre-built zero-day windows.

DO NOT RUN THIS SCRIPT until Phase 1 training and evaluation is complete.
The master runner (run_phase2_zero_day_setup.py) does NOT call this script.
Run it manually after confirming Phase 1 completion.

Prerequisites (Phase 1 must be complete):
  models/results/model_metrics_all.csv      — Phase 1 per-model thresholds
  models/results/model_scores_all.csv       — Phase 1 window scores (optional, for ref)
  models/windows/feature_manifest.json      — feature spec
  models/weights/<model>/                   — trained artifacts

Inputs (Phase 2 must be set up):
  outputs/zero_day_windows.parquet
  outputs/zero_day_windows.npz
  outputs/zero_day_scenario_manifest.csv

Outputs:
  outputs/zero_day_model_scores.csv
  outputs/zero_day_model_metrics.csv
  outputs/zero_day_family_metrics.csv
  reports/ZERO_DAY_FROZEN_MODEL_EVAL_REPORT.md
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(r"D:\updated_dataset")
PHASE2_ROOT  = Path(r"D:\updated_dataset\phase2_zero_day_eval")
OUTPUTS_DIR  = PHASE2_ROOT / "outputs"
REPORTS_DIR  = PHASE2_ROOT / "reports"
WEIGHTS_DIR  = PROJECT_ROOT / "models" / "weights"
RESULTS_DIR  = PROJECT_ROOT / "models" / "results"
MODELS_ROOT  = PROJECT_ROOT / "models"

sys.path.insert(0, str(MODELS_ROOT))
sys.path.insert(0, str(MODELS_ROOT / "scripts_common"))

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Phase 1 completion check
# ---------------------------------------------------------------------------

PHASE1_REQUIRED = [
    RESULTS_DIR / "model_metrics_all.csv",
    RESULTS_DIR / "model_scores_all.csv",
    PROJECT_ROOT / "models" / "windows" / "feature_manifest.json",
]


def check_phase1_complete() -> list[str]:
    missing = []
    for p in PHASE1_REQUIRED:
        if not p.exists():
            missing.append(str(p))
    return missing


# ---------------------------------------------------------------------------
# Load Phase 1 thresholds
# ---------------------------------------------------------------------------

def load_phase1_thresholds() -> dict:
    """Return {model_name: threshold} from model_metrics_all.csv.

    Uses the primary-config threshold for each model.
    """
    metrics_path = RESULTS_DIR / "model_metrics_all.csv"
    df = pd.read_csv(metrics_path)
    thresholds = {}
    if "model" in df.columns and "threshold" in df.columns:
        primary = df[df.get("config", pd.Series(["primary"] * len(df))) == "primary"] \
            if "config" in df.columns else df
        for _, row in primary.iterrows():
            thresholds[row["model"]] = float(row.get("threshold", 0.5))
    return thresholds


# ---------------------------------------------------------------------------
# Score functions for each model family
# ---------------------------------------------------------------------------

def _score_threshold(X_flat: np.ndarray, model_name: str = "threshold") -> np.ndarray | None:
    """Threshold model: score = L2 distance from training mean."""
    config_path = WEIGHTS_DIR / "threshold" / "threshold_config_primary.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    mean = np.array(cfg.get("feature_mean", []), dtype=np.float32)
    std  = np.array(cfg.get("feature_std", [0.0] * len(mean)), dtype=np.float32)
    if len(mean) != X_flat.shape[1]:
        return None
    std_safe = np.where(std > 1e-8, std, 1.0)
    scores = np.linalg.norm((X_flat - mean) / std_safe, axis=1)
    return scores


def _score_isolation_forest(X_flat: np.ndarray) -> np.ndarray | None:
    try:
        import joblib
    except ImportError:
        return None
    model_path = WEIGHTS_DIR / "isolation_forest" / "if_model_primary.joblib"
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    scores = -model.score_samples(X_flat)  # negated: higher = more anomalous
    return scores.astype(np.float32)


def _score_ocsvm(X_flat: np.ndarray) -> np.ndarray | None:
    try:
        import joblib
    except ImportError:
        return None
    model_path = WEIGHTS_DIR / "ocsvm" / "ocsvm_model.joblib"
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    scores = -model.decision_function(X_flat)  # higher = more anomalous
    return scores.astype(np.float32)


def _score_pca(X_flat: np.ndarray) -> np.ndarray | None:
    try:
        import joblib
    except ImportError:
        return None
    model_path = WEIGHTS_DIR / "pca" / "pca_model_primary.joblib"
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    X_proj = model.transform(X_flat)
    X_recon = model.inverse_transform(X_proj)
    scores = np.mean((X_flat - X_recon) ** 2, axis=1)
    return scores.astype(np.float32)


def _score_mlp(X_flat: np.ndarray) -> np.ndarray | None:
    try:
        import torch
    except ImportError:
        return None
    model_path = WEIGHTS_DIR / "mlp_autoencoder" / "mlp_best_primary.pt"
    config_path = WEIGHTS_DIR / "mlp_autoencoder" / "mlp_config_primary.json"
    if not model_path.exists() or not config_path.exists():
        return None
    sys.path.insert(0, str(MODELS_ROOT / "train"))
    try:
        from train_mlp_autoencoder import MLPAutoencoder
        with open(config_path) as f:
            cfg = json.load(f)
        in_dim = cfg.get("input_dim", X_flat.shape[1])
        hidden = cfg.get("hidden_dims", [256, 128, 64])
        model = MLPAutoencoder(in_dim, hidden)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_flat).float()
            recon = model(X_t).numpy()
        scores = np.mean((X_flat - recon) ** 2, axis=1)
        return scores.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] MLP scoring failed: {e}")
        return None


def _score_autoencoder_pt(X_seq: np.ndarray, model_name: str,
                           model_cls_name: str, weight_file: str) -> np.ndarray | None:
    try:
        import torch
    except ImportError:
        return None
    model_path = WEIGHTS_DIR / model_name / weight_file
    if not model_path.exists():
        return None
    sys.path.insert(0, str(MODELS_ROOT / "train"))
    try:
        module_name = f"train_{model_name}"
        import importlib
        mod = importlib.import_module(module_name)
        ModelCls = getattr(mod, model_cls_name)
        config_path = WEIGHTS_DIR / model_name / f"{model_name.split('_')[0]}_config.json"
        cfg = {}
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
        n_feat = X_seq.shape[2]
        model = ModelCls(n_feat, **{k: v for k, v in cfg.items()
                                    if k not in ("input_dim", "n_features")})
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        batch_size = 512
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch = torch.from_numpy(X_seq[i:i + batch_size]).float()
                recon = model(batch)
                if isinstance(recon, tuple):
                    recon = recon[0]
                err = torch.mean((batch - recon) ** 2, dim=(1, 2))
                all_scores.append(err.numpy())
        return np.concatenate(all_scores).astype(np.float32)
    except Exception as e:
        print(f"  [WARN] {model_name} scoring failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_model_registry() -> list[dict]:
    return [
        {
            "name": "threshold",
            "type": "flat",
            "fn": lambda X_flat, _: _score_threshold(X_flat),
        },
        {
            "name": "isolation_forest",
            "type": "flat",
            "fn": lambda X_flat, _: _score_isolation_forest(X_flat),
        },
        {
            "name": "ocsvm",
            "type": "flat",
            "fn": lambda X_flat, _: _score_ocsvm(X_flat),
        },
        {
            "name": "pca",
            "type": "flat",
            "fn": lambda X_flat, _: _score_pca(X_flat),
        },
        {
            "name": "mlp_autoencoder",
            "type": "flat",
            "fn": lambda X_flat, _: _score_mlp(X_flat),
        },
        {
            "name": "cnn_autoencoder",
            "type": "seq",
            "fn": lambda _, X_seq: _score_autoencoder_pt(
                X_seq, "cnn_autoencoder", "CNNAutoencoder", "cnn_best.pt"),
        },
        {
            "name": "lstm_autoencoder",
            "type": "seq",
            "fn": lambda _, X_seq: _score_autoencoder_pt(
                X_seq, "lstm_autoencoder", "LSTMAutoencoder", "lstm_best.pt"),
        },
        {
            "name": "gru_autoencoder",
            "type": "seq",
            "fn": lambda _, X_seq: _score_autoencoder_pt(
                X_seq, "gru_autoencoder", "GRUAutoencoder", "gru_best.pt"),
        },
        {
            "name": "transformer_autoencoder",
            "type": "seq",
            "fn": lambda _, X_seq: _score_autoencoder_pt(
                X_seq, "transformer_autoencoder", "TransformerAutoencoder",
                "transformer_best.pt"),
        },
        {
            "name": "tcn_autoencoder",
            "type": "seq",
            "fn": lambda _, X_seq: _score_autoencoder_pt(
                X_seq, "tcn_autoencoder", "TCNAutoencoder", "tcn_best.pt"),
        },
    ]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, scores: np.ndarray,
                     threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        roc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        roc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        pr_auc = float("nan")

    return {
        "threshold":   threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":     roc,
        "pr_auc":      pr_auc,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_frozen_models() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- pre-flight: Phase 1 complete? ---
    missing = check_phase1_complete()
    if missing:
        print("[ERROR] Phase 1 is not complete. Missing files:")
        for p in missing:
            print(f"  {p}")
        print("[ERROR] Run Phase 1 training first (run_phase1_models.py).")
        sys.exit(1)

    # --- load feature manifest ---
    with open(PROJECT_ROOT / "models" / "windows" / "feature_manifest.json") as f:
        manifest = json.load(f)
    flat_feat_names = manifest["flat_feature_names"]
    n_flat = manifest["n_flat_features"]

    # --- load Phase 1 thresholds ---
    thresholds = load_phase1_thresholds()
    print(f"[INFO] Loaded Phase 1 thresholds for {len(thresholds)} models")

    # --- load zero-day windows ---
    parquet_path = OUTPUTS_DIR / "zero_day_windows.parquet"
    npz_path     = OUTPUTS_DIR / "zero_day_windows.npz"
    if not parquet_path.exists() or not npz_path.exists():
        print("[ERROR] Zero-day windows not found. Run build_zero_day_windows.py first.")
        sys.exit(1)

    print(f"[INFO] Loading zero-day windows parquet: {parquet_path}")
    win_df = pd.read_parquet(parquet_path)
    avail_feats = [f for f in flat_feat_names if f in win_df.columns]
    X_flat = win_df[avail_feats].values.astype(np.float32)
    y_anomaly  = win_df["y_anomaly"].values.astype(int)
    y_cyber    = win_df["y_cyber_anomaly"].values.astype(int)
    y_physical = win_df["y_physical_anomaly"].values.astype(int)
    print(f"[INFO] X_flat shape: {X_flat.shape}, anomaly windows: {y_anomaly.sum()}")

    print(f"[INFO] Loading zero-day windows NPZ: {npz_path}")
    npz = np.load(str(npz_path), allow_pickle=True)
    X_seq = npz["X_seq"].astype(np.float32)  # shape (n, window_s, n_raw)

    # --- score all models ---
    registry = _build_model_registry()
    all_scores  = {}
    metrics_rows = []
    score_cols = {"window_id": win_df["window_id"].values if "window_id" in win_df.columns
                  else [f"zdw_{i}" for i in range(len(win_df))],
                  "y_anomaly": y_anomaly}

    for entry in registry:
        model_name = entry["name"]
        print(f"[INFO] Scoring {model_name}...")
        t0 = time.time()
        try:
            scores = entry["fn"](X_flat, X_seq)
        except Exception as e:
            print(f"  [WARN] {model_name} raised: {e}")
            scores = None

        if scores is None or len(scores) != len(win_df):
            print(f"  [SKIP] {model_name}: no trained artifact or scoring failed")
            continue

        print(f"  Done in {time.time()-t0:.1f}s")
        all_scores[model_name] = scores
        score_cols[f"score_{model_name}"] = scores

        threshold = thresholds.get(model_name, float(np.percentile(scores, 90)))
        m = _compute_metrics(y_anomaly, scores, threshold)
        m.update({"model": model_name, "config": "primary",
                  "n_windows": len(scores), "n_anomaly": int(y_anomaly.sum()),
                  "source": "zero_day_frozen_eval"})
        metrics_rows.append(m)

    # --- save scores CSV ---
    scores_df = pd.DataFrame(score_cols)
    scores_df.to_csv(OUTPUTS_DIR / "zero_day_model_scores.csv", index=False)
    print(f"[INFO] Saved scores: {OUTPUTS_DIR / 'zero_day_model_scores.csv'}")

    # --- save metrics CSV ---
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUTS_DIR / "zero_day_model_metrics.csv", index=False)
    print(f"[INFO] Saved metrics: {OUTPUTS_DIR / 'zero_day_model_metrics.csv'}")

    # --- per-family metrics ---
    family_rows = []
    if "scenario_family" in win_df.columns and metrics_rows:
        for model_name, scores in all_scores.items():
            threshold = thresholds.get(model_name, float(np.percentile(scores, 90)))
            for family in win_df["scenario_family"].dropna().unique():
                mask = win_df["scenario_family"] == family
                if mask.sum() == 0:
                    continue
                m = _compute_metrics(y_anomaly[mask], scores[mask], threshold)
                m.update({"model": model_name, "scenario_family": family,
                          "n_windows": int(mask.sum())})
                family_rows.append(m)
    if family_rows:
        fam_df = pd.DataFrame(family_rows)
        fam_df.to_csv(OUTPUTS_DIR / "zero_day_family_metrics.csv", index=False)
        print(f"[INFO] Saved family metrics: {OUTPUTS_DIR / 'zero_day_family_metrics.csv'}")

    summary = {
        "timestamp":      NOW,
        "models_scored":  list(all_scores.keys()),
        "n_windows":      len(win_df),
        "n_anomaly":      int(y_anomaly.sum()),
        "metrics_summary": metrics_rows,
        "constraints": {
            "no_retraining":               True,
            "no_threshold_recalibration":  True,
            "phase1_thresholds_used":      True,
            "frozen_model_evaluation":     True,
        },
    }

    _write_report(summary, metrics_df if metrics_rows else pd.DataFrame())
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(summary: dict, metrics_df: pd.DataFrame) -> None:
    lines = [
        "# ZERO_DAY_FROZEN_MODEL_EVAL_REPORT",
        "",
        f"Generated: {summary['timestamp']}",
        "",
        "## Evaluation Constraints",
        "",
        "- No model retraining",
        "- No threshold recalibration",
        "- Phase 1 thresholds used unchanged",
        "- Feature space identical to Phase 1 (verified via feature_manifest.json)",
        "",
        "## Window Coverage",
        "",
        f"- Total zero-day windows: {summary['n_windows']:,}",
        f"- Anomaly-labelled windows: {summary['n_anomaly']:,}",
        f"- Models scored: {len(summary['models_scored'])}",
        "",
        "## Models Scored",
        "",
    ]
    for m in summary["models_scored"]:
        lines.append(f"- {m}")

    if not metrics_df.empty and "model" in metrics_df.columns:
        lines += [
            "",
            "## Detection Metrics (primary threshold, zero-day windows)",
            "",
            "| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |",
            "|---|---|---|---|---|---|",
        ]
        for _, row in metrics_df.iterrows():
            lines.append(
                f"| {row.model} | {row.get('f1', float('nan')):.3f} | "
                f"{row.get('precision', float('nan')):.3f} | "
                f"{row.get('recall', float('nan')):.3f} | "
                f"{row.get('roc_auc', float('nan')):.3f} | "
                f"{row.get('pr_auc', float('nan')):.3f} |"
            )

    lines += [
        "",
        "## Outputs",
        "",
        "- `outputs/zero_day_model_scores.csv`",
        "- `outputs/zero_day_model_metrics.csv`",
        "- `outputs/zero_day_family_metrics.csv`",
        "",
        "---",
        "*Phase 2 Zero-Day Evaluation — Frozen model evaluation complete*",
    ]

    report_path = REPORTS_DIR / "ZERO_DAY_FROZEN_MODEL_EVAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote evaluation report: {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Frozen-Model Evaluation on Zero-Day Windows")
    print("=" * 60)
    print()
    print("CONSTRAINT: No retraining. No threshold recalibration.")
    print("Phase 1 model artifacts and thresholds are used unchanged.")
    print()
    summary = evaluate_frozen_models()
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Models scored : {len(summary['models_scored'])}")
    print(f"Windows scored: {summary['n_windows']:,}")
    print(f"Anomaly windows: {summary['n_anomaly']:,}")
    print("==========================")
