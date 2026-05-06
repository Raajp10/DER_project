"""Common paths for Phase 1 model benchmark.

Reads from: D:/updated_dataset/data_updated/ (frozen dataset)
Writes to:  D:/updated_dataset/models/ (model artifacts)
"""
from pathlib import Path

# Dataset root (frozen — do NOT write here)
DATASET_ROOT = Path(r"D:\updated_dataset")
DATA = DATASET_ROOT / "data_updated"

# Frozen dataset inputs
CLEAN_CSV = DATA / "raw" / "physical_timeseries_clean_improved_7d.csv"
ATTACKED_CSV = DATA / "raw" / "physical_timeseries_attacked_improved_7d.csv"
SCENARIO_MANIFEST_CSV = DATA / "scenarios" / "scenario_manifest_improved_7d.csv"
CONTEXT_WINDOWS_CSV = DATA / "processed" / "event_specific_context_windows_7d.csv"
LIFECYCLE_MAP_CSV = DATA / "processed" / "cyber_physical_lifecycle_map_7d.csv"
EVIDENCE_PACKETS_JSONL = DATA / "processed" / "cyber_physical_evidence_packets_7d.jsonl"
FINAL_VALIDATION_JSON = DATA / "validation" / "final_updated_dataset_validation_summary.json"
ODS_RESULTS_JSON = DATA / "metadata" / "opendss_event_window_results.json"

# Model outputs root
MODELS_ROOT = DATASET_ROOT / "models"
WINDOWS_DIR = MODELS_ROOT / "windows"
TRAIN_DIR = MODELS_ROOT / "train"
WEIGHTS_DIR = MODELS_ROOT / "weights"
RESULTS_DIR = MODELS_ROOT / "results"
FIGURES_DIR = MODELS_ROOT / "figures"
REPORTS_DIR = MODELS_ROOT / "reports"
DOCS_DIR = MODELS_ROOT / "docs"
LOGS_DIR = MODELS_ROOT / "logs"

# Window files
WINDOWS_NORMAL_PARQUET = WINDOWS_DIR / "windows_normal.parquet"
WINDOWS_ATTACKED_PARQUET = WINDOWS_DIR / "windows_attacked.parquet"
WINDOWS_ALL_PARQUET = WINDOWS_DIR / "windows_all.parquet"
FEATURE_MANIFEST_JSON = WINDOWS_DIR / "feature_manifest.json"
WINDOW_BUILD_SUMMARY_JSON = WINDOWS_DIR / "window_build_summary.json"
SPLIT_MANIFEST_JSON = WINDOWS_DIR / "split_manifest.json"

SEQ_NORMAL_NPZ = WINDOWS_DIR / "sequence_windows_normal.npz"
SEQ_ATTACKED_NPZ = WINDOWS_DIR / "sequence_windows_attacked.npz"
SEQ_ALL_NPZ = WINDOWS_DIR / "sequence_windows_all.npz"

# Results files
MODEL_SCORES_CSV = RESULTS_DIR / "model_scores_all.csv"
MODEL_METRICS_CSV = RESULTS_DIR / "model_metrics_all.csv"
MODEL_METRICS_BEST_CSV = RESULTS_DIR / "model_metrics_best.csv"
MODEL_THRESHOLDS_JSON = RESULTS_DIR / "model_thresholds.json"
CONFUSION_MATRICES_JSON = RESULTS_DIR / "confusion_matrices.json"
SCENARIO_FAMILY_CSV = RESULTS_DIR / "scenario_family_metrics.csv"
ZERO_DAY_CSV = RESULTS_DIR / "zero_day_holdout_metrics.csv"
DETECTION_LATENCY_CSV = RESULTS_DIR / "detection_latency_metrics.csv"
ENSEMBLE_METRICS_CSV = RESULTS_DIR / "ensemble_metrics.csv"
ENSEMBLE_SCORES_CSV = RESULTS_DIR / "ensemble_scores.csv"
DATASET_FREEZE_JSON = RESULTS_DIR / "dataset_freeze_check.json"
METRIC_RECOMPUTE_JSON = RESULTS_DIR / "metric_recompute_audit.json"
LEAKAGE_AUDIT_JSON = RESULTS_DIR / "leakage_audit_summary.json"


def ensure_all_dirs():
    """Create all output directories."""
    EVAL_DIR = MODELS_ROOT / "evaluate"
    for d in [WINDOWS_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, DOCS_DIR, LOGS_DIR, EVAL_DIR,
              WEIGHTS_DIR / "threshold", WEIGHTS_DIR / "isolation_forest",
              WEIGHTS_DIR / "ocsvm", WEIGHTS_DIR / "pca",
              WEIGHTS_DIR / "mlp_autoencoder", WEIGHTS_DIR / "cnn_autoencoder",
              WEIGHTS_DIR / "lstm_autoencoder", WEIGHTS_DIR / "gru_autoencoder",
              WEIGHTS_DIR / "transformer_autoencoder", WEIGHTS_DIR / "tcn_autoencoder",
              WEIGHTS_DIR / "ttm"]:
        d.mkdir(parents=True, exist_ok=True)
