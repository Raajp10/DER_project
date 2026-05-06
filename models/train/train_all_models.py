"""
Master trainer for all Phase 1 models.

Runs all 11 primary models in order, then secondary window-config sweeps
(30s/10s and 120s/30s) for tabular models (threshold, isolation_forest,
pca, mlp). Secondary sweeps are skipped with NOT_RUN if the corresponding
windows_all_<config>.parquet file does not exist.

Order:
  1. threshold_baseline
  2. isolation_forest
  3. one_class_svm
  4. pca_reconstruction
  5. mlp_autoencoder
  6. cnn_autoencoder
  7. lstm_autoencoder
  8. gru_autoencoder
  9. transformer_autoencoder
  10. tcn_autoencoder
  11. ttm
  (then secondary sweeps: sec30s + sec120s for threshold/IF/pca/mlp)
"""
import sys
import json
import time
import traceback
from pathlib import Path

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))
sys.path.insert(0, str(ROOT / "models" / "train"))

from common_paths import RESULTS_DIR, WEIGHTS_DIR, WINDOWS_DIR, ensure_all_dirs
from model_utils import save_metrics_row


def _run_model(name: str, fn):
    """Run a model training function, catch all exceptions."""
    print(f"\n{'='*60}")
    print(f"  [{name}]")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        status = result.get("status", "UNKNOWN") if isinstance(result, dict) else "DONE"
        print(f"  => {name}: {status} in {elapsed:.1f}s")
        return {"model": name, "status": status, "elapsed_s": round(elapsed, 2),
                "error": None, "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f"  => {name}: EXCEPTION after {elapsed:.1f}s")
        print(f"     {e}")
        return {"model": name, "status": "EXCEPTION", "elapsed_s": round(elapsed, 2),
                "error": str(e), "traceback": tb, "result": None}


def _run_secondary(name: str, fn, window_config: str) -> dict:
    """Run a secondary window-config sweep. Returns NOT_RUN if parquet missing."""
    parquet = WINDOWS_DIR / f"windows_all_{window_config}.parquet"
    if not parquet.exists():
        reason = (f"Secondary parquet windows_all_{window_config}.parquet not found. "
                  f"Re-run 00_build_model_windows.py with secondary configs to generate it.")
        print(f"\n  [{name} / {window_config}] SKIPPED - {reason}")
        save_metrics_row(
            {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
             "roc_auc": float("nan"), "pr_auc": float("nan"),
             "accuracy": float("nan"), "balanced_accuracy": float("nan"),
             "tp": 0, "fp": 0, "tn": 0, "fn": 0,
             "false_positive_count": 0, "false_negative_count": 0},
            name, window_config, "NOT_RUN",
            {"failure_reason": reason, "runtime_seconds": 0.0, "artifact_path": ""},
            RESULTS_DIR,
        )
        return {"model": name, "config": window_config, "status": "NOT_RUN",
                "elapsed_s": 0.0, "error": reason, "result": None}
    return _run_model(f"{name}/{window_config}", lambda: fn(window_config))


def main():
    ensure_all_dirs()
    t_start = time.time()

    print("\n" + "="*70)
    print("  PHASE 1 MODEL TRAINING - ALL 11 MODELS + SECONDARY SWEEPS")
    print("="*70)

    # Import all training functions
    from train_threshold import train_threshold
    from train_isolation_forest import train_isolation_forest
    from train_ocsvm import train_ocsvm
    from train_pca import train_pca
    from train_mlp_autoencoder import train_mlp
    from train_cnn_autoencoder import train_cnn
    from train_lstm_autoencoder import train_lstm
    from train_gru_autoencoder import train_gru
    from train_transformer_autoencoder import train_transformer
    from train_tcn_autoencoder import train_tcn
    from train_ttm import train_ttm

    PRIMARY_MODELS = [
        ("threshold_baseline",       lambda: train_threshold("primary")),
        ("isolation_forest",         lambda: train_isolation_forest("primary")),
        ("one_class_svm",            train_ocsvm),
        ("pca_reconstruction",       lambda: train_pca("primary")),
        ("mlp_autoencoder",          lambda: train_mlp("primary")),
        ("cnn_autoencoder",          train_cnn),
        ("lstm_autoencoder",         train_lstm),
        ("gru_autoencoder",          train_gru),
        ("transformer_autoencoder",  train_transformer),
        ("tcn_autoencoder",          train_tcn),
        ("ttm",                      train_ttm),
    ]

    # Secondary window-config sweeps (tabular models only)
    SECONDARY_TABULAR = [
        ("threshold_baseline",  train_threshold),
        ("isolation_forest",    train_isolation_forest),
        ("pca_reconstruction",  train_pca),
        ("mlp_autoencoder",     train_mlp),
    ]
    SECONDARY_CONFIGS = ["30s", "120s"]

    # ── Primary ───────────────────────────────────────────────────────────────
    run_results = []
    print("\n  -- Primary models (11) --")
    for name, fn in PRIMARY_MODELS:
        r = _run_model(name, fn)
        run_results.append(r)

    # ── Secondary sweeps ──────────────────────────────────────────────────────
    secondary_results = []
    print("\n  -- Secondary window-config sweeps --")
    for name, fn in SECONDARY_TABULAR:
        for cfg in SECONDARY_CONFIGS:
            r = _run_secondary(name, fn, cfg)
            secondary_results.append(r)

    all_results = run_results + secondary_results

    # Summary
    total_elapsed = time.time() - t_start
    passed = [r for r in run_results if r["status"] == "PASS"]
    not_run = [r for r in run_results if r["status"] == "NOT_RUN"]
    failed = [r for r in run_results if r["status"] not in ("PASS", "NOT_RUN")]

    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Primary PASS:    {len(passed)}/11")
    print(f"  Primary NOT_RUN: {len(not_run)}/11")
    print(f"  Primary FAILED:  {len(failed)}/11")
    print()
    for r in run_results:
        icon = "[OK]" if r["status"] == "PASS" else ("[SKIP]" if r["status"] == "NOT_RUN" else "[FAIL]")
        print(f"  {icon} {r['model']:30s} {r['status']:12s} {r['elapsed_s']:7.1f}s")

    sec_pass = sum(1 for r in secondary_results if r["status"] == "PASS")
    sec_skip = sum(1 for r in secondary_results if r["status"] == "NOT_RUN")
    print(f"\n  Secondary sweeps: {sec_pass} PASS, {sec_skip} NOT_RUN (missing parquet)")

    # Check if metrics file exists and print F1 summary
    metrics_csv = RESULTS_DIR / "model_metrics_all.csv"
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        primary = df[df["config_id"] == "primary"].copy()
        if not primary.empty:
            primary_pass = primary[primary["status"] == "PASS"]
            print(f"\n  F1 scores (primary config, PASS models):")
            for _, row in primary_pass.sort_values("f1", ascending=False).iterrows():
                print(f"    {row['model_name']:30s} F1={row['f1']:.4f}  "
                      f"ROC-AUC={row['roc_auc']:.4f}")

    # Save run summary
    summary = {
        "total_models": 11,
        "passed": len(passed),
        "not_run": len(not_run),
        "failed": len(failed),
        "total_elapsed_s": round(total_elapsed, 2),
        "results": run_results,
        "secondary_results": secondary_results,
    }
    summary_path = RESULTS_DIR / "training_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Run summary saved: {summary_path}")

    if failed:
        print(f"\n  WARNING: {len(failed)} model(s) raised exceptions:")
        for r in failed:
            print(f"    {r['model']}: {r['error']}")

    print("\n  Training complete. Run evaluate_models.py next.")
    return summary


if __name__ == "__main__":
    main()
