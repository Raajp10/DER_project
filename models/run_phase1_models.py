"""
Phase 1 Master Run Script.

Runs the complete Phase 1 model benchmark pipeline via subprocess:
  STEP 0: Dataset freeze check
  STEP 1: Build sliding windows (60s/10s primary + secondary)
  STEP 2: Leakage audit
  STEP 3: Train all 11 models
  STEP 4: Evaluate (scenario family, zero-day, latency, ensembles)
  STEP 5: Generate 11 figures
  STEP 6: Generate 8 reports + 4 docs
  STEP 7: Package ZIP

Usage:
    python models/run_phase1_models.py [--skip-windows] [--skip-training]
"""
import sys
import json
import time
import hashlib
import subprocess
import argparse
import zipfile
import traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    CLEAN_CSV, ATTACKED_CSV, SCENARIO_MANIFEST_CSV,
    WINDOWS_ALL_PARQUET, SEQ_ALL_NPZ, RESULTS_DIR, WEIGHTS_DIR,
    DOCS_DIR, FIGURES_DIR, LOGS_DIR, ensure_all_dirs,
)

TRAIN_DIR = ROOT / "models" / "train"
EVAL_DIR = ROOT / "models" / "evaluate"
PYTHON = sys.executable


def _run_script(script_path: Path, step_name: str, required: bool = True,
                timeout_seconds: int = 7200) -> bool:
    """Run a Python script as subprocess. Returns True on success."""
    print(f"\n{'-'*70}")
    print(f"  STEP: {step_name}")
    print(f"  Script: {script_path.relative_to(ROOT)}")
    print(f"{'-'*70}")
    t0 = time.time()
    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            capture_output=False,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  [OK] {step_name} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"  [FAIL] {step_name} FAILED (exit {result.returncode}) in {elapsed:.1f}s")
            if required:
                raise RuntimeError(f"Required step '{step_name}' failed.")
            return False
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [FAIL] {step_name} TIMEOUT after {elapsed:.1f}s")
        if required:
            raise
        return False
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] {step_name} EXCEPTION after {elapsed:.1f}s: {e}")
        if required:
            raise
        return False


def step0_dataset_freeze_check():
    """Verify source dataset files exist and compute checksums."""
    required_files = [CLEAN_CSV, ATTACKED_CSV, SCENARIO_MANIFEST_CSV]
    results = {}
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Frozen dataset file missing: {f}")
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            h.update(fh.read(4 * 1024 * 1024))
        results[f.name] = {
            "exists": True,
            "size_mb": round(f.stat().st_size / 1e6, 2),
            "sha256_4mb": h.hexdigest()[:16],
        }
    freeze_check = {
        "status": "PASS",
        "timestamp": datetime.now().isoformat(),
        "files": results,
        "note": "SHA256 of first 4MB. Dataset frozen - no writes to data_updated/.",
    }
    freeze_path = RESULTS_DIR / "dataset_freeze_check.json"
    with open(freeze_path, "w") as f:
        json.dump(freeze_check, f, indent=2)
    print(f"  Freeze check PASS - {len(results)} files verified")
    print(f"  Saved: {freeze_path}")


def step7_package_zip():
    zip_path = ROOT / "models" / "phase1_model_results_package.zip"
    include_dirs = [RESULTS_DIR, FIGURES_DIR, DOCS_DIR]

    all_files = []
    for d in include_dirs:
        if d.exists():
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    all_files.append((f, Path("phase1") / d.name / f.relative_to(d)))

    # Scripts
    for d in [ROOT / "models" / "scripts_common", TRAIN_DIR, EVAL_DIR]:
        if d.exists():
            for f in sorted(d.glob("*.py")):
                all_files.append((f, Path("phase1") / "scripts" / d.name / f.name))

    all_files.append((ROOT / "models" / "run_phase1_models.py",
                       Path("phase1") / "run_phase1_models.py"))
    all_files.append((ROOT / "models" / "run_phase1_models.bat",
                       Path("phase1") / "run_phase1_models.bat"))

    # Weight configs + small artifacts
    for model_dir in sorted((ROOT / "models" / "weights").iterdir()):
        if not model_dir.is_dir():
            continue
        for ext in ["*.json", "*.csv", "*.joblib"]:
            for f in model_dir.glob(ext):
                all_files.append((f, Path("phase1") / "weights" / model_dir.name / f.name))
        for pt in model_dir.glob("*.pt"):
            if pt.stat().st_size < 200 * 1024 * 1024:
                all_files.append((pt, Path("phase1") / "weights" / model_dir.name / pt.name))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc in all_files:
            if src.exists():
                zf.write(src, arc)

    size_mb = zip_path.stat().st_size / 1e6
    print(f"  ZIP: {zip_path}")
    print(f"  Size: {size_mb:.1f} MB  ({len(all_files)} files)")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Model Benchmark")
    parser.add_argument("--skip-windows", action="store_true",
                        help="Skip window builder if windows already built")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training (requires existing metrics)")
    args = parser.parse_args()

    ensure_all_dirs()
    t_start = time.time()

    print("\n" + "="*70)
    print("  PHASE 1 MODEL BENCHMARK - FULL PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    steps_ok = {}

    # STEP 0: Freeze check
    print(f"\n{'-'*70}")
    print("  STEP: Dataset Freeze Check")
    print(f"{'-'*70}")
    try:
        step0_dataset_freeze_check()
        steps_ok["freeze"] = True
    except Exception as e:
        print(f"  [FAIL] Freeze check FAILED: {e}")
        print("  Cannot continue without frozen dataset. Exiting.")
        sys.exit(1)

    # STEP 1: Build windows
    if args.skip_windows and WINDOWS_ALL_PARQUET.exists() and SEQ_ALL_NPZ.exists():
        print(f"\n{'-'*70}")
        print("  STEP: Build Sliding Windows - SKIPPED (already built)")
        print(f"{'-'*70}")
        steps_ok["windows"] = True
    else:
        steps_ok["windows"] = _run_script(
            TRAIN_DIR / "00_build_model_windows.py",
            "Build Sliding Windows", required=True)

    # STEP 2: Leakage audit
    steps_ok["leakage"] = _run_script(
        TRAIN_DIR / "01_leakage_audit.py",
        "Leakage Audit", required=True)

    # STEP 3: Train all models
    if args.skip_training and (RESULTS_DIR / "model_metrics_all.csv").exists():
        print(f"\n{'-'*70}")
        print("  STEP: Train All Models - SKIPPED (metrics already exist)")
        print(f"{'-'*70}")
        steps_ok["training"] = True
    else:
        steps_ok["training"] = _run_script(
            TRAIN_DIR / "train_all_models.py",
            "Train All 11 Models", required=True, timeout_seconds=28800)

    # STEP 4: Evaluate
    steps_ok["evaluate"] = _run_script(
        EVAL_DIR / "evaluate_models.py",
        "Evaluate (scenario/zero-day/latency/ensembles)", required=False)

    # STEP 5: Figures
    steps_ok["figures"] = _run_script(
        EVAL_DIR / "generate_figures.py",
        "Generate 11 Figures", required=False)

    # STEP 6: Reports
    steps_ok["reports"] = _run_script(
        EVAL_DIR / "generate_reports.py",
        "Generate 8 Reports + 4 Docs", required=False)

    # STEP 7: ZIP
    print(f"\n{'-'*70}")
    print("  STEP: Package ZIP")
    print(f"{'-'*70}")
    try:
        step7_package_zip()
        steps_ok["zip"] = True
    except Exception as e:
        print(f"  [FAIL] ZIP failed: {e}")
        steps_ok["zip"] = False

    total_elapsed = time.time() - t_start

    # ─── Final Summary ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PHASE 1 PIPELINE SUMMARY")
    print("="*70)
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    for step, ok in steps_ok.items():
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon} {step}")

    # Print metrics table
    metrics_csv = RESULTS_DIR / "model_metrics_all.csv"
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        primary = df[df.get("config_id", pd.Series(dtype=str)) == "primary"] \
            if "config_id" in df.columns else df
        print(f"\n  {'Model':32s} {'Status':8s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'AUC':>7s}")
        print(f"  {'-'*32} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for _, row in primary.sort_values("f1", ascending=False, na_position="last").iterrows():
            f1_s = f"{row['f1']:7.4f}" if not pd.isna(row["f1"]) else "      -"
            p_s = f"{row['precision']:7.4f}" if not pd.isna(row.get("precision", float("nan"))) else "      -"
            r_s = f"{row['recall']:7.4f}" if not pd.isna(row.get("recall", float("nan"))) else "      -"
            a_s = f"{row['roc_auc']:7.4f}" if not pd.isna(row.get("roc_auc", float("nan"))) else "      -"
            print(f"  {row['model_name']:32s} {row.get('status','?'):8s} {f1_s} {p_s} {r_s} {a_s}")

    n_pass = 0
    n_not_run = 0
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        if "config_id" in df.columns:
            p = df[df["config_id"] == "primary"]
            n_pass = int((p["status"] == "PASS").sum())
            n_not_run = int((p["status"] == "NOT_RUN").sum())

    fig_count = len(list(FIGURES_DIR.glob("*.png")))

    print(f"\n  {'='*70}")
    if n_pass >= 9 and n_not_run <= 2 and fig_count >= 9:
        print(f"  PHASE 1 COMPLETE - {n_pass}/11 PASS | {fig_count} figures | "
              f"NOT_RUN: {n_not_run}")
    else:
        print(f"  PHASE 1 PARTIAL - {n_pass}/11 PASS | {fig_count} figures | "
              f"NOT_RUN: {n_not_run}")
    print(f"  {'='*70}")
    print(f"\n  Outputs:")
    print(f"    Results: {RESULTS_DIR}")
    print(f"    Figures: {FIGURES_DIR}")
    print(f"    Docs:    {DOCS_DIR}")
    zip_path = ROOT / "models" / "phase1_model_results_package.zip"
    if zip_path.exists():
        print(f"    ZIP:     {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
