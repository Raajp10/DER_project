"""
Phase 1 Result Finalization.

Reads model_scores_all.csv and model_metrics_all.csv.
Produces:
  results/model_metrics_best.csv      — best config per model by F1
  results/model_thresholds.json       — threshold per model (primary config)
  results/confusion_matrices.json     — TP/FP/TN/FN per model (primary config)
  results/metric_recompute_audit.json — recomputed vs saved metrics comparison
  reports/METRIC_RECOMPUTE_AUDIT.md   — human-readable audit report

Fails (raises RuntimeError) if recomputed metrics disagree with saved metrics
beyond the tolerance defined in TOLERANCE.
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
    accuracy_score, balanced_accuracy_score,
    confusion_matrix,
)

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    RESULTS_DIR, REPORTS_DIR, WEIGHTS_DIR, ensure_all_dirs,
    MODEL_METRICS_CSV, MODEL_SCORES_CSV, MODEL_METRICS_BEST_CSV,
    MODEL_THRESHOLDS_JSON, CONFUSION_MATRICES_JSON, METRIC_RECOMPUTE_JSON,
)

# Numeric tolerance for metric agreement (floating-point rounding expected)
TOLERANCE = 1e-4
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_metrics() -> pd.DataFrame:
    if not MODEL_METRICS_CSV.exists():
        raise FileNotFoundError(f"model_metrics_all.csv not found: {MODEL_METRICS_CSV}")
    return pd.read_csv(MODEL_METRICS_CSV)


def _load_scores() -> pd.DataFrame:
    if not MODEL_SCORES_CSV.exists():
        raise FileNotFoundError(f"model_scores_all.csv not found: {MODEL_SCORES_CSV}")
    return pd.read_csv(MODEL_SCORES_CSV)


# ── best metrics CSV ──────────────────────────────────────────────────────────

def write_model_metrics_best(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Write model_metrics_best.csv: best config per model by primary-config F1."""
    primary_pass = metrics_df[
        (metrics_df.get("config_id", pd.Series(dtype=str)) == "primary") &
        (metrics_df.get("status", pd.Series(dtype=str)) == "PASS")
    ] if "config_id" in metrics_df.columns else pd.DataFrame()

    if primary_pass.empty:
        print("  finalize: no primary PASS rows — model_metrics_best.csv will be empty")
        best_df = pd.DataFrame()
    else:
        # One row per model (best F1 among any config)
        all_pass = metrics_df[
            metrics_df.get("status", pd.Series(dtype=str)) == "PASS"
        ] if "status" in metrics_df.columns else metrics_df

        rows = []
        for model in all_pass["model_name"].unique() if "model_name" in all_pass.columns else []:
            model_rows = all_pass[all_pass["model_name"] == model]
            best_row = model_rows.sort_values("f1", ascending=False).iloc[0]
            rows.append(best_row)
        best_df = pd.DataFrame(rows).reset_index(drop=True)

    best_df.to_csv(MODEL_METRICS_BEST_CSV, index=False)
    print(f"  Saved: {MODEL_METRICS_BEST_CSV.name} ({len(best_df)} rows)")
    return best_df


# ── model thresholds JSON ─────────────────────────────────────────────────────

def write_model_thresholds(metrics_df: pd.DataFrame):
    """Write model_thresholds.json: threshold for each model, primary config."""
    thresholds = {}
    if "model_name" in metrics_df.columns and "threshold" in metrics_df.columns:
        primary = metrics_df[
            metrics_df.get("config_id", pd.Series(dtype=str)) == "primary"
        ] if "config_id" in metrics_df.columns else metrics_df

        for _, row in primary.iterrows():
            model = row["model_name"]
            thr = row["threshold"]
            thresholds[model] = {
                "threshold": float(thr) if not pd.isna(thr) else None,
                "status": row.get("status", "UNKNOWN"),
                "config_id": row.get("config_id", "primary"),
            }

    with open(MODEL_THRESHOLDS_JSON, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  Saved: {MODEL_THRESHOLDS_JSON.name} ({len(thresholds)} models)")


# ── confusion matrices JSON ────────────────────────────────────────────────────

def write_confusion_matrices(metrics_df: pd.DataFrame):
    """Write confusion_matrices.json: TP/FP/TN/FN per model (primary config)."""
    matrices = {}
    if "model_name" not in metrics_df.columns:
        with open(CONFUSION_MATRICES_JSON, "w") as f:
            json.dump(matrices, f, indent=2)
        return

    primary = metrics_df[
        metrics_df.get("config_id", pd.Series(dtype=str)) == "primary"
    ] if "config_id" in metrics_df.columns else metrics_df

    for _, row in primary.iterrows():
        model = row["model_name"]
        matrices[model] = {
            "status": row.get("status", "UNKNOWN"),
            "tp": int(row["tp"]) if "tp" in row and not pd.isna(row["tp"]) else None,
            "fp": int(row["fp"]) if "fp" in row and not pd.isna(row["fp"]) else None,
            "tn": int(row["tn"]) if "tn" in row and not pd.isna(row["tn"]) else None,
            "fn": int(row["fn"]) if "fn" in row and not pd.isna(row["fn"]) else None,
        }

    with open(CONFUSION_MATRICES_JSON, "w") as f:
        json.dump(matrices, f, indent=2)
    print(f"  Saved: {CONFUSION_MATRICES_JSON.name} ({len(matrices)} models)")


# ── metric recompute audit ────────────────────────────────────────────────────

def _recompute_from_scores(scores_df: pd.DataFrame,
                            model_name: str, config_id: str) -> dict:
    """Recompute all metrics from raw scores for one model/config."""
    sub = scores_df[
        (scores_df["model_name"] == model_name) &
        (scores_df["config_id"] == config_id)
    ]
    if sub.empty:
        return None

    y_true = sub["y_true"].values.astype(int)
    y_pred = sub["y_pred"].values.astype(int)
    scores = sub["score"].values.astype(float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)

    result = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if len(np.unique(y_true)) > 1:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            result["roc_auc"] = float("nan")
        try:
            result["pr_auc"] = float(average_precision_score(y_true, scores))
        except Exception:
            result["pr_auc"] = float("nan")
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")

    return result


def run_metric_recompute_audit(metrics_df: pd.DataFrame,
                                scores_df: pd.DataFrame) -> dict:
    """Recompute metrics from scores and compare against saved metrics.

    Returns audit dict. Raises RuntimeError if any metric disagrees beyond TOLERANCE.
    """
    NUMERIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc",
                    "accuracy", "balanced_accuracy"]
    INT_COLS = ["tp", "fp", "tn", "fn"]

    audit_rows = []
    failures = []

    primary_pass = metrics_df[
        (metrics_df.get("config_id", pd.Series(dtype=str)) == "primary") &
        (metrics_df.get("status", pd.Series(dtype=str)) == "PASS")
    ] if "config_id" in metrics_df.columns and "status" in metrics_df.columns else pd.DataFrame()

    if primary_pass.empty:
        print("  finalize: no primary PASS rows to audit")
        audit = {
            "status": "SKIPPED",
            "reason": "no primary PASS rows in model_metrics_all.csv",
            "timestamp": NOW,
            "models_audited": 0,
            "rows": [],
        }
        with open(METRIC_RECOMPUTE_JSON, "w") as f:
            json.dump(audit, f, indent=2)
        return audit

    for _, row in primary_pass.iterrows():
        model = row["model_name"]
        config = row.get("config_id", "primary")

        recomputed = _recompute_from_scores(scores_df, model, config)
        if recomputed is None:
            audit_rows.append({
                "model_name": model,
                "config_id": config,
                "status": "SKIP_NO_SCORES",
                "note": "No rows in model_scores_all.csv for this model/config",
            })
            continue

        row_audit = {
            "model_name": model,
            "config_id": config,
            "status": "PASS",
            "discrepancies": [],
        }

        for col in NUMERIC_COLS:
            saved_val = row.get(col, float("nan"))
            recomp_val = recomputed.get(col, float("nan"))
            if pd.isna(saved_val) and np.isnan(recomp_val):
                continue
            if pd.isna(saved_val) or np.isnan(recomp_val):
                diff = float("inf")
            else:
                diff = abs(float(saved_val) - float(recomp_val))
            if diff > TOLERANCE:
                discrepancy = {
                    "metric": col,
                    "saved": float(saved_val) if not pd.isna(saved_val) else None,
                    "recomputed": float(recomp_val) if not np.isnan(recomp_val) else None,
                    "abs_diff": float(diff),
                }
                row_audit["discrepancies"].append(discrepancy)
                failures.append(f"{model}/{config}/{col}: "
                                 f"saved={saved_val:.6f}, recomputed={recomp_val:.6f}, "
                                 f"diff={diff:.2e}")

        for col in INT_COLS:
            saved_val = row.get(col, float("nan"))
            recomp_val = recomputed.get(col, -999)
            if pd.notna(saved_val) and int(saved_val) != int(recomp_val):
                discrepancy = {
                    "metric": col,
                    "saved": int(saved_val),
                    "recomputed": int(recomp_val),
                    "abs_diff": abs(int(saved_val) - int(recomp_val)),
                }
                row_audit["discrepancies"].append(discrepancy)
                failures.append(f"{model}/{config}/{col}: "
                                 f"saved={int(saved_val)}, recomputed={int(recomp_val)}")

        if row_audit["discrepancies"]:
            row_audit["status"] = "FAIL"
        audit_rows.append(row_audit)

    overall_status = "FAIL" if failures else "PASS"
    audit = {
        "status": overall_status,
        "timestamp": NOW,
        "tolerance": TOLERANCE,
        "models_audited": len(audit_rows),
        "failures": len(failures),
        "failure_details": failures,
        "rows": audit_rows,
    }

    with open(METRIC_RECOMPUTE_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Saved: {METRIC_RECOMPUTE_JSON.name}  "
          f"[{overall_status}: {len(audit_rows)} models, {len(failures)} failures]")

    return audit


# ── METRIC_RECOMPUTE_AUDIT.md ─────────────────────────────────────────────────

def write_recompute_audit_md(audit: dict):
    """Write human-readable METRIC_RECOMPUTE_AUDIT.md to reports/."""
    status = audit.get("status", "UNKNOWN")
    lines = [
        "# Metric Recompute Audit",
        f"_Generated: {NOW}_",
        "",
        f"## Status: {status}",
        "",
        f"- Models audited: {audit.get('models_audited', 0)}",
        f"- Failures: {audit.get('failures', 0)}",
        f"- Tolerance: {audit.get('tolerance', TOLERANCE)}",
        "",
    ]

    if audit.get("reason"):
        lines += [f"**Reason skipped:** {audit['reason']}", ""]

    failures = audit.get("failure_details", [])
    if failures:
        lines += [
            "## Discrepancies Found (FAIL)",
            "",
            "The following saved metrics do not match recomputed metrics:",
            "",
        ]
        for f in failures:
            lines.append(f"- {f}")
        lines += [
            "",
            "**This indicates that the saved metrics in model_metrics_all.csv do not match**",
            "**the raw scores in model_scores_all.csv. Re-train affected models.**",
        ]
    else:
        lines += [
            "## Discrepancies: NONE",
            "",
            "All saved metrics agree with metrics recomputed from model_scores_all.csv.",
        ]

    rows = audit.get("rows", [])
    if rows:
        lines += [
            "",
            "## Per-Model Audit",
            "",
            "| Model | Config | Status | Discrepancies |",
            "|---|---|---|---|",
        ]
        for r in rows:
            n_disc = len(r.get("discrepancies", []))
            lines.append(
                f"| {r['model_name']} | {r.get('config_id','?')} | "
                f"{r['status']} | {n_disc} |"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "METRIC_RECOMPUTE_AUDIT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {report_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_all_dirs()
    t0 = time.time()
    print("\n" + "="*60)
    print("  PHASE 1 RESULT FINALIZATION")
    print("="*60)

    try:
        metrics_df = _load_metrics()
        print(f"  Loaded metrics: {len(metrics_df)} rows")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  Skipping finalization — run train_all_models.py first.")
        return

    try:
        scores_df = _load_scores()
        print(f"  Loaded scores:  {len(scores_df)} rows")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        scores_df = pd.DataFrame()

    print()

    # 1. Best metrics CSV
    write_model_metrics_best(metrics_df)

    # 2. Thresholds JSON
    write_model_thresholds(metrics_df)

    # 3. Confusion matrices JSON
    write_confusion_matrices(metrics_df)

    # 4. Metric recompute audit
    if scores_df.empty:
        print("  WARNING: model_scores_all.csv missing — skipping recompute audit")
        audit = {"status": "SKIPPED", "reason": "no scores file", "timestamp": NOW,
                 "models_audited": 0, "rows": []}
        with open(METRIC_RECOMPUTE_JSON, "w") as f:
            json.dump(audit, f, indent=2)
    else:
        audit = run_metric_recompute_audit(metrics_df, scores_df)
        write_recompute_audit_md(audit)

        if audit.get("status") == "FAIL":
            raise RuntimeError(
                f"Metric recompute audit FAILED: {audit['failures']} discrepancy(ies). "
                f"See {METRIC_RECOMPUTE_JSON} for details.\n"
                + "\n".join(audit.get("failure_details", []))
            )

    elapsed = time.time() - t0
    print(f"\n  Finalization complete in {elapsed:.1f}s")
    print(f"  Results dir: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
