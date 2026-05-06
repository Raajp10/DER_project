"""
Phase 1 Model Evaluation.

Reads model_scores_all.csv and model_metrics_all.csv (written by training scripts).
Produces:
  - scenario_family_metrics.csv
  - zero_day_holdout_metrics.csv
  - detection_latency_metrics.csv
  - ensemble_metrics.csv
  - ZERO_DAY_HOLDOUT_REPORT.md
  - ENSEMBLE_REPORT.md

Ensembles:
  1. threshold + mlp (OR gate)
  2. threshold + isolation_forest (OR gate)
  3. best-3 majority vote
  4. best-learned (logistic regression on val scores)
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    RESULTS_DIR, WEIGHTS_DIR, WINDOWS_ALL_PARQUET, ensure_all_dirs,
)
from model_utils import compute_metrics, save_metrics_row


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_scores():
    """Load model_scores_all.csv. Returns DataFrame or None."""
    p = RESULTS_DIR / "model_scores_all.csv"
    if not p.exists():
        print("  evaluate: model_scores_all.csv not found")
        return None
    return pd.read_csv(p)


def _load_metrics():
    p = RESULTS_DIR / "model_metrics_all.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _load_windows_meta():
    """Load window metadata from parquet.

    Reads: window_id, source_dataset, split, y_anomaly, scenario_id, anomaly_type.
    Does NOT require scenario_family in the parquet — it is derived here from anomaly_type.
    window_ids are globally unique (prefixed with source_dataset) so merges are safe.
    """
    if not WINDOWS_ALL_PARQUET.exists():
        return None

    all_cols = pd.read_parquet(WINDOWS_ALL_PARQUET).columns.tolist()

    read_cols = ["window_id", "split", "y_anomaly", "scenario_id", "anomaly_type"]
    if "source_dataset" in all_cols:
        read_cols.append("source_dataset")

    df = pd.read_parquet(WINDOWS_ALL_PARQUET, columns=read_cols)

    # Derive scenario_family from anomaly_type (first token of underscore-separated name)
    if "anomaly_type" in df.columns:
        df["scenario_family"] = (
            df["anomaly_type"]
            .fillna("normal")
            .astype(str)
            .str.split("_")
            .str[0]
        )
    else:
        df["scenario_family"] = "unknown"

    return df


# ─── scenario family metrics ──────────────────────────────────────────────────

def compute_scenario_family_metrics(scores_df: pd.DataFrame,
                                    meta_df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario-family F1/precision/recall for each model on test split."""
    test_meta = meta_df[meta_df["split"] == "test"].copy()
    if "scenario_family" not in test_meta.columns:
        # Derive from anomaly_type prefix
        test_meta["scenario_family"] = test_meta["anomaly_type"].str.split("_").str[0]

    rows = []
    for model_name in scores_df["model_name"].unique():
        model_scores = scores_df[
            (scores_df["model_name"] == model_name) &
            (scores_df["config_id"] == "primary")
        ].copy()
        if model_scores.empty:
            continue
        merged = test_meta.merge(model_scores, on="window_id", how="inner")
        if merged.empty:
            continue

        for family in merged["scenario_family"].unique():
            sub = merged[merged["scenario_family"] == family]
            if len(sub) == 0:
                continue
            y_true = sub["y_true"].values
            y_pred = sub["y_pred"].values
            scores_sub = sub["score"].values

            if len(np.unique(y_true)) < 2:
                f1 = precision = recall = roc_auc = float("nan")
            else:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                try:
                    roc_auc = roc_auc_score(y_true, scores_sub)
                except Exception:
                    roc_auc = float("nan")

            rows.append({
                "model_name": model_name,
                "scenario_family": family,
                "n_windows": len(sub),
                "n_anomaly": int(y_true.sum()),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "roc_auc": round(roc_auc, 4),
            })

    return pd.DataFrame(rows)


# ─── zero-day holdout ─────────────────────────────────────────────────────────

def compute_zero_day_holdout(scores_df: pd.DataFrame,
                              meta_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate on zero-day scenarios.

    Zero-day definition: test-split attacked-source windows whose scenario_id
    did NOT appear in the validation split. Since training uses only clean-source
    normal windows (y_anomaly==0, source_dataset=='clean'), the threshold is the
    only place where anomaly patterns are implicitly observed — through the val
    split. Zero-day windows are therefore test windows from scenarios completely
    absent from val.
    """
    # Scenarios seen during threshold selection (val split)
    val_meta = meta_df[meta_df["split"] == "val"]
    seen_scenarios = set(val_meta["scenario_id"].unique())
    seen_scenarios.discard("none")  # "none" is not an attack scenario

    test_meta = meta_df[meta_df["split"] == "test"].copy()

    # Zero-day: test attacked-source windows with scenario_id not in val
    has_source = "source_dataset" in test_meta.columns
    if has_source:
        attacked_test = test_meta[test_meta["source_dataset"] == "attacked"]
    else:
        attacked_test = test_meta[test_meta["y_anomaly"] == 1]

    attacked_test = attacked_test.copy()
    attacked_test["is_zero_day"] = ~attacked_test["scenario_id"].isin(seen_scenarios)
    zero_day = attacked_test[attacked_test["is_zero_day"]]

    if len(zero_day) == 0:
        print("  evaluate: No zero-day scenarios found (all test attacked scenarios in val)")
        zero_day = attacked_test  # fall back to all test attacked windows

    rows = []
    for model_name in scores_df["model_name"].unique():
        model_scores = scores_df[
            (scores_df["model_name"] == model_name) &
            (scores_df["config_id"] == "primary")
        ]
        if model_scores.empty:
            continue
        merged = zero_day.merge(model_scores, on="window_id", how="inner")
        if len(merged) == 0:
            continue

        y_true = merged["y_anomaly"].values
        y_pred = merged["y_pred"].values
        scores_sub = merged["score"].values

        if len(np.unique(y_true)) < 2:
            f1 = precision = recall = roc_auc = float("nan")
        else:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            try:
                roc_auc = roc_auc_score(y_true, scores_sub)
            except Exception:
                roc_auc = float("nan")

        rows.append({
            "model_name": model_name,
            "n_zero_day_windows": len(merged),
            "n_anomaly": int(y_true.sum()),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "roc_auc": round(roc_auc, 4),
            "zero_day_definition": "test windows from scenario_ids unseen in training split",
        })

    return pd.DataFrame(rows)


# ─── detection latency ────────────────────────────────────────────────────────

def compute_detection_latency(scores_df: pd.DataFrame,
                               meta_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate detection latency for each model.

    Latency = number of windows after anomaly onset before first TP detection.
    For each attacked scenario in test: find first window where y_anomaly=1,
    then find the first window where y_pred=1, compute gap in windows.
    """
    test_meta = meta_df[meta_df["split"] == "test"].copy()
    test_meta = test_meta.sort_values(["scenario_id", "window_id"])

    rows = []
    for model_name in scores_df["model_name"].unique():
        model_scores = scores_df[
            (scores_df["model_name"] == model_name) &
            (scores_df["config_id"] == "primary")
        ]
        if model_scores.empty:
            continue
        merged = test_meta.merge(model_scores, on="window_id", how="inner")
        if merged.empty:
            continue

        latencies = []
        for scen_id, grp in merged.groupby("scenario_id"):
            grp_sorted = grp.sort_values("window_id")
            y_true_arr = grp_sorted["y_anomaly"].values
            y_pred_arr = grp_sorted["y_pred"].values

            if y_true_arr.sum() == 0:
                continue  # normal scenario

            first_anomaly = np.argmax(y_true_arr == 1)
            # Look for first TP detection at or after onset
            tp_indices = np.where((y_pred_arr == 1) & (y_true_arr == 1))[0]
            tp_after_onset = tp_indices[tp_indices >= first_anomaly]
            if len(tp_after_onset) == 0:
                # Missed entirely
                latencies.append(len(grp_sorted) - first_anomaly)
            else:
                latencies.append(int(tp_after_onset[0] - first_anomaly))

        if latencies:
            rows.append({
                "model_name": model_name,
                "n_attacked_scenarios": len(latencies),
                "mean_latency_windows": round(float(np.mean(latencies)), 2),
                "median_latency_windows": round(float(np.median(latencies)), 2),
                "min_latency_windows": int(np.min(latencies)),
                "max_latency_windows": int(np.max(latencies)),
                "zero_latency_count": int(np.sum(np.array(latencies) == 0)),
            })

    return pd.DataFrame(rows)


# ─── ensembles ────────────────────────────────────────────────────────────────

def compute_ensembles(scores_df: pd.DataFrame,
                      meta_df: pd.DataFrame,
                      metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build 4 ensembles and evaluate on test split."""
    test_meta = meta_df[meta_df["split"] == "test"].copy()
    val_meta = meta_df[meta_df["split"] == "val"].copy()

    def get_model_scores(model_name, split_meta):
        s = scores_df[
            (scores_df["model_name"] == model_name) &
            (scores_df["config_id"] == "primary")
        ]
        merged = split_meta.merge(s, on="window_id", how="inner")
        merged = merged.sort_values("window_id")
        return merged

    def safe_model(name):
        prow = metrics_df[
            (metrics_df["model_name"] == name) &
            (metrics_df["config_id"] == "primary") &
            (metrics_df["status"] == "PASS")
        ]
        return len(prow) > 0

    # Get best-3 models by val F1 from metrics
    primary_pass = metrics_df[
        (metrics_df["config_id"] == "primary") &
        (metrics_df["status"] == "PASS")
    ].sort_values("f1", ascending=False)
    top3_models = primary_pass["model_name"].head(3).tolist()

    rows = []

    # Ensemble 1: threshold OR mlp
    if safe_model("threshold_baseline") and safe_model("mlp_autoencoder"):
        te_thr = get_model_scores("threshold_baseline", test_meta)
        te_mlp = get_model_scores("mlp_autoencoder", test_meta)
        common = set(te_thr["window_id"]) & set(te_mlp["window_id"])
        if common:
            te_thr_c = te_thr[te_thr["window_id"].isin(common)].sort_values("window_id")
            te_mlp_c = te_mlp[te_mlp["window_id"].isin(common)].sort_values("window_id")
            y_true = te_thr_c["y_true"].values
            y_pred_or = ((te_thr_c["y_pred"].values == 1) | (te_mlp_c["y_pred"].values == 1)).astype(int)
            score_max = np.maximum(te_thr_c["score"].values, te_mlp_c["score"].values)
            metrics = compute_metrics(y_true, y_pred_or, score_max)
            rows.append({
                "ensemble_name": "threshold_OR_mlp",
                "components": "threshold_baseline,mlp_autoencoder",
                "method": "OR",
                **metrics,
            })

    # Ensemble 2: threshold OR isolation_forest
    if safe_model("threshold_baseline") and safe_model("isolation_forest"):
        te_thr = get_model_scores("threshold_baseline", test_meta)
        te_if = get_model_scores("isolation_forest", test_meta)
        common = set(te_thr["window_id"]) & set(te_if["window_id"])
        if common:
            te_thr_c = te_thr[te_thr["window_id"].isin(common)].sort_values("window_id")
            te_if_c = te_if[te_if["window_id"].isin(common)].sort_values("window_id")
            y_true = te_thr_c["y_true"].values
            y_pred_or = ((te_thr_c["y_pred"].values == 1) | (te_if_c["y_pred"].values == 1)).astype(int)
            score_max = np.maximum(te_thr_c["score"].values, te_if_c["score"].values)
            metrics = compute_metrics(y_true, y_pred_or, score_max)
            rows.append({
                "ensemble_name": "threshold_OR_isolation_forest",
                "components": "threshold_baseline,isolation_forest",
                "method": "OR",
                **metrics,
            })

    # Ensemble 3: best-3 majority vote
    if len(top3_models) >= 3:
        te_data = [get_model_scores(m, test_meta) for m in top3_models]
        # Find common window_ids
        common = set(te_data[0]["window_id"])
        for td in te_data[1:]:
            common &= set(td["window_id"])
        if common:
            preds = []
            scores_list = []
            y_true = None
            for td in te_data:
                tc = td[td["window_id"].isin(common)].sort_values("window_id")
                preds.append(tc["y_pred"].values)
                scores_list.append(tc["score"].values)
                if y_true is None:
                    y_true = tc["y_true"].values
            vote = np.array(preds).sum(axis=0)
            y_pred_maj = (vote >= 2).astype(int)
            score_avg = np.mean(scores_list, axis=0)
            metrics = compute_metrics(y_true, y_pred_maj, score_avg)
            rows.append({
                "ensemble_name": "best3_majority_vote",
                "components": ",".join(top3_models),
                "method": "majority_vote",
                **metrics,
            })

    # Ensemble 4: best-learned (logistic regression on val scores)
    avail_models = primary_pass["model_name"].tolist()
    if len(avail_models) >= 2:
        # Collect val scores for all available models
        val_score_dfs = []
        for m in avail_models:
            vs = scores_df[
                (scores_df["model_name"] == m) &
                (scores_df["config_id"] == "primary")
            ]
            vm = val_meta.merge(vs, on="window_id", how="inner")
            if len(vm) > 0:
                vm = vm.sort_values("window_id")[["window_id", "score", "y_true"]].copy()
                vm = vm.rename(columns={"score": f"score_{m}"})
                val_score_dfs.append(vm)

        if len(val_score_dfs) >= 2:
            val_merged = val_score_dfs[0][["window_id", "y_true"]].copy()
            for vsdf in val_score_dfs:
                score_col = [c for c in vsdf.columns if c.startswith("score_")][0]
                val_merged = val_merged.merge(
                    vsdf[["window_id", score_col]], on="window_id", how="inner")

            # Collect test scores
            te_score_dfs = []
            for m in avail_models:
                ts = scores_df[
                    (scores_df["model_name"] == m) &
                    (scores_df["config_id"] == "primary")
                ]
                tm = test_meta.merge(ts, on="window_id", how="inner")
                if len(tm) > 0:
                    tm = tm.sort_values("window_id")[["window_id", "score", "y_true"]].copy()
                    tm = tm.rename(columns={"score": f"score_{m}"})
                    te_score_dfs.append(tm)

            if len(te_score_dfs) >= 2:
                te_merged = te_score_dfs[0][["window_id", "y_true"]].copy()
                for tsdf in te_score_dfs:
                    score_col = [c for c in tsdf.columns if c.startswith("score_")][0]
                    te_merged = te_merged.merge(
                        tsdf[["window_id", score_col]], on="window_id", how="inner")

                score_cols = [c for c in val_merged.columns if c.startswith("score_")]
                X_val_ens = val_merged[score_cols].values
                y_val_ens = val_merged["y_true"].values
                X_te_ens = te_merged[score_cols].values
                y_te_ens = te_merged["y_true"].values

                if len(np.unique(y_val_ens)) >= 2:
                    lr = LogisticRegression(max_iter=1000, random_state=42)
                    lr.fit(X_val_ens, y_val_ens)
                    y_pred_lr = lr.predict(X_te_ens)
                    score_lr = lr.predict_proba(X_te_ens)[:, 1]
                    metrics = compute_metrics(y_te_ens, y_pred_lr, score_lr)
                    rows.append({
                        "ensemble_name": "best_learned_lr",
                        "components": ",".join(avail_models[:len(score_cols)]),
                        "method": "logistic_regression",
                        **metrics,
                    })

    return pd.DataFrame(rows)


# ─── reports ──────────────────────────────────────────────────────────────────

def _write_zero_day_report(zd_df: pd.DataFrame, out_path: Path):
    lines = [
        "# Zero-Day Holdout Evaluation Report",
        "",
        "## Definition",
        "Zero-day windows: test-split windows from scenario_ids with no representation in the training split.",
        "These scenarios are completely unseen — neither their normal nor attacked windows participated in threshold selection.",
        "",
        "## Results by Model",
        "",
        "| Model | N Windows | N Anomaly | F1 | Precision | Recall | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in zd_df.sort_values("f1", ascending=False).iterrows():
        lines.append(
            f"| {row['model_name']} | {row['n_zero_day_windows']} | "
            f"{row['n_anomaly']} | {row['f1']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['roc_auc']:.4f} |"
        )
    lines += ["", "## Notes", "- Threshold was selected on the primary validation split only.",
              "- Zero-day results reflect true generalization to unseen attack scenarios."]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_ensemble_report(ens_df: pd.DataFrame, out_path: Path):
    lines = [
        "# Ensemble Model Evaluation Report",
        "",
        "## Ensembles Evaluated",
        "",
        "| Ensemble | Components | Method | F1 | Precision | Recall | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in ens_df.sort_values("f1", ascending=False).iterrows():
        lines.append(
            f"| {row['ensemble_name']} | {row['components']} | {row['method']} | "
            f"{row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['roc_auc']:.4f} |"
        )
    lines += ["", "## Notes",
              "- OR gate ensembles: predict anomaly if either component predicts anomaly.",
              "- Majority vote: predict anomaly if ≥2/3 top models predict anomaly.",
              "- best_learned_lr: logistic regression trained on val scores from all PASS models."]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_all_dirs()
    t0 = time.time()
    print("\n" + "="*60)
    print("  PHASE 1 MODEL EVALUATION")
    print("="*60)

    scores_df = _load_scores()
    metrics_df = _load_metrics()
    meta_df = _load_windows_meta()

    if scores_df is None or metrics_df is None:
        print("  ERROR: model_scores_all.csv or model_metrics_all.csv missing.")
        print("  Run train_all_models.py first.")
        return

    if meta_df is None:
        print("  ERROR: windows_all.parquet not found.")
        return

    print(f"  Loaded: {len(scores_df)} score rows, {len(metrics_df)} metric rows")
    print(f"  Window meta: {len(meta_df)} windows")

    # 1. Scenario family metrics
    print("\n  Computing scenario family metrics...")
    fam_df = compute_scenario_family_metrics(scores_df, meta_df)
    fam_path = RESULTS_DIR / "scenario_family_metrics.csv"
    fam_df.to_csv(fam_path, index=False)
    print(f"    Saved: {fam_path} ({len(fam_df)} rows)")

    # 2. Zero-day holdout
    print("\n  Computing zero-day holdout metrics...")
    zd_df = compute_zero_day_holdout(scores_df, meta_df)
    zd_path = RESULTS_DIR / "zero_day_holdout_metrics.csv"
    zd_df.to_csv(zd_path, index=False)
    print(f"    Saved: {zd_path} ({len(zd_df)} rows)")
    _write_zero_day_report(zd_df, RESULTS_DIR / "ZERO_DAY_HOLDOUT_REPORT.md")

    # 3. Detection latency
    print("\n  Computing detection latency...")
    lat_df = compute_detection_latency(scores_df, meta_df)
    lat_path = RESULTS_DIR / "detection_latency_metrics.csv"
    lat_df.to_csv(lat_path, index=False)
    print(f"    Saved: {lat_path} ({len(lat_df)} rows)")

    # 4. Ensembles
    print("\n  Computing ensembles...")
    ens_df = compute_ensembles(scores_df, meta_df, metrics_df)
    if len(ens_df) > 0:
        ens_path = RESULTS_DIR / "ensemble_metrics.csv"
        ens_df.to_csv(ens_path, index=False)
        print(f"    Saved: {ens_path} ({len(ens_df)} ensembles)")
        _write_ensemble_report(ens_df, RESULTS_DIR / "ENSEMBLE_REPORT.md")
    else:
        print("    No ensembles computed (insufficient PASS models)")

    # 5. Finalize results (best CSV, thresholds, confusion matrices, recompute audit)
    print("\n  Running result finalization...")
    try:
        from finalize_results import main as finalize_main
        finalize_main()
    except Exception as e:
        print(f"  WARNING: finalize_results failed: {e}")

    elapsed = time.time() - t0
    print(f"\n  Evaluation complete in {elapsed:.1f}s")

    # Print top models
    primary_pass = metrics_df[
        (metrics_df.get("config_id", pd.Series(dtype=str)) == "primary") &
        (metrics_df.get("status", pd.Series(dtype=str)) == "PASS")
    ].sort_values("f1", ascending=False) if "config_id" in metrics_df.columns else pd.DataFrame()
    if not primary_pass.empty:
        print(f"\n  Top models (primary, PASS) by F1:")
        for _, r in primary_pass.head(5).iterrows():
            print(f"    {r['model_name']:30s} F1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
