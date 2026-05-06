"""
Phase 1 Figure Generation (17 figures from real CSV data).

Reads model_metrics_all.csv, model_scores_all.csv, scenario_family_metrics.csv,
detection_latency_metrics.csv, ensemble_metrics.csv, and loss curve CSVs.

Original 11 figures:
  1.  model_comparison_f1.png           — bar chart F1 by model
  2.  model_comparison_roc_auc.png      — bar chart ROC-AUC by model
  3.  pr_curves.png                     — P-R curves for each model
  4.  roc_curves.png                    — ROC curves for each model
  5.  threshold_sweep.png               — threshold vs F1/P/R for best model
  6.  scenario_family_heatmap.png       — F1 heatmap: model × scenario family
  7.  detection_latency.png             — latency bar per model
  8.  ensemble_comparison.png           — ensemble vs individual bar chart
  9.  zero_day_holdout.png              — F1 on zero-day vs full test
  10. training_loss_curves.png          — train/val loss curves (deep models)
  11. window_split_distribution.png     — window count by split and class

Added 6 figures:
  12. model_comparison_pr_auc.png        — bar chart PR-AUC by model
  13. precision_recall_f1_grouped.png    — grouped bar: P / R / F1 per model
  14. score_distribution_normal_vs_anomaly.png — KDE of scores per model
  15. score_timeline_anomaly_windows.png — anomaly score vs window index (test)
  16. scenario_family_f1.png             — bar chart F1 per scenario family (best model)
  17. confusion_matrix_best_model.png    — confusion matrix heatmap (best model)

Writes reports/PLOT_AUDIT_REPORT.md listing generated vs missing figures.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import RESULTS_DIR, WEIGHTS_DIR, FIGURES_DIR, WINDOWS_ALL_PARQUET, ensure_all_dirs

warnings.filterwarnings("ignore")
PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _savefig(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path.name}")


def _load_primary_metrics():
    p = RESULTS_DIR / "model_metrics_all.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[(df["config_id"] == "primary") & (df["status"] == "PASS")]


def _load_scores():
    p = RESULTS_DIR / "model_scores_all.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ─── Figure 1: Model comparison F1 ───────────────────────────────────────────

def fig_model_comparison_f1(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return
    df = metrics_df.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["model_name"], df["f1"],
                  color=PALETTE[:len(df)], edgecolor="white")
    for bar, val in zip(bars, df["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison — F1 Score (Primary Config, Test Split)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _savefig(fig, "model_comparison_f1.png")


# ─── Figure 2: Model comparison ROC-AUC ──────────────────────────────────────

def fig_model_comparison_roc_auc(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return
    df = metrics_df.sort_values("roc_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["model_name"], df["roc_auc"],
                  color=PALETTE[:len(df)], edgecolor="white")
    for bar, val in zip(bars, df["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Model Comparison — ROC-AUC (Primary Config, Test Split)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _savefig(fig, "model_comparison_roc_auc.png")


# ─── Figure 3: PR curves ──────────────────────────────────────────────────────

def fig_pr_curves(scores_df: pd.DataFrame, metrics_df: pd.DataFrame):
    if scores_df.empty or metrics_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    pass_models = metrics_df["model_name"].tolist()
    for i, model_name in enumerate(pass_models):
        sub = scores_df[
            (scores_df["model_name"] == model_name) &
            (scores_df["config_id"] == "primary")
        ]
        if sub.empty:
            continue
        y_true = sub["y_true"].values
        scores = sub["score"].values
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = metrics_df.loc[metrics_df["model_name"] == model_name, "pr_auc"].values
        auc_val = pr_auc[0] if len(pr_auc) > 0 else 0
        ax.plot(rec, prec, label=f"{model_name} (AUC={auc_val:.3f})",
                color=PALETTE[i % len(PALETTE)], linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Test Split, Primary Config)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _savefig(fig, "pr_curves.png")


# ─── Figure 4: ROC curves ─────────────────────────────────────────────────────

def fig_roc_curves(scores_df: pd.DataFrame, metrics_df: pd.DataFrame):
    if scores_df.empty or metrics_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    pass_models = metrics_df["model_name"].tolist()
    for i, model_name in enumerate(pass_models):
        sub = scores_df[
            (scores_df["model_name"] == model_name) &
            (scores_df["config_id"] == "primary")
        ]
        if sub.empty:
            continue
        y_true = sub["y_true"].values
        scores = sub["score"].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_vals = metrics_df.loc[metrics_df["model_name"] == model_name, "roc_auc"].values
        auc_val = auc_vals[0] if len(auc_vals) > 0 else 0
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})",
                color=PALETTE[i % len(PALETTE)], linewidth=1.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Split, Primary Config)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _savefig(fig, "roc_curves.png")


# ─── Figure 5: Threshold sweep ────────────────────────────────────────────────

def fig_threshold_sweep(scores_df: pd.DataFrame, metrics_df: pd.DataFrame):
    if scores_df.empty or metrics_df.empty:
        return
    # Use best model by F1
    best_model = metrics_df.sort_values("f1", ascending=False)["model_name"].iloc[0]
    sub = scores_df[
        (scores_df["model_name"] == best_model) &
        (scores_df["config_id"] == "primary")
    ]
    if sub.empty:
        return
    y_true = sub["y_true"].values
    scores = sub["score"].values
    if len(np.unique(y_true)) < 2:
        return

    thresholds = np.percentile(scores, np.linspace(1, 99, 99))
    f1s, precs, recs = [], [], []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f1s, label="F1", linewidth=2)
    ax.plot(thresholds, precs, label="Precision", linewidth=1.5, linestyle="--")
    ax.plot(thresholds, recs, label="Recall", linewidth=1.5, linestyle=":")
    best_idx = int(np.argmax(f1s))
    ax.axvline(thresholds[best_idx], color="red", linestyle="-.", linewidth=1,
               label=f"Best thr={thresholds[best_idx]:.4f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sweep — {best_model} (Test Split)")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "threshold_sweep.png")


# ─── Figure 6: Scenario family heatmap ───────────────────────────────────────

def fig_scenario_family_heatmap():
    p = RESULTS_DIR / "scenario_family_metrics.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    pivot = df.pivot_table(index="model_name", columns="scenario_family",
                           values="f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), max(5, len(pivot) * 0.6)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.2 < val < 0.8 else "white")
    plt.colorbar(im, ax=ax, label="F1 Score")
    ax.set_title("F1 by Model × Scenario Family (Test Split)")
    fig.tight_layout()
    _savefig(fig, "scenario_family_heatmap.png")


# ─── Figure 7: Detection latency ─────────────────────────────────────────────

def fig_detection_latency():
    p = RESULTS_DIR / "detection_latency_metrics.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    df_sorted = df.sort_values("mean_latency_windows")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_sorted))]
    bars = ax.bar(df_sorted["model_name"], df_sorted["mean_latency_windows"],
                  color=colors, edgecolor="white")
    for bar, val in zip(bars, df_sorted["mean_latency_windows"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Latency (windows after onset)")
    ax.set_title("Detection Latency by Model (Test Split)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _savefig(fig, "detection_latency.png")


# ─── Figure 8: Ensemble comparison ───────────────────────────────────────────

def fig_ensemble_comparison(metrics_df: pd.DataFrame):
    ens_p = RESULTS_DIR / "ensemble_metrics.csv"
    if not ens_p.exists() or metrics_df.empty:
        return
    ens_df = pd.read_csv(ens_p)
    if ens_df.empty:
        return

    # Combine individual and ensemble F1s
    ind_df = metrics_df[["model_name", "f1"]].copy()
    ind_df["type"] = "individual"
    ind_df = ind_df.rename(columns={"model_name": "name"})

    ens_sub = ens_df[["ensemble_name", "f1"]].copy()
    ens_sub["type"] = "ensemble"
    ens_sub = ens_sub.rename(columns={"ensemble_name": "name"})

    combined = pd.concat([ind_df, ens_sub], ignore_index=True)
    combined = combined.sort_values("f1", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3" if t == "individual" else "#FF9800" for t in combined["type"]]
    bars = ax.bar(combined["name"], combined["f1"], color=colors, edgecolor="white")
    for bar, val in zip(bars, combined["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model / Ensemble")
    ax.set_ylabel("F1 Score")
    ax.set_title("Individual vs Ensemble F1 (Test Split)")
    ax.tick_params(axis="x", rotation=40)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2196F3", label="Individual"),
                        Patch(color="#FF9800", label="Ensemble")])
    fig.tight_layout()
    _savefig(fig, "ensemble_comparison.png")


# ─── Figure 9: Zero-day holdout ───────────────────────────────────────────────

def fig_zero_day_holdout(metrics_df: pd.DataFrame):
    zd_p = RESULTS_DIR / "zero_day_holdout_metrics.csv"
    if not zd_p.exists() or metrics_df.empty:
        return
    zd_df = pd.read_csv(zd_p)
    if zd_df.empty:
        return

    # Compare zero-day F1 vs full-test F1
    merged = metrics_df[["model_name", "f1"]].rename(columns={"f1": "f1_full_test"})
    merged = merged.merge(zd_df[["model_name", "f1"]].rename(columns={"f1": "f1_zero_day"}),
                          on="model_name", how="inner")
    if merged.empty:
        return

    x = np.arange(len(merged))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, merged["f1_full_test"], width, label="Full Test", color="#2196F3", edgecolor="white")
    ax.bar(x + width / 2, merged["f1_zero_day"], width, label="Zero-Day", color="#F44336", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["model_name"], rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Full Test vs Zero-Day F1 by Model")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "zero_day_holdout.png")


# ─── Figure 10: Training loss curves ─────────────────────────────────────────

def fig_training_loss_curves():
    MODEL_NAMES = [
        ("mlp_autoencoder", "MLP"),
        ("cnn_autoencoder", "CNN"),
        ("lstm_autoencoder", "LSTM"),
        ("gru_autoencoder", "GRU"),
        ("transformer_autoencoder", "Transformer"),
        ("tcn_autoencoder", "TCN"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    has_any = False
    for ax, (folder_name, display_name) in zip(axes, MODEL_NAMES):
        # Try to find loss curve file
        loss_file = None
        for suffix in [folder_name, folder_name.replace("_autoencoder", "")]:
            candidate = WEIGHTS_DIR / folder_name / f"{suffix.split('_')[0]}_loss_curve.csv"
            # Also try primary suffix
            for fname in [
                f"{folder_name.split('_')[0]}_loss_curve.csv",
                f"{folder_name.split('_')[0]}_loss_curve_primary.csv",
            ]:
                p = WEIGHTS_DIR / folder_name / fname
                if p.exists():
                    loss_file = p
                    break
            if loss_file:
                break

        if loss_file is None:
            ax.set_visible(False)
            continue

        lc = pd.read_csv(loss_file)
        has_any = True
        ax.plot(lc["train_loss"], label="Train", linewidth=1.5)
        ax.plot(lc["val_loss"], label="Val", linewidth=1.5, linestyle="--")
        ax.set_title(f"{display_name} Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)

    if not has_any:
        plt.close(fig)
        return

    fig.suptitle("Training Loss Curves", fontsize=13)
    fig.tight_layout()
    _savefig(fig, "training_loss_curves.png")


# ─── Figure 11: Window split distribution ────────────────────────────────────

def fig_window_split_distribution():
    if not WINDOWS_ALL_PARQUET.exists():
        return
    df = pd.read_parquet(WINDOWS_ALL_PARQUET, columns=["split", "y_anomaly"])
    counts = df.groupby(["split", "y_anomaly"]).size().unstack(fill_value=0)
    counts.columns = ["Normal", "Anomaly"]
    splits_order = [s for s in ["train", "val", "test"] if s in counts.index]
    counts = counts.loc[splits_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(counts))
    width = 0.35
    ax.bar(x - width / 2, counts["Normal"], width, label="Normal", color="#4CAF50", edgecolor="white")
    ax.bar(x + width / 2, counts["Anomaly"], width, label="Anomaly", color="#F44336", edgecolor="white")
    for i, (norm, anom) in enumerate(zip(counts["Normal"], counts["Anomaly"])):
        ax.text(i - width / 2, norm + 200, f"{norm:,}", ha="center", fontsize=8)
        ax.text(i + width / 2, anom + 200, f"{anom:,}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(splits_order)
    ax.set_ylabel("Window Count")
    ax.set_title("Window Distribution by Split and Class")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "window_split_distribution.png")


# ─── Figure 12: Model comparison PR-AUC ──────────────────────────────────────

def fig_model_comparison_pr_auc(metrics_df: pd.DataFrame):
    if metrics_df.empty or "pr_auc" not in metrics_df.columns:
        return
    df = metrics_df.dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=False)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["model_name"], df["pr_auc"],
                  color=PALETTE[:len(df)], edgecolor="white")
    for bar, val in zip(bars, df["pr_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Model Comparison — PR-AUC (Primary Config, Test Split)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _savefig(fig, "model_comparison_pr_auc.png")


# ─── Figure 13: Precision / Recall / F1 grouped bar ──────────────────────────

def fig_precision_recall_f1_grouped(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return
    cols = ["model_name", "precision", "recall", "f1"]
    missing = [c for c in cols if c not in metrics_df.columns]
    if missing:
        return
    df = metrics_df[cols].dropna().sort_values("f1", ascending=False)
    if df.empty:
        return
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.0), 5))
    ax.bar(x - width, df["precision"], width, label="Precision", color="#2196F3", edgecolor="white")
    ax.bar(x,         df["f1"],        width, label="F1",        color="#4CAF50", edgecolor="white")
    ax.bar(x + width, df["recall"],    width, label="Recall",    color="#FF9800", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Precision / F1 / Recall by Model (Primary Config, Test Split)")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "precision_recall_f1_grouped.png")


# ─── Figure 14: Score distribution normal vs anomaly ─────────────────────────

def fig_score_distribution_normal_vs_anomaly(scores_df: pd.DataFrame,
                                              metrics_df: pd.DataFrame):
    if scores_df.empty or metrics_df.empty:
        return
    pass_models = metrics_df["model_name"].tolist()
    n = len(pass_models)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, model_name in zip(axes, pass_models):
        sub = scores_df[
            (scores_df["model_name"] == model_name) &
            (scores_df["config_id"] == "primary")
        ]
        if sub.empty:
            ax.set_visible(False)
            continue
        normal_scores = sub.loc[sub["y_true"] == 0, "score"].values
        anomaly_scores = sub.loc[sub["y_true"] == 1, "score"].values
        if len(normal_scores) == 0 and len(anomaly_scores) == 0:
            ax.set_visible(False)
            continue
        if len(normal_scores) > 0:
            ax.hist(normal_scores, bins=50, alpha=0.6, color="#4CAF50",
                    label=f"Normal (n={len(normal_scores)})", density=True)
        if len(anomaly_scores) > 0:
            ax.hist(anomaly_scores, bins=50, alpha=0.6, color="#F44336",
                    label=f"Anomaly (n={len(anomaly_scores)})", density=True)
        ax.set_title(model_name, fontsize=9)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Score Distribution: Normal vs Anomaly (Test Split)", fontsize=12)
    fig.tight_layout()
    _savefig(fig, "score_distribution_normal_vs_anomaly.png")


# ─── Figure 15: Score timeline anomaly windows ───────────────────────────────

def fig_score_timeline_anomaly_windows(scores_df: pd.DataFrame,
                                        metrics_df: pd.DataFrame):
    if scores_df.empty or metrics_df.empty:
        return
    best_model = metrics_df.sort_values("f1", ascending=False)["model_name"].iloc[0]
    sub = scores_df[
        (scores_df["model_name"] == best_model) &
        (scores_df["config_id"] == "primary")
    ].reset_index(drop=True)
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    normal_idx = sub[sub["y_true"] == 0].index
    anomaly_idx = sub[sub["y_true"] == 1].index
    ax.scatter(normal_idx,  sub.loc[normal_idx,  "score"],
               s=4, alpha=0.4, color="#4CAF50", label="Normal", rasterized=True)
    ax.scatter(anomaly_idx, sub.loc[anomaly_idx, "score"],
               s=6, alpha=0.7, color="#F44336", label="Anomaly", rasterized=True)

    thr_vals = metrics_df.loc[metrics_df["model_name"] == best_model, "threshold"]
    if len(thr_vals) > 0 and not pd.isna(thr_vals.values[0]):
        ax.axhline(thr_vals.values[0], color="black", linestyle="--",
                   linewidth=1, label=f"Threshold={thr_vals.values[0]:.4f}")

    ax.set_xlabel("Window Index (test set)")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(f"Score Timeline — {best_model} (Test Split)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, "score_timeline_anomaly_windows.png")


# ─── Figure 16: Scenario family F1 (best model) ──────────────────────────────

def fig_scenario_family_f1(metrics_df: pd.DataFrame):
    p = RESULTS_DIR / "scenario_family_metrics.csv"
    if not p.exists() or metrics_df.empty:
        return
    sf_df = pd.read_csv(p)
    if sf_df.empty:
        return
    best_model = metrics_df.sort_values("f1", ascending=False)["model_name"].iloc[0]
    sub = sf_df[sf_df["model_name"] == best_model]
    if sub.empty:
        return
    sub = sub.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(sub) * 1.0), 5))
    bars = ax.bar(sub["scenario_family"], sub["f1"],
                  color=PALETTE[:len(sub)], edgecolor="white")
    for bar, val in zip(bars, sub["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Scenario Family")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1 by Scenario Family — {best_model} (Best Model, Test Split)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _savefig(fig, "scenario_family_f1.png")


# ─── Figure 17: Confusion matrix best model ──────────────────────────────────

def fig_confusion_matrix_best_model(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        return
    cm_cols = ["tp", "fp", "tn", "fn"]
    if not all(c in metrics_df.columns for c in cm_cols):
        return
    best_row = metrics_df.sort_values("f1", ascending=False).iloc[0]
    tp = int(best_row["tp"]) if not pd.isna(best_row["tp"]) else 0
    fp = int(best_row["fp"]) if not pd.isna(best_row["fp"]) else 0
    tn = int(best_row["tn"]) if not pd.isna(best_row["tn"]) else 0
    fn = int(best_row["fn"]) if not pd.isna(best_row["fn"]) else 0
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ["Normal", "Anomaly"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Normal", "Predicted Anomaly"])
    ax.set_yticklabels(["Actual Normal", "Actual Anomaly"])
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total if total > 0 else 0
            ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax)
    model_name = best_row["model_name"]
    f1_val = best_row["f1"]
    ax.set_title(f"Confusion Matrix — {model_name}\nF1={f1_val:.4f} (Test Split)")
    fig.tight_layout()
    _savefig(fig, "confusion_matrix_best_model.png")


# ─── PLOT_AUDIT_REPORT.md ─────────────────────────────────────────────────────

def write_plot_audit_report(expected_figures: list):
    from datetime import datetime
    from common_paths import REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    generated = {f.name for f in FIGURES_DIR.glob("*.png")}
    lines = [
        "# Plot Audit Report",
        f"_Generated: {now}_",
        "",
        f"## Summary",
        f"- Expected figures: {len(expected_figures)}",
        f"- Generated: {len([f for f in expected_figures if f in generated])}",
        f"- Missing: {len([f for f in expected_figures if f not in generated])}",
        "",
        "## Figure Status",
        "",
        "| # | Filename | Status |",
        "|---|----------|--------|",
    ]
    for i, fname in enumerate(expected_figures, 1):
        status = "GENERATED" if fname in generated else "MISSING"
        lines.append(f"| {i} | {fname} | {status} |")

    extra = sorted(generated - set(expected_figures))
    if extra:
        lines += ["", "## Extra Figures (not in expected list)", ""]
        for fname in extra:
            lines.append(f"- {fname}")

    report_path = REPORTS_DIR / "PLOT_AUDIT_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {report_path.name}")


# ─── main ─────────────────────────────────────────────────────────────────────

EXPECTED_FIGURES = [
    "model_comparison_f1.png",
    "model_comparison_roc_auc.png",
    "pr_curves.png",
    "roc_curves.png",
    "threshold_sweep.png",
    "scenario_family_heatmap.png",
    "detection_latency.png",
    "ensemble_comparison.png",
    "zero_day_holdout.png",
    "training_loss_curves.png",
    "window_split_distribution.png",
    "model_comparison_pr_auc.png",
    "precision_recall_f1_grouped.png",
    "score_distribution_normal_vs_anomaly.png",
    "score_timeline_anomaly_windows.png",
    "scenario_family_f1.png",
    "confusion_matrix_best_model.png",
]


def main():
    ensure_all_dirs()
    print("\n" + "="*60)
    print("  PHASE 1 FIGURE GENERATION (17 figures)")
    print("="*60)

    metrics_df = _load_primary_metrics()
    scores_df = _load_scores()

    print(f"\n  Loaded: {len(metrics_df)} PASS primary models, "
          f"{len(scores_df)} score rows")

    print("\n  Generating figures...")
    # Original 11
    fig_model_comparison_f1(metrics_df)
    fig_model_comparison_roc_auc(metrics_df)
    fig_pr_curves(scores_df, metrics_df)
    fig_roc_curves(scores_df, metrics_df)
    fig_threshold_sweep(scores_df, metrics_df)
    fig_scenario_family_heatmap()
    fig_detection_latency()
    fig_ensemble_comparison(metrics_df)
    fig_zero_day_holdout(metrics_df)
    fig_training_loss_curves()
    fig_window_split_distribution()
    # New 6
    fig_model_comparison_pr_auc(metrics_df)
    fig_precision_recall_f1_grouped(metrics_df)
    fig_score_distribution_normal_vs_anomaly(scores_df, metrics_df)
    fig_score_timeline_anomaly_windows(scores_df, metrics_df)
    fig_scenario_family_f1(metrics_df)
    fig_confusion_matrix_best_model(metrics_df)

    write_plot_audit_report(EXPECTED_FIGURES)

    fig_files = list(FIGURES_DIR.glob("*.png"))
    print(f"\n  Figures directory: {FIGURES_DIR}")
    print(f"  Total figures: {len(fig_files)}")
    for f in sorted(fig_files):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
