"""
Phase 3 Step 5: Generate paper-quality figures from explanation scoring results.
Uses matplotlib only. 300 dpi. Does not fake values.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
RPT_DIR = ROOT / "reports"
FIG_DIR.mkdir(exist_ok=True)
RPT_DIR.mkdir(exist_ok=True)

AUDIT = []  # figure audit log

def note_audit(figname, status, reason=""):
    AUDIT.append({"figure": figname, "status": status, "reason": reason})

def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

# ── Load data ─────────────────────────────────────────────────────────────────
matrix_path  = OUT_DIR / "explanation_evidence_matrix.csv"
parsed_path  = OUT_DIR / "llm_explanations_parsed.csv"
summary_path = OUT_DIR / "explanation_score_summary.json"

data_ok = matrix_path.exists() and parsed_path.exists()
matrix_df = pd.read_csv(matrix_path) if matrix_path.exists() else pd.DataFrame()
parsed_df = pd.read_csv(parsed_path) if parsed_path.exists() else pd.DataFrame()
summary   = json.loads(summary_path.read_text()) if summary_path.exists() else {}

parsed_ok = matrix_df[matrix_df.get("parse_success", pd.Series(dtype=int)) == 1] if len(matrix_df) > 0 else pd.DataFrame()
print(f"[INFO] Matrix rows: {len(matrix_df)}, parsed OK: {len(parsed_ok)}")

CLASS_ORDER = ["normal", "physical_only", "cyber_only", "cyber_physical"]
COLORS = {
    "normal":        "#4daf4a",
    "physical_only": "#377eb8",
    "cyber_only":    "#e41a1c",
    "cyber_physical":"#ff7f00",
    "insufficient_evidence": "#984ea3",
    "": "#aaaaaa",
}

# ── Figure 1: Explanation type confusion matrix ───────────────────────────────
print("[INFO] Figure 1: confusion matrix ...")
fig_name = "explanation_type_confusion_matrix.png"

if len(parsed_ok) >= 2:
    all_types = CLASS_ORDER + ["insufficient_evidence"]
    true_col  = "true_scenario_class"
    pred_col  = "predicted_explanation_type"

    true_vals = parsed_ok[true_col].fillna("unknown")
    pred_vals = parsed_ok[pred_col].fillna("insufficient_evidence")

    labels = sorted(set(true_vals.tolist() + pred_vals.tolist()))
    n = len(labels)
    conf = np.zeros((n, n), dtype=int)
    label_idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(true_vals, pred_vals):
        ti = label_idx.get(t, 0)
        pi = label_idx.get(p, 0)
        conf[ti][pi] += 1

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(conf, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_yticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_xlabel("Predicted Explanation Type", fontsize=10)
    ax.set_ylabel("True Scenario Class", fontsize=10)
    ax.set_title("Explanation Type vs True Scenario Class\n(Smoke Run)", fontsize=11)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(conf[i][j]), ha="center", va="center",
                    fontsize=10, color="white" if conf[i][j] > conf.max()*0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_fig(fig, fig_name)
    note_audit(fig_name, "OK", f"{len(parsed_ok)} parsed explanations")
else:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, f"Insufficient parsed data\n(N={len(parsed_ok)})\nLLM may not have been available.",
            ha="center", va="center", fontsize=11, color="gray",
            transform=ax.transAxes)
    ax.set_title("Explanation Type Confusion Matrix (No Data)", fontsize=11)
    ax.axis("off")
    save_fig(fig, fig_name)
    note_audit(fig_name, "NO_DATA", f"Only {len(parsed_ok)} parsed rows")

# ── Figure 2: Evidence score by class ────────────────────────────────────────
print("[INFO] Figure 2: evidence score by class ...")
fig_name = "evidence_score_by_class.png"

if len(parsed_ok) >= 2 and "overall_evidence_score" in parsed_ok.columns:
    cls_col = "true_scenario_class"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    class_data = []
    class_labels = []
    for cls in CLASS_ORDER:
        sub = parsed_ok[parsed_ok[cls_col] == cls]["overall_evidence_score"].dropna()
        if len(sub) > 0:
            class_data.append(sub.values)
            class_labels.append(cls.replace("_", "\n"))

    if class_data:
        bp = ax.boxplot(class_data, patch_artist=True, medianprops=dict(color="black", linewidth=2))
        for patch, cls_lbl in zip(bp["boxes"], [l.replace("\n","_") for l in class_labels]):
            patch.set_facecolor(COLORS.get(cls_lbl, "#aaaaaa"))
            patch.set_alpha(0.7)
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_xlabel("True Scenario Class", fontsize=10)
        ax.set_ylabel("Overall Evidence Score (0-1)", fontsize=10)
        ax.set_title("Evidence Score Distribution by Scenario Class\n(Smoke Run)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="0.5 threshold")
        ax.legend(fontsize=9)
        fig.tight_layout()
    else:
        ax.text(0.5, 0.5, "No class data available", ha="center", va="center", transform=ax.transAxes)
    save_fig(fig, fig_name)
    note_audit(fig_name, "OK", f"{len(parsed_ok)} rows")
else:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, f"Insufficient data (N={len(parsed_ok)})\nRun LLM explanations first.",
            ha="center", va="center", fontsize=11, color="gray", transform=ax.transAxes)
    ax.axis("off")
    save_fig(fig, fig_name)
    note_audit(fig_name, "NO_DATA", f"Only {len(parsed_ok)} parsed rows")

# ── Figure 3: Evidence matrix heatmap ────────────────────────────────────────
print("[INFO] Figure 3: evidence matrix heatmap ...")
fig_name = "evidence_matrix_heatmap.png"

POSITIVE_COLS = [
    "scenario_class_match", "asset_match", "physical_signal_match",
    "cyber_state_match", "timing_match", "expected_vs_observed_match",
    "glossary_field_usage_correct", "explanation_mentions_evidence_fields",
    "timing_relationship_correct", "operator_action_relevance",
]
avail_pos_cols = [c for c in POSITIVE_COLS if c in parsed_ok.columns]

if len(parsed_ok) >= 2 and avail_pos_cols:
    cls_col = "true_scenario_class"
    present_classes = [c for c in CLASS_ORDER if c in parsed_ok[cls_col].values]
    heatmap_data = np.zeros((len(present_classes), len(avail_pos_cols)))
    for ci, cls in enumerate(present_classes):
        sub = parsed_ok[parsed_ok[cls_col] == cls]
        for cj, col in enumerate(avail_pos_cols):
            heatmap_data[ci, cj] = sub[col].mean() if len(sub) > 0 else 0.0

    col_labels = [c.replace("_match","").replace("_", "\n") for c in avail_pos_cols]
    row_labels  = [c.replace("_", "\n") for c in present_classes]

    fig, ax = plt.subplots(figsize=(max(9, len(avail_pos_cols)*0.9), max(4, len(present_classes)*0.8 + 1.5)))
    im = ax.imshow(heatmap_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(avail_pos_cols)))
    ax.set_yticks(range(len(present_classes)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=35, ha="right")
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Evidence Check Rates by Scenario Class\n(Smoke Run — 1.0 = all pass)", fontsize=11)
    for i in range(len(present_classes)):
        for j in range(len(avail_pos_cols)):
            ax.text(j, i, f"{heatmap_data[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Fraction passing check")
    fig.tight_layout()
    save_fig(fig, fig_name)
    note_audit(fig_name, "OK", f"{len(present_classes)} classes x {len(avail_pos_cols)} checks")
else:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, f"Insufficient data (N={len(parsed_ok)})", ha="center", va="center",
            fontsize=11, color="gray", transform=ax.transAxes)
    ax.axis("off")
    save_fig(fig, fig_name)
    note_audit(fig_name, "NO_DATA", f"Only {len(parsed_ok)} parsed rows")

# ── Figure 4: Unsupported claims summary ─────────────────────────────────────
print("[INFO] Figure 4: unsupported claims ...")
fig_name = "unsupported_claims_summary.png"

claim_cols = {
    "unsupported_claim_flag": "Unsupported\nclaim",
    "packet_claim_flag":      "Packet-level\nclaim",
    "field_telemetry_claim_flag": "Field telemetry\nclaim",
    "external_attacker_claim_flag": "External\nattacker claim",
    "uses_old_asset_name_flag": "Old asset\nname used",
}
avail_claim_cols = {k: v for k, v in claim_cols.items() if k in matrix_df.columns}

if len(matrix_df) > 0 and avail_claim_cols:
    counts = {v: int(matrix_df[k].sum()) for k, v in avail_claim_cols.items()}
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Count (violations)", fontsize=10)
    ax.set_title("Guardrail Violations in LLM Explanations\n(Smoke Run)", fontsize=11)
    ax.set_ylim(0, max(max(values) + 2, 4))
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ok_patch  = mpatches.Patch(color="#2ca02c", label="0 violations (pass)")
    bad_patch = mpatches.Patch(color="#d62728", label=">0 violations (fail)")
    ax.legend(handles=[ok_patch, bad_patch], fontsize=9, loc="upper right")
    fig.tight_layout()
    save_fig(fig, fig_name)
    note_audit(fig_name, "OK", f"Total violations: {sum(values)}")
else:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "No scoring data available", ha="center", va="center",
            fontsize=11, color="gray", transform=ax.transAxes)
    ax.axis("off")
    save_fig(fig, fig_name)
    note_audit(fig_name, "NO_DATA", "matrix_df empty")

# ── Figure 5: Parse success summary ──────────────────────────────────────────
print("[INFO] Figure 5: parse success ...")
fig_name = "parse_success_summary.png"

if len(parsed_df) > 0 and "parse_status" in parsed_df.columns:
    status_counts = parsed_df["parse_status"].value_counts()
    labels = list(status_counts.index)
    values = list(status_counts.values)
    status_colors = []
    for lbl in labels:
        if "OK" in lbl:
            status_colors.append("#2ca02c")
        elif "SKIP" in lbl:
            status_colors.append("#ff7f0e")
        else:
            status_colors.append("#d62728")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    ax = axes[0]
    bars = ax.bar(range(len(labels)), values, color=status_colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Parse Status Distribution", fontsize=11)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(val),
                ha="center", va="bottom", fontsize=9)

    # Pie chart
    ax2 = axes[1]
    ok_count   = sum(v for l, v in zip(labels, values) if "OK" in l)
    skip_count = sum(v for l, v in zip(labels, values) if "SKIP" in l)
    fail_count = sum(v for l, v in zip(labels, values) if "FAIL" in l)
    pie_data   = [(ok_count, "OK", "#2ca02c"), (skip_count, "Skipped", "#ff7f0e"), (fail_count, "Failed", "#d62728")]
    pie_data   = [(v, l, c) for v, l, c in pie_data if v > 0]
    if pie_data:
        ax2.pie(
            [v for v, l, c in pie_data],
            labels=[f"{l}\n({v})" for v, l, c in pie_data],
            colors=[c for v, l, c in pie_data],
            autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9}
        )
        ax2.set_title("Parse Outcome", fontsize=11)

    fig.suptitle("LLM Explanation Parse Results (Smoke Run)", fontsize=12)
    fig.tight_layout()
    save_fig(fig, fig_name)
    note_audit(fig_name, "OK", f"Total: {len(parsed_df)}")
else:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "No parsed data available", ha="center", va="center",
            fontsize=11, color="gray", transform=ax.transAxes)
    ax.axis("off")
    save_fig(fig, fig_name)
    note_audit(fig_name, "NO_DATA", "parsed_df empty")

# ── Figure audit report ───────────────────────────────────────────────────────
audit_lines = [
    "# PHASE 3 FIGURE AUDIT REPORT",
    "",
    f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "## Figure Status",
    "",
    "| Figure | Status | Notes |",
    "|---|---|---|",
]
for a in AUDIT:
    audit_lines.append(f"| {a['figure']} | {a['status']} | {a['reason']} |")

audit_lines += [
    "",
    "## Notes",
    "",
    "- `NO_DATA` figures contain a placeholder chart with an explanatory message.",
    "- Re-run after LLM explanations are available for populated figures.",
    "- All figures are 300 dpi, matplotlib only.",
    ""
]

audit_path = RPT_DIR / "PHASE3_FIGURE_AUDIT_REPORT.md"
with open(audit_path, "w", encoding="utf-8") as f:
    f.write("\n".join(audit_lines))
print(f"[INFO] Figure audit saved: {audit_path}")

ok_figs   = sum(1 for a in AUDIT if a["status"] == "OK")
data_figs = sum(1 for a in AUDIT if a["status"] == "NO_DATA")
print(f"[DONE] {len(AUDIT)} figures: {ok_figs} OK, {data_figs} no-data placeholders.")
