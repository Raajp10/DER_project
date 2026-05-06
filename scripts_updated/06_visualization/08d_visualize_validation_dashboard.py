"""Generate final validation dashboard figure."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import (
    FIGURES, FINAL_VALIDATION_JSON, PHYSICAL_CONSTRAINTS_JSON,
    CYBER_VALIDATION_JSON, CONTEXT_VALIDATION_JSON,
    CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, CYBER_ANOMALOUS_CSV,
)
from plot_utils import apply_style, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_json_safe(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating validation dashboard...")

    apply_style()
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("DER Cyber-Physical Dataset — Final Validation Dashboard", fontsize=14, y=1.01)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── 1. Validation gate PASS/FAIL ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    gates = {
        "Physical\nConstraints": load_json_safe(PHYSICAL_CONSTRAINTS_JSON).get("overall", "UNKNOWN"),
        "Cyber\nSemantic": load_json_safe(CYBER_VALIDATION_JSON).get("overall", "UNKNOWN"),
        "Context\nCausality": load_json_safe(CONTEXT_VALIDATION_JSON).get("overall", "UNKNOWN"),
        "Final\nGate": load_json_safe(FINAL_VALIDATION_JSON).get("overall", "UNKNOWN"),
    }
    colors = ["#4CAF50" if v == "PASS" else "#F44336" if v == "FAIL" else "#FF9800"
              for v in gates.values()]
    bars = ax1.bar(list(gates.keys()), [1] * len(gates), color=colors, edgecolor="white",
                   linewidth=1.5)
    for bar, (k, v) in zip(bars, gates.items()):
        ax1.text(bar.get_x() + bar.get_width() / 2, 0.5, v,
                 ha="center", va="center", fontsize=11, fontweight="bold",
                 color="white")
    ax1.set_ylim(0, 1.5)
    ax1.set_yticks([])
    ax1.set_title("Validation Gate Status", fontsize=10)

    # ── 2. Row counts ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    row_data = {}
    for label, path in [("Clean", CLEAN_PHYSICAL_CSV), ("Attacked", ATTACKED_PHYSICAL_CSV)]:
        if path.exists():
            row_data[label] = sum(1 for _ in open(path)) - 1  # subtract header
        else:
            row_data[label] = 0
    if CYBER_ANOMALOUS_CSV.exists():
        row_data["Cyber Log"] = sum(1 for _ in open(CYBER_ANOMALOUS_CSV)) - 1
    ax2.bar(list(row_data.keys()), list(row_data.values()),
            color=["#2196F3", "#FF9800", "#9C27B0"], alpha=0.85, edgecolor="white")
    ax2.set_ylabel("Row Count")
    ax2.set_title("Dataset Row Counts", fontsize=10)
    for i, (k, v) in enumerate(row_data.items()):
        ax2.text(i, v + 500, f"{v:,}", ha="center", fontsize=9)

    # ── 3. Generation method distribution ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if ATTACKED_PHYSICAL_CSV.exists():
        df = pd.read_csv(ATTACKED_PHYSICAL_CSV, usecols=["generation_method"])
        counts = df["generation_method"].value_counts()
        ax3.pie(counts.values, labels=[l[:25] for l in counts.index],
                autopct="%1.1f%%", colors=plt.cm.Pastel1.colors[:len(counts)],
                textprops={"fontsize": 8})
        ax3.set_title("Generation Method", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "No data", transform=ax3.transAxes, ha="center")

    # ── 4. Physical validation check counts ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    phys_j = load_json_safe(PHYSICAL_CONSTRAINTS_JSON)
    if phys_j:
        passed = phys_j.get("passed", 0)
        failed = phys_j.get("failed", 0)
        ax4.bar(["Passed", "Failed"], [passed, failed],
                color=["#4CAF50", "#F44336"], alpha=0.85, edgecolor="white")
        ax4.set_title("Physical Validation Checks", fontsize=10)
        ax4.set_ylabel("Check Count")
    else:
        ax4.text(0.5, 0.5, "No validation data", transform=ax4.transAxes, ha="center")

    # ── 5. Label distribution ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if CYBER_ANOMALOUS_CSV.exists():
        df_c = pd.read_csv(CYBER_ANOMALOUS_CSV,
                           usecols=["label_anomaly", "label_cyber_anomaly", "label_physical_anomaly"])
        labels_vals = {
            "Anomaly": int(df_c["label_anomaly"].sum()),
            "Cyber": int(df_c["label_cyber_anomaly"].sum()),
            "Physical": int(df_c["label_physical_anomaly"].sum()),
            "Normal": int((df_c["label_anomaly"] == 0).sum()),
        }
        ax5.bar(list(labels_vals.keys()), list(labels_vals.values()),
                color=plt.cm.Set2.colors[:4], alpha=0.85, edgecolor="white")
        ax5.set_title("Cyber Log Label Distribution", fontsize=10)
        ax5.set_ylabel("Event Count")
    else:
        ax5.text(0.5, 0.5, "No cyber log", transform=ax5.transAxes, ha="center")

    # ── 6. Protocol claim level ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if CYBER_ANOMALOUS_CSV.exists():
        df_p = pd.read_csv(CYBER_ANOMALOUS_CSV, usecols=["protocol_claim_level"])
        counts_p = df_p["protocol_claim_level"].value_counts()
        ax6.bar([l[:20] for l in counts_p.index], counts_p.values,
                color=plt.cm.Set3.colors[:len(counts_p)], alpha=0.85, edgecolor="white")
        ax6.set_title("Protocol Claim Level", fontsize=10)
        ax6.set_ylabel("Event Count")
        ax6.tick_params(axis="x", rotation=20)
    else:
        ax6.text(0.5, 0.5, "No cyber log", transform=ax6.transAxes, ha="center")

    plt.tight_layout()
    save_fig(fig, FIGURES / "final_validation_dashboard.png")
    print("Validation dashboard saved.")


if __name__ == "__main__":
    main()
