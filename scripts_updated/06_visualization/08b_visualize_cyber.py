"""Generate cyber layer visualizations."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import CYBER_ANOMALOUS_CSV, CYBER_NORMAL_CSV, FIGURES
from plot_utils import apply_style, save_fig, bar_chart

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_lifecycle_stage_counts(df: pd.DataFrame):
    if "lifecycle_stage" not in df.columns:
        return
    counts = df["lifecycle_stage"].value_counts()
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index, counts.values, color=plt.cm.tab20.colors[:len(counts)], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Event Count")
    ax.set_title("IEEE 2030.5-Style Lifecycle Stage Counts — Anomalous Cyber Log")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    save_fig(fig, FIGURES / "cyber_lifecycle_timeline_examples.png")


def plot_scenario_name_counts(df: pd.DataFrame):
    if "scenario_id" not in df.columns:
        return
    # Parse scenario name from scenario_id
    df = df.copy()
    df["sname"] = df["scenario_id"].str.split("_").str[1:].apply(lambda x: "_".join(x) if isinstance(x, list) else "")
    counts = df["sname"].value_counts().head(20)
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index, counts.values, color="#42A5F5", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Event Count")
    ax.set_title("Cyber Events by Scenario Name")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    save_fig(fig, FIGURES / "cyber_anomaly_distribution.png")


def plot_cia_dimension_counts(df: pd.DataFrame):
    if "cia_dimension" not in df.columns:
        return
    counts = df["cia_dimension"].value_counts()
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726"]
    ax.bar(counts.index, counts.values, color=colors[:len(counts)], alpha=0.85, edgecolor="white")
    ax.set_ylabel("Event Count")
    ax.set_title("CIA Dimension Distribution — Anomalous Cyber Log")
    plt.tight_layout()
    save_fig(fig, FIGURES / "protocol_claim_level_summary.png")


def plot_delivery_status(df: pd.DataFrame):
    if "delivery_status" not in df.columns:
        return
    counts = df["delivery_status"].value_counts()
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=plt.cm.Pastel1.colors[:len(counts)])
    ax.set_title("Delivery Status Distribution")
    plt.tight_layout()
    save_fig(fig, FIGURES / "cyber_delivery_status.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating cyber visualizations...")

    if not CYBER_ANOMALOUS_CSV.exists():
        print("  SKIP: Anomalous cyber log not found.")
        return

    df = pd.read_csv(CYBER_ANOMALOUS_CSV)
    plot_lifecycle_stage_counts(df)
    plot_scenario_name_counts(df)
    plot_cia_dimension_counts(df)
    plot_delivery_status(df)
    print("Cyber visualizations done.")


if __name__ == "__main__":
    main()
