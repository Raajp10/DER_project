"""Generate context mapping visualizations."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import LIFECYCLE_MAP_CSV, CONTEXT_WINDOWS_CSV, FIGURES
from plot_utils import apply_style, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_ts(ts_str) -> float:
    """Convert timestamp string to hours since 2026-01-01."""
    try:
        if not ts_str or str(ts_str).strip() in ("", "nan", "NaT"):
            return np.nan
        t = pd.Timestamp(str(ts_str))
        origin = pd.Timestamp("2026-01-01T00:00:00Z")
        return (t - origin).total_seconds() / 3600.0
    except Exception:
        return np.nan


def plot_command_to_response_alignment():
    if not LIFECYCLE_MAP_CSV.exists():
        print("  SKIP: Lifecycle map not found.")
        return

    print("  Plotting command-to-physical response alignment...")
    lc = pd.read_csv(LIFECYCLE_MAP_CSV)

    apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # For cyber-physical scenarios, show command_sent vs physical_effect_start
    cp = lc[lc["scenario_class"] == "cyber_physical"].copy()
    if len(cp) == 0:
        ax.text(0.5, 0.5, "No cyber-physical scenarios", transform=ax.transAxes, ha="center")
    else:
        sent_h = cp["command_sent_time_utc"].apply(parse_ts).values
        phys_h = cp["physical_effect_start_time_utc"].apply(parse_ts).values
        apply_h = cp["command_apply_time_utc"].apply(parse_ts).values

        valid = ~(np.isnan(sent_h) | np.isnan(phys_h))
        sc_names = cp["scenario_name"].values

        ax.scatter(sent_h[valid], phys_h[valid], c="steelblue", s=60, alpha=0.7,
                   label="Command Sent → Physical Effect Start", zorder=3)
        # Diagonal reference line
        mi = min(np.nanmin(sent_h), np.nanmin(phys_h))
        ma = max(np.nanmax(sent_h), np.nanmax(phys_h))
        ax.plot([mi, ma], [mi, ma], "k--", linewidth=0.8, alpha=0.5, label="t_sent = t_physical")
        ax.set_xlabel("Command Sent Time (hours from 2026-01-01)")
        ax.set_ylabel("Physical Effect Start Time (hours from 2026-01-01)")
        ax.set_title("Cyber Command to Physical Response Alignment — Cyber-Physical Scenarios")
        ax.legend()

    plt.tight_layout()
    save_fig(fig, FIGURES / "context_command_to_physical_response_alignment.png")


def plot_lifecycle_timing_bars():
    if not LIFECYCLE_MAP_CSV.exists():
        return
    print("  Plotting lifecycle timing bars...")
    lc = pd.read_csv(LIFECYCLE_MAP_CSV)

    time_cols = [
        ("cyber_onset_time_utc", "Cyber Onset"),
        ("command_sent_time_utc", "Cmd Sent"),
        ("command_recv_time_utc", "Cmd Recv"),
        ("command_accept_time_utc", "Cmd Accept"),
        ("command_apply_time_utc", "Cmd Apply"),
        ("command_response_time_utc", "Cmd Response"),
        ("physical_effect_start_time_utc", "Phys Start"),
        ("physical_effect_peak_time_utc", "Phys Peak"),
        ("physical_effect_end_time_utc", "Phys End"),
    ]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of cyber onset times
    onset_h = lc["cyber_onset_time_utc"].apply(parse_ts).dropna()
    phys_h = lc["physical_effect_start_time_utc"].apply(parse_ts).dropna()

    if len(onset_h) > 0:
        axes[0].hist(onset_h, bins=30, color="steelblue", alpha=0.75, edgecolor="white")
        axes[0].set_xlabel("Hour of Week")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Cyber Onset Times")

    if len(phys_h) > 0:
        axes[1].hist(phys_h, bins=30, color="darkorange", alpha=0.75, edgecolor="white")
        axes[1].set_xlabel("Hour of Week")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Physical Effect Start Times")

    fig.suptitle("Cyber and Physical Event Timing Distributions (7 Days)", fontsize=11)
    plt.tight_layout()
    save_fig(fig, FIGURES / "cyber_lifecycle_timing_distributions.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating context visualizations...")
    plot_command_to_response_alignment()
    plot_lifecycle_timing_bars()
    print("Context visualizations done.")


if __name__ == "__main__":
    main()
