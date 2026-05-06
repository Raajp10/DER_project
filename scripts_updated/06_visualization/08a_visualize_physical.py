"""
Generate physical layer visualizations.
Writes to: figures/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import (
    CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, SCENARIO_MANIFEST_CSV,
    FIGURES,
)
from plot_utils import (
    apply_style, save_fig, downsample, multi_subplot_time,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TOTAL_S = 604_800
DS = 300  # downsample factor: 300-second averages for full 7-day plots


def plot_clean_vs_attacked():
    if not CLEAN_PHYSICAL_CSV.exists() or not ATTACKED_PHYSICAL_CSV.exists():
        print("  SKIP: Physical CSVs not found.")
        return

    print("  Plotting clean vs attacked overview...")
    cols = ["pv_actual_p_kw", "bess_actual_p_kw", "bess_soc_percent",
            "pcc_v_a_pu", "pcc_p_kw"]
    labels = ["PV Output (kW)", "BESS Power (kW)", "BESS SOC (%)",
              "PCC Voltage Phase A (pu)", "PCC Net Power (kW)"]

    clean = pd.read_csv(CLEAN_PHYSICAL_CSV, usecols=["time_s"] + cols)
    attacked = pd.read_csv(ATTACKED_PHYSICAL_CSV, usecols=["time_s"] + cols)

    apply_style()
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 3.5 * len(cols)), sharex=True)
    x_h = np.linspace(0, TOTAL_S / 3600, len(downsample(clean[cols[0]].values, DS)))
    palette = plt.cm.tab10.colors

    for i, (col, lbl) in enumerate(zip(cols, labels)):
        c_ds = downsample(clean[col].values, DS)
        a_ds = downsample(attacked[col].values, DS)
        axes[i].fill_between(x_h, c_ds, alpha=0.35, color=palette[0], label="Clean")
        axes[i].plot(x_h, c_ds, color=palette[0], linewidth=0.8, alpha=0.7)
        axes[i].plot(x_h, a_ds, color=palette[1], linewidth=0.8, alpha=0.85, label="Attacked")
        axes[i].set_ylabel(lbl, fontsize=9)
        axes[i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (hours from 2026-01-01 00:00 UTC)")
    fig.suptitle("IEEE 123-Bus DER: Clean vs Attacked Physical Timeseries (7 Days, 5-min averages)",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    save_fig(fig, FIGURES / "physical_clean_vs_attacked_overview.png")


def plot_scenario_examples():
    if not ATTACKED_PHYSICAL_CSV.exists() or not SCENARIO_MANIFEST_CSV.exists():
        print("  SKIP: Required files not found.")
        return
    if not CLEAN_PHYSICAL_CSV.exists():
        return

    print("  Plotting scenario physical effect examples...")
    scenarios = pd.read_csv(SCENARIO_MANIFEST_CSV)
    attacked = pd.read_csv(ATTACKED_PHYSICAL_CSV,
                           usecols=["time_s", "pv_actual_p_kw", "bess_actual_p_kw",
                                    "bess_soc_percent", "pcc_voltage_mean_pu",
                                    "pcc_p_kw", "irradiance_pu",
                                    "physical_effect_active_flag", "physical_effect_type"])
    clean = pd.read_csv(CLEAN_PHYSICAL_CSV,
                        usecols=["time_s", "pv_actual_p_kw", "bess_actual_p_kw",
                                 "bess_soc_percent", "pcc_voltage_mean_pu", "pcc_p_kw"])

    target_scenarios = [
        "physical_irradiance_drop", "physical_load_step", "delayed_pv_limit",
        "wrong_pv_setpoint", "bess_wrong_direction", "replay_command",
        "high_rate_command_burst", "soc_constraint_violation", "voltage_sag",
    ]

    apply_style()
    n_plots = len(target_scenarios)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]
    palette = plt.cm.tab10.colors

    for i, sname in enumerate(target_scenarios):
        sc_rows = scenarios[scenarios["scenario_name"] == sname]
        ax = axes[i]
        if len(sc_rows) == 0:
            ax.text(0.5, 0.5, f"{sname}: no scenarios in manifest",
                    transform=ax.transAxes, ha="center")
            ax.set_title(sname)
            continue

        sc = sc_rows.iloc[0]
        s_s = max(0, int(sc["start_time_s"]) - 120)
        e_s = min(604_799, int(sc["end_time_s"]) + 120)

        w_att = attacked[(attacked["time_s"] >= s_s) & (attacked["time_s"] <= e_s)]
        w_cln = clean[(clean["time_s"] >= s_s) & (clean["time_s"] <= e_s)]

        if len(w_att) == 0:
            ax.text(0.5, 0.5, f"{sname}: no data in window", transform=ax.transAxes, ha="center")
            ax.set_title(sname)
            continue

        x = (w_att["time_s"].values - s_s) / 60  # minutes from window start

        # Pick signal based on scenario type
        if "bess" in sname or "soc" in sname:
            col = "bess_soc_percent"
            ylabel = "BESS SOC (%)"
        elif "voltage" in sname:
            col = "pcc_voltage_mean_pu"
            ylabel = "PCC Voltage (pu)"
        else:
            col = "pv_actual_p_kw"
            ylabel = "PV Output (kW)"

        ax.plot(x, w_cln[col].values[:len(x)] if col in w_cln.columns else np.zeros(len(x)),
                color=palette[0], linewidth=1.2, label="Clean", alpha=0.7)
        ax.plot(x, w_att[col].values, color=palette[1], linewidth=1.2, label="Attacked")

        # Shade attack window
        att_start_min = (int(sc["start_time_s"]) - s_s) / 60
        att_end_min = (int(sc["end_time_s"]) - s_s) / 60
        ax.axvspan(att_start_min, att_end_min, alpha=0.15, color="red", label="Attack window")

        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f"{sname}", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("Minutes from window start")

    fig.suptitle("Scenario Physical Effect Examples — IEEE 123-Bus DER Dataset", fontsize=11, y=1.01)
    plt.tight_layout()
    save_fig(fig, FIGURES / "scenario_physical_effect_examples.png")


def plot_generation_method_summary():
    if not ATTACKED_PHYSICAL_CSV.exists():
        print("  SKIP: Attacked physical CSV not found.")
        return

    print("  Plotting generation method summary...")
    attacked = pd.read_csv(ATTACKED_PHYSICAL_CSV, usecols=["generation_method"])
    counts = attacked["generation_method"].value_counts()

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=plt.cm.Set2.colors[:len(counts)],
                  alpha=0.85, edgecolor="white")
    ax.set_ylabel("Row Count")
    ax.set_title("Physical Data Generation Method Distribution (7-Day Attacked Dataset)")
    ax.tick_params(axis="x", rotation=15)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000, f"{v:,}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig(fig, FIGURES / "generation_method_summary.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Generating physical visualizations...")
    plot_clean_vs_attacked()
    plot_scenario_examples()
    plot_generation_method_summary()
    print("Physical visualizations done.")


if __name__ == "__main__":
    main()
