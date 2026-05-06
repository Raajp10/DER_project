"""
DER component visualization and OpenDSS event-window summary figures.
Generates:
  - der_pv_output_profile.png
  - der_bess_soc_profile.png
  - der_bess_power_profile.png
  - der_pcc_voltage_profile.png
  - der_component_overview.png
  - opendss_event_window_resolution_summary.png
  - generation_method_summary.png  (updated with new counts)
  - scenario_resolution_method_by_type.png
"""
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

from paths import CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, FIGURES, METADATA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

TOTAL_S = 604_800
DS = 300  # 5-minute averages for 7-day plots
DAYS = np.linspace(0, 7, TOTAL_S // DS)


def downsample(arr, factor):
    n = (len(arr) // factor) * factor
    return arr[:n].reshape(-1, factor).mean(axis=1)


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def save_fig(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"    Saved: {name}")


def load_dfs():
    if not CLEAN_PHYSICAL_CSV.exists() or not ATTACKED_PHYSICAL_CSV.exists():
        return None, None
    cols = [
        "time_s", "pv_actual_p_kw", "pv_available_kw", "pv_s_rated_kva",
        "bess_actual_p_kw", "bess_soc_percent", "bess_capacity_kwh",
        "pcc_v_a_pu", "pcc_v_b_pu", "pcc_v_c_pu",
        "pcc_voltage_mean_pu", "pcc_p_kw", "pcc_q_kvar",
        "irradiance_pu", "temperature_c",
        "physical_effect_active_flag", "physical_effect_type", "generation_method",
    ]
    avail_cols_c = [c for c in cols if c in pd.read_csv(CLEAN_PHYSICAL_CSV, nrows=0).columns]
    avail_cols_a = [c for c in cols if c in pd.read_csv(ATTACKED_PHYSICAL_CSV, nrows=0).columns]
    clean = pd.read_csv(CLEAN_PHYSICAL_CSV, usecols=avail_cols_c)
    attacked = pd.read_csv(ATTACKED_PHYSICAL_CSV, usecols=avail_cols_a)
    return clean, attacked


def plot_pv_output(clean, attacked):
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("PV System Output Profile — 7-Day Clean vs Attacked", fontsize=13, fontweight="bold")

    x = DAYS[:len(downsample(clean["pv_actual_p_kw"].values, DS))]

    # Panel 1: PV power output
    c_pv = downsample(clean["pv_actual_p_kw"].values, DS)
    a_pv = downsample(attacked["pv_actual_p_kw"].values, DS)
    axes[0].fill_between(x, c_pv, alpha=0.30, color="#2196F3", label="Clean")
    axes[0].plot(x, c_pv, color="#2196F3", linewidth=0.8)
    axes[0].plot(x, a_pv, color="#F44336", linewidth=0.8, alpha=0.85, label="Attacked")
    axes[0].set_ylabel("PV Output (kW)")
    axes[0].axhline(y=100, color="#FF9800", linestyle="--", linewidth=1.2, label="Rated 100 kW")
    axes[0].legend(loc="upper right")
    axes[0].set_title("PV Active Power Output")

    # Panel 2: PV available vs actual (curtailment)
    if "pv_available_kw" in clean.columns:
        c_avail = downsample(clean["pv_available_kw"].values, DS)
        axes[1].fill_between(x, c_avail, c_pv, alpha=0.4, color="#FF9800", label="Curtailment")
        axes[1].fill_between(x, c_pv, alpha=0.3, color="#2196F3", label="Actual Output")
        axes[1].set_ylabel("PV Power (kW)")
        axes[1].legend(loc="upper right")
    else:
        axes[1].plot(x, c_pv, color="#2196F3", linewidth=0.8)
        axes[1].set_ylabel("PV Power (kW)")
    axes[1].set_title("PV Available vs Actual (Clean)")

    # Panel 3: Irradiance
    if "irradiance_pu" in clean.columns:
        c_irr = downsample(clean["irradiance_pu"].values, DS)
        a_irr = downsample(attacked["irradiance_pu"].values, DS)
        axes[2].plot(x, c_irr, color="#2196F3", linewidth=0.8, label="Clean irradiance")
        axes[2].plot(x, a_irr, color="#F44336", linewidth=0.8, alpha=0.85, label="Attacked irradiance")
        axes[2].set_ylabel("Irradiance (pu)")
        axes[2].legend(loc="upper right")
    else:
        axes[2].plot(x, c_pv / 100.0, color="#2196F3", linewidth=0.8)
        axes[2].set_ylabel("PV Output (normalized)")
    axes[2].set_title("Solar Irradiance Profile")
    axes[2].set_xlabel("Time (days)")

    plt.tight_layout()
    save_fig(fig, "der_pv_output_profile.png")


def plot_bess_soc(clean, attacked):
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("BESS State of Charge Profile — 7-Day", fontsize=13, fontweight="bold")

    x = DAYS[:len(downsample(clean["bess_soc_percent"].values, DS))]
    c_soc = downsample(clean["bess_soc_percent"].values, DS)
    a_soc = downsample(attacked["bess_soc_percent"].values, DS)

    # SOC plot
    axes[0].fill_between(x, c_soc, alpha=0.25, color="#4CAF50")
    axes[0].plot(x, c_soc, color="#4CAF50", linewidth=1.2, label="Clean SOC")
    axes[0].plot(x, a_soc, color="#F44336", linewidth=0.9, alpha=0.85, label="Attacked SOC")
    axes[0].axhline(y=90, color="#FF9800", linestyle="--", linewidth=1.2, label="SOC max 90%")
    axes[0].axhline(y=10, color="#9C27B0", linestyle="--", linewidth=1.2, label="SOC min 10%")
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Battery State of Charge")
    # Shade violation zones
    axes[0].fill_between(x, 0, 10, alpha=0.08, color="#9C27B0", label="_nolegend_")
    axes[0].fill_between(x, 90, 105, alpha=0.08, color="#FF9800", label="_nolegend_")

    # SOC delta
    soc_delta = a_soc - c_soc
    axes[1].fill_between(x, soc_delta, 0,
                         where=(soc_delta > 0), alpha=0.4, color="#4CAF50", label="SOC higher than clean")
    axes[1].fill_between(x, soc_delta, 0,
                         where=(soc_delta < 0), alpha=0.4, color="#F44336", label="SOC lower than clean")
    axes[1].plot(x, soc_delta, color="#333333", linewidth=0.6)
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_ylabel("SOC Delta (pp)")
    axes[1].set_xlabel("Time (days)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("SOC Residual (Attacked - Clean)")

    plt.tight_layout()
    save_fig(fig, "der_bess_soc_profile.png")


def plot_bess_power(clean, attacked):
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("BESS Active Power Profile — 7-Day", fontsize=13, fontweight="bold")

    x = DAYS[:len(downsample(clean["bess_actual_p_kw"].values, DS))]
    c_bess = downsample(clean["bess_actual_p_kw"].values, DS)
    a_bess = downsample(attacked["bess_actual_p_kw"].values, DS)

    axes[0].fill_between(x, c_bess, 0,
                         where=(c_bess > 0), alpha=0.3, color="#F44336", label="Discharging (clean)")
    axes[0].fill_between(x, c_bess, 0,
                         where=(c_bess < 0), alpha=0.3, color="#2196F3", label="Charging (clean)")
    axes[0].plot(x, c_bess, color="#333333", linewidth=0.7)
    axes[0].plot(x, a_bess, color="#FF9800", linewidth=0.8, alpha=0.85, label="Attacked")
    axes[0].axhline(y=50, color="#F44336", linestyle="--", linewidth=1.0, label="Max discharge 50 kW")
    axes[0].axhline(y=-50, color="#2196F3", linestyle="--", linewidth=1.0, label="Max charge -50 kW")
    axes[0].set_ylabel("BESS Power (kW)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=7)
    axes[0].set_title("BESS Active Power (+ = Discharge, - = Charge)")

    # Power delta
    bess_delta = a_bess - c_bess
    axes[1].fill_between(x, bess_delta, 0, alpha=0.4, color="#FF9800")
    axes[1].plot(x, bess_delta, color="#333333", linewidth=0.6)
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Power Delta (kW)")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_title("BESS Power Residual (Attacked - Clean)")

    plt.tight_layout()
    save_fig(fig, "der_bess_power_profile.png")


def plot_pcc_voltage(clean, attacked):
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("PCC Voltage Profile — 7-Day (All Three Phases)", fontsize=13, fontweight="bold")

    x = DAYS[:len(downsample(clean["pcc_v_a_pu"].values, DS))]

    # Phase voltages
    for col, color, label in [
        ("pcc_v_a_pu", "#F44336", "Phase A"),
        ("pcc_v_b_pu", "#4CAF50", "Phase B"),
        ("pcc_v_c_pu", "#2196F3", "Phase C"),
    ]:
        if col in clean.columns:
            c_v = downsample(clean[col].values, DS)
            axes[0].plot(x, c_v, color=color, linewidth=0.8, label=f"{label} (clean)")

    if "pcc_voltage_mean_pu" in attacked.columns:
        a_vmean = downsample(attacked["pcc_voltage_mean_pu"].values, DS)
        axes[0].plot(x, a_vmean, color="#FF9800", linewidth=1.0, linestyle="--",
                     label="Mean (attacked)", alpha=0.85)

    axes[0].axhline(y=1.05, color="red", linestyle=":", linewidth=0.8, label="ANSI C84.1 upper")
    axes[0].axhline(y=0.95, color="red", linestyle=":", linewidth=0.8, label="ANSI C84.1 lower")
    axes[0].set_ylabel("Voltage (pu)")
    axes[0].set_ylim(0.85, 1.10)
    axes[0].legend(loc="upper right", ncol=2, fontsize=7)
    axes[0].set_title("Three-Phase PCC Voltage (Clean Baseline)")

    # Voltage delta
    if "pcc_v_a_pu" in attacked.columns:
        c_va = downsample(clean["pcc_v_a_pu"].values, DS)
        a_va = downsample(attacked["pcc_v_a_pu"].values, DS)
        vdelta = a_va - c_va
        axes[1].fill_between(x, vdelta, 0,
                             where=(vdelta < 0), alpha=0.4, color="#F44336", label="Voltage sag")
        axes[1].fill_between(x, vdelta, 0,
                             where=(vdelta > 0), alpha=0.3, color="#4CAF50", label="Voltage rise")
        axes[1].plot(x, vdelta, color="#333333", linewidth=0.5)
        axes[1].axhline(y=0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Voltage Delta (pu)")
        axes[1].legend(loc="upper right")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_title("Phase A Voltage Residual (Attacked - Clean)")

    plt.tight_layout()
    save_fig(fig, "der_pcc_voltage_profile.png")


def plot_der_component_overview(clean, attacked):
    """4-panel DER component overview for quick visual inspection."""
    apply_style()
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("DER Component Overview — IEEE 123-Bus Site (Bus 65)\n"
                 "7-Day, 1-Second Resolution | Physics-Constrained Surrogate",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    x = DAYS[:len(downsample(clean["pv_actual_p_kw"].values, DS))]
    palette = plt.cm.tab10.colors

    # PV output
    ax1 = fig.add_subplot(gs[0, 0])
    c_pv = downsample(clean["pv_actual_p_kw"].values, DS)
    a_pv = downsample(attacked["pv_actual_p_kw"].values, DS)
    ax1.fill_between(x, c_pv, alpha=0.3, color=palette[0], label="Clean")
    ax1.plot(x, c_pv, color=palette[0], linewidth=0.7)
    ax1.plot(x, a_pv, color=palette[1], linewidth=0.7, label="Attacked")
    ax1.axhline(y=100, color=palette[2], linestyle="--", linewidth=1.0, label="Rated 100 kW")
    ax1.set_title("PV Output (kW)")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("kW")
    ax1.legend(loc="upper right", fontsize=7)

    # BESS SOC
    ax2 = fig.add_subplot(gs[0, 1])
    c_soc = downsample(clean["bess_soc_percent"].values, DS)
    a_soc = downsample(attacked["bess_soc_percent"].values, DS)
    ax2.fill_between(x, c_soc, alpha=0.3, color=palette[2], label="Clean")
    ax2.plot(x, c_soc, color=palette[2], linewidth=0.7)
    ax2.plot(x, a_soc, color=palette[1], linewidth=0.7, label="Attacked")
    ax2.axhline(y=90, color=palette[4], linestyle="--", linewidth=1.0, label="SOC max 90%")
    ax2.axhline(y=10, color=palette[6], linestyle="--", linewidth=1.0, label="SOC min 10%")
    ax2.fill_between(x, 0, 10, alpha=0.07, color=palette[6])
    ax2.fill_between(x, 90, 105, alpha=0.07, color=palette[4])
    ax2.set_title("BESS State of Charge (%)")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("%")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=7)

    # BESS power
    ax3 = fig.add_subplot(gs[1, 0])
    c_bess = downsample(clean["bess_actual_p_kw"].values, DS)
    a_bess = downsample(attacked["bess_actual_p_kw"].values, DS)
    ax3.fill_between(x, c_bess, 0,
                     where=(c_bess > 0), alpha=0.3, color=palette[1], label="Discharge (clean)")
    ax3.fill_between(x, c_bess, 0,
                     where=(c_bess < 0), alpha=0.3, color=palette[0], label="Charge (clean)")
    ax3.plot(x, c_bess, color=palette[9], linewidth=0.7)
    ax3.plot(x, a_bess, color=palette[1], linewidth=0.7, alpha=0.85, label="Attacked")
    ax3.axhline(y=50, color=palette[3], linestyle="--", linewidth=0.8)
    ax3.axhline(y=-50, color=palette[3], linestyle="--", linewidth=0.8, label="±50 kW limit")
    ax3.set_title("BESS Active Power (+ Discharge, - Charge)")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("kW")
    ax3.legend(loc="upper right", fontsize=7)

    # PCC voltage
    ax4 = fig.add_subplot(gs[1, 1])
    for col, color, lbl in [("pcc_v_a_pu", palette[1], "Phase A"),
                             ("pcc_v_b_pu", palette[2], "Phase B"),
                             ("pcc_v_c_pu", palette[0], "Phase C")]:
        if col in clean.columns:
            ax4.plot(x, downsample(clean[col].values, DS), color=color,
                     linewidth=0.6, label=lbl, alpha=0.8)
    ax4.axhline(y=1.05, color="red", linestyle=":", linewidth=0.8)
    ax4.axhline(y=0.95, color="red", linestyle=":", linewidth=0.8, label="ANSI limits")
    ax4.set_title("PCC Three-Phase Voltage (Clean, pu)")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("pu")
    ax4.set_ylim(0.88, 1.08)
    ax4.legend(loc="upper right", fontsize=7)

    save_fig(fig, "der_component_overview.png")


def plot_opendss_event_window_summary():
    """Bar chart of OpenDSS event-window resolution results by scenario."""
    ods_path = METADATA / "opendss_event_window_results.json"
    if not ods_path.exists():
        apply_style()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "OpenDSS event-window stage not yet run.\n"
                "Run pipeline with 04e enabled.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.8))
        ax.set_title("OpenDSS Event-Window Resolution Summary (Stage Not Run)")
        ax.axis("off")
        save_fig(fig, "opendss_event_window_resolution_summary.png")
        return

    with open(ods_path, encoding="utf-8") as f:
        ods = json.load(f)

    scenarios = ods.get("scenarios", [])
    stage = ods.get("stage", 1)
    successes = ods.get("successes", 0)
    failures = ods.get("failures", 0)

    if not scenarios:
        apply_style()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Stage {stage} — No scenarios run yet.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("OpenDSS Event-Window Resolution Summary")
        ax.axis("off")
        save_fig(fig, "opendss_event_window_resolution_summary.png")
        return

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"OpenDSS Event-Window Resolution — Stage {stage}\n"
                 f"{successes} succeeded, {failures} failed",
                 fontsize=12, fontweight="bold")

    # Left: per-scenario bar chart
    names = [s["effect_type"].replace("_", "\n") for s in scenarios]
    colors = ["#4CAF50" if s["status"] == "PASS" else "#F44336" for s in scenarios]
    rows_resolved = [s.get("rows_resolved", 0) for s in scenarios]
    y_pos = range(len(scenarios))

    axes[0].barh(y_pos, rows_resolved, color=colors)
    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("Rows Resolved")
    axes[0].set_title("Rows Resolved per Scenario")
    pass_patch = mpatches.Patch(color="#4CAF50", label="PASS")
    fail_patch = mpatches.Patch(color="#F44336", label="FAIL")
    axes[0].legend(handles=[pass_patch, fail_patch], loc="lower right")

    # Right: pie chart of generation_method counts
    gm_counts = ods.get("generation_method_counts", {})
    if gm_counts:
        labels = list(gm_counts.keys())
        sizes = [int(v) for v in gm_counts.values()]
        colors_pie = ["#2196F3", "#4CAF50", "#FF9800", "#9E9E9E"][:len(labels)]
        wedges, texts, autotexts = axes[1].pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors_pie, startangle=90,
            textprops={"fontsize": 8})
        axes[1].set_title("Generation Method Distribution")
    else:
        axes[1].text(0.5, 0.5, "No generation method data",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")

    plt.tight_layout()
    save_fig(fig, "opendss_event_window_resolution_summary.png")


def plot_generation_method_summary(attacked):
    """Updated generation_method_summary.png with current counts."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Generation Method Summary — Attacked Physical Dataset",
                 fontsize=12, fontweight="bold")

    gm_counts = attacked["generation_method"].value_counts()

    colors_map = {
        "physics_constrained_surrogate": "#FF9800",
        "opendss_event_window_resolved": "#4CAF50",
        "opendss_clean_baseline": "#2196F3",
        "csv_rule_legacy": "#9E9E9E",
    }
    colors = [colors_map.get(k, "#BDBDBD") for k in gm_counts.index]

    # Bar chart
    axes[0].bar(range(len(gm_counts)), gm_counts.values, color=colors)
    axes[0].set_xticks(range(len(gm_counts)))
    axes[0].set_xticklabels([k.replace("_", "\n") for k in gm_counts.index], fontsize=8)
    axes[0].set_ylabel("Row Count")
    axes[0].set_title("Row Count by Generation Method")
    for i, v in enumerate(gm_counts.values):
        axes[0].text(i, v + max(gm_counts.values) * 0.01, f"{v:,}", ha="center", fontsize=8)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        gm_counts.values, labels=[k.replace("_", "\n") for k in gm_counts.index],
        autopct="%1.1f%%", colors=colors, startangle=90,
        textprops={"fontsize": 8})
    axes[1].set_title("Generation Method Distribution")

    plt.tight_layout()
    save_fig(fig, "generation_method_summary.png")


def plot_scenario_resolution_by_type(attacked):
    """Show generation_method breakdown per physical_effect_type."""
    if "physical_effect_type" not in attacked.columns:
        return

    apply_style()
    pivot = (attacked[attacked["physical_effect_active_flag"] == 1]
             .groupby(["physical_effect_type", "generation_method"])
             .size()
             .unstack(fill_value=0))

    colors_map = {
        "physics_constrained_surrogate": "#FF9800",
        "opendss_event_window_resolved": "#4CAF50",
        "opendss_clean_baseline": "#2196F3",
        "csv_rule_legacy": "#9E9E9E",
    }
    bar_colors = [colors_map.get(c, "#BDBDBD") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Scenario Resolution Method by Anomaly Type\n"
                 "(Attack-Active Rows Only)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Physical Effect Type")
    ax.set_ylabel("Row Count")
    ax.set_xticklabels([x.replace("_", "\n") for x in pivot.index], rotation=0, fontsize=8)
    ax.legend(title="generation_method", fontsize=8, title_fontsize=8)
    plt.tight_layout()
    save_fig(fig, "scenario_resolution_method_by_type.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("  Loading physical CSVs...")
    clean, attacked = load_dfs()

    if clean is None or attacked is None:
        print("  SKIP: Physical CSVs not found.")
        # Still generate OpenDSS summary (may show 'not yet run')
        plot_opendss_event_window_summary()
        return

    print("  Plotting DER component overview...")
    plot_der_component_overview(clean, attacked)

    print("  Plotting PV output profile...")
    plot_pv_output(clean, attacked)

    print("  Plotting BESS SOC profile...")
    plot_bess_soc(clean, attacked)

    print("  Plotting BESS power profile...")
    plot_bess_power(clean, attacked)

    print("  Plotting PCC voltage profile...")
    plot_pcc_voltage(clean, attacked)

    print("  Plotting OpenDSS event-window summary...")
    plot_opendss_event_window_summary()

    print("  Plotting updated generation_method summary...")
    plot_generation_method_summary(attacked)

    print("  Plotting scenario resolution by type...")
    plot_scenario_resolution_by_type(attacked)

    print("  DER component visualizations complete.")


if __name__ == "__main__":
    main()
