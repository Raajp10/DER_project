"""
Phase 2 Zero-Day Comparison Plots.

Generates paper-style normal-vs-attack comparison figures for all accepted
bundles and scenario families.

Inputs:
  data_updated/raw/physical_timeseries_clean_improved_7d.csv
  outputs/zero_day_physical_attacked.csv
  outputs/zero_day_scenario_manifest.csv

Outputs (all in figures/comparison_plots/):
  chatgpt_normal_vs_attack.png
  claude_normal_vs_attack.png
  gemini_normal_vs_attack.png
  grok_normal_vs_attack.png
  chatgpt_pv_p_kw_comparison.png
  claude_bess_soc_percent_comparison.png
  gemini_pv_p_kw_comparison.png
  gemini_bess_p_kw_comparison.png
  grok_pcc_v_a_pu_comparison.png
  cross_bundle_comparison_panel.png
  scenario_family_comparison.png

Also writes:
  reports/ZERO_DAY_FIGURE_INDEX.md
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(r"D:\updated_dataset")
PHASE2_ROOT  = Path(r"D:\updated_dataset\phase2_zero_day_eval")
OUTPUTS_DIR  = PHASE2_ROOT / "outputs"
REPORTS_DIR  = PHASE2_ROOT / "reports"
FIGURES_DIR  = PHASE2_ROOT / "figures" / "comparison_plots"

CLEAN_CSV    = PROJECT_ROOT / "data_updated" / "raw" / "physical_timeseries_clean_improved_7d.csv"
ATTACKED_CSV = OUTPUTS_DIR / "zero_day_physical_attacked.csv"
MANIFEST_CSV = OUTPUTS_DIR / "zero_day_scenario_manifest.csv"

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ── Plot styling ─────────────────────────────────────────────────────────────
STYLE = {
    "figure.dpi": 150,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
}
CLEAN_COLOR   = "#1f77b4"   # muted blue
ATTACK_COLOR  = "#d62728"   # brick red
SHADE_COLOR   = "#ff9896"   # light red
SHADE_ALPHA   = 0.25

SIGNAL_LABELS = {
    "pv_p_kw":             "PV Active Power (kW)",
    "pv_q_kvar":           "PV Reactive Power (kvar)",
    "bess_p_kw":           "BESS Active Power (kW)",
    "bess_q_kvar":         "BESS Reactive Power (kvar)",
    "bess_soc_percent":    "BESS State of Charge (%)",
    "pcc_v_a_pu":          "PCC Voltage Phase A (pu)",
    "pcc_v_b_pu":          "PCC Voltage Phase B (pu)",
    "pcc_v_c_pu":          "PCC Voltage Phase C (pu)",
    "pcc_i_a_amp":         "PCC Current Phase A (A)",
    "pcc_p_kw":            "PCC Active Power (kW)",
    "pcc_q_kvar":          "PCC Reactive Power (kvar)",
    "irradiance_pu":       "Irradiance (pu)",
    "pv_actual_p_kw":      "PV Active Power (kW)",
    "pv_actual_q_kvar":    "PV Reactive Power (kvar)",
    "bess_actual_p_kw":    "BESS Active Power (kW)",
    "bess_actual_q_kvar":  "BESS Reactive Power (kvar)",
}

# ── Representative scenario config ──────────────────────────────────────────
# Chosen for clear physical visibility; one per bundle + one per family

BUNDLE_REPS = {
    "chatgpt": {
        "scenario_id": "zdl_chatgpt_pv_curtail_004",
        "signals": ["pv_p_kw", "pcc_p_kw"],
        "title": "ChatGPT — PV Curtailment Mismatch",
        "reason": "Midday PV curtailment is visually clear; two signals show correlated response",
    },
    "claude": {
        "scenario_id": "zdl_claude_stale_meas_002",
        "signals": ["bess_soc_percent"],
        "title": "Claude — Stale BESS SOC Measurement",
        "reason": "hold_stale effect produces a flat line that is immediately distinguishable from the clean drifting SOC baseline",
    },
    "gemini": {
        "scenario_id": "zdl_gemini_cp_002",
        "signals": ["pv_p_kw", "bess_p_kw"],
        "title": "Gemini — Coordinated PV-BESS Response",
        "reason": "Two correlated signals move in opposite directions; good cross-asset illustration",
    },
    "grok": {
        "scenario_id": "zdl_grok_pcc_voltage_004",
        "signals": ["pcc_v_a_pu"],
        "title": "Grok — PCC Voltage Sag",
        "reason": "Voltage sag is a classic and clearly visible physical anomaly",
    },
}

FAMILY_REPS = {
    "pv_curtailment_mismatch": {
        "scenario_id": "zdl_chatgpt_pv_curtail_004",
        "signal": "pv_p_kw",
        "bundle": "chatgpt",
        "title": "PV Curtailment Mismatch",
    },
    "soc_aware_bess_dispatch_anomaly": {
        "scenario_id": "zdl_chatgpt_soc_dispatch_002",
        "signal": "bess_p_kw",
        "bundle": "chatgpt",
        "title": "SOC-Aware BESS Dispatch Anomaly",
    },
    "pcc_voltage_deviation": {
        "scenario_id": "zdl_grok_pcc_voltage_004",
        "signal": "pcc_v_a_pu",
        "bundle": "grok",
        "title": "PCC Voltage Deviation",
    },
    "oscillatory_bess_control": {
        "scenario_id": "zdl_grok_oscillatory_bess_002",
        "signal": "bess_p_kw",
        "bundle": "grok",
        "title": "Oscillatory BESS Control",
    },
}

# Extra per-variable plots
EXTRA_PLOTS = [
    {"bundle": "chatgpt", "scenario_id": "zdl_chatgpt_pv_curtail_004",
     "signal": "pv_p_kw",  "filename": "chatgpt_pv_p_kw_comparison.png"},
    {"bundle": "claude",   "scenario_id": "zdl_claude_stale_meas_002",
     "signal": "bess_soc_percent", "filename": "claude_bess_soc_percent_comparison.png"},
    {"bundle": "gemini",   "scenario_id": "zdl_gemini_cp_002",
     "signal": "pv_p_kw",  "filename": "gemini_pv_p_kw_comparison.png"},
    {"bundle": "gemini",   "scenario_id": "zdl_gemini_cp_002",
     "signal": "bess_p_kw", "filename": "gemini_bess_p_kw_comparison.png"},
    {"bundle": "grok",     "scenario_id": "zdl_grok_pcc_voltage_004",
     "signal": "pcc_v_a_pu", "filename": "grok_pcc_v_a_pu_comparison.png"},
]


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_data():
    print("[INFO] Loading CSVs...")
    clean   = pd.read_csv(CLEAN_CSV,   usecols=lambda c: True, low_memory=False)
    attacked= pd.read_csv(ATTACKED_CSV,usecols=lambda c: True, low_memory=False)
    manifest= pd.read_csv(MANIFEST_CSV)
    print(f"  Clean   : {len(clean):,} rows")
    print(f"  Attacked: {len(attacked):,} rows")
    print(f"  Manifest: {len(manifest)} scenarios")
    return clean, attacked, manifest


def _get_scenario(manifest: pd.DataFrame, sid: str) -> dict | None:
    rows = manifest[manifest["scenario_id"] == sid]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _extract_context(df: pd.DataFrame, start_s: int, end_s: int,
                     pad_s: int = 600) -> pd.DataFrame:
    """Extract a window with padding on each side."""
    lo = max(0, start_s - pad_s)
    hi = min(604799, end_s + pad_s)
    mask = (df["time_s"] >= lo) & (df["time_s"] <= hi)
    return df[mask].copy()


# ── Single-signal comparison plot ────────────────────────────────────────────

def plot_single_signal(clean: pd.DataFrame, attacked: pd.DataFrame,
                       scenario: dict, signal: str,
                       title: str, outpath: Path,
                       figsize=(11, 4)) -> None:

    start_s = int(scenario["start_time_s"])
    end_s   = int(scenario["end_time_s"])
    pad_s   = max(600, int((end_s - start_s) * 1.5))

    ctx_clean   = _extract_context(clean,   start_s, end_s, pad_s)
    ctx_attacked= _extract_context(attacked, start_s, end_s, pad_s)

    if signal not in ctx_clean.columns or signal not in ctx_attacked.columns:
        print(f"  [SKIP] signal '{signal}' not in columns for {scenario['scenario_id']}")
        return

    t_clean    = ctx_clean["time_s"].values
    t_attacked = ctx_attacked["time_s"].values

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(t_clean,    ctx_clean[signal].values,
                color=CLEAN_COLOR, linewidth=1.6, label="Clean baseline", zorder=3)
        ax.plot(t_attacked, ctx_attacked[signal].values,
                color=ATTACK_COLOR, linewidth=1.4, linestyle="--",
                label="Zero-day attacked", zorder=4)

        ax.axvspan(start_s, end_s, color=SHADE_COLOR, alpha=SHADE_ALPHA,
                   label=f"Attack window [{start_s}s–{end_s}s]", zorder=2)
        ax.axvline(start_s, color=ATTACK_COLOR, linewidth=0.8,
                   linestyle=":", alpha=0.7, zorder=5)
        ax.axvline(end_s,   color=ATTACK_COLOR, linewidth=0.8,
                   linestyle=":", alpha=0.7, zorder=5)

        ax.set_xlabel("Time (seconds since dataset start)", fontsize=11)
        y_label = SIGNAL_LABELS.get(signal, signal)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f"{title}\nScenario: {scenario['scenario_id']}  |  "
                     f"Asset: {scenario.get('target_asset_id','—')}  |  "
                     f"Class: {scenario.get('scenario_class','—')}",
                     fontsize=11)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

        legend = ax.legend(loc="upper left", framealpha=0.9,
                           bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.set_xlim(t_clean[0], t_clean[-1])

        fig.tight_layout(rect=[0, 0, 0.82, 1])
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  [OK] {outpath.name}")


# ── Multi-signal comparison plot ─────────────────────────────────────────────

def plot_multi_signal(clean: pd.DataFrame, attacked: pd.DataFrame,
                      scenario: dict, signals: list,
                      title: str, outpath: Path) -> None:

    available = [s for s in signals
                 if s in clean.columns and s in attacked.columns]
    if not available:
        print(f"  [SKIP] No signals available for {scenario['scenario_id']}")
        return
    if len(available) == 1:
        plot_single_signal(clean, attacked, scenario, available[0],
                           title, outpath)
        return

    n = len(available)
    start_s = int(scenario["start_time_s"])
    end_s   = int(scenario["end_time_s"])
    pad_s   = max(600, int((end_s - start_s) * 1.5))

    ctx_clean    = _extract_context(clean,    start_s, end_s, pad_s)
    ctx_attacked = _extract_context(attacked, start_s, end_s, pad_s)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n),
                                  sharex=True, constrained_layout=True)
        if n == 1:
            axes = [axes]

        for i, sig in enumerate(available):
            ax = axes[i]
            t_c = ctx_clean["time_s"].values
            t_a = ctx_attacked["time_s"].values

            ax.plot(t_c, ctx_clean[sig].values,
                    color=CLEAN_COLOR, linewidth=1.6, label="Clean baseline")
            ax.plot(t_a, ctx_attacked[sig].values,
                    color=ATTACK_COLOR, linewidth=1.4, linestyle="--",
                    label="Zero-day attacked")
            ax.axvspan(start_s, end_s, color=SHADE_COLOR,
                       alpha=SHADE_ALPHA, label="Attack window")
            ax.axvline(start_s, color=ATTACK_COLOR,
                       linewidth=0.8, linestyle=":", alpha=0.7)
            ax.axvline(end_s,   color=ATTACK_COLOR,
                       linewidth=0.8, linestyle=":", alpha=0.7)
            ax.set_ylabel(SIGNAL_LABELS.get(sig, sig), fontsize=10)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
            if i == 0:
                ax.legend(loc="upper left",
                          bbox_to_anchor=(1.01, 1), borderaxespad=0,
                          fontsize=9)

        axes[-1].set_xlabel("Time (seconds since dataset start)", fontsize=11)
        axes[0].set_title(
            f"{title}\n{scenario['scenario_id']}  |  "
            f"Class: {scenario.get('scenario_class','—')}",
            fontsize=11)

        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  [OK] {outpath.name}")


# ── Cross-bundle 2×2 panel ────────────────────────────────────────────────────

def plot_cross_bundle_panel(clean: pd.DataFrame, attacked: pd.DataFrame,
                            manifest: pd.DataFrame) -> None:
    outpath = FIGURES_DIR / "cross_bundle_comparison_panel.png"
    bundles_order = ["chatgpt", "claude", "gemini", "grok"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                                  constrained_layout=True)
        axes_flat = axes.flatten()

        for idx, bundle in enumerate(bundles_order):
            ax = axes_flat[idx]
            cfg = BUNDLE_REPS[bundle]
            sid = cfg["scenario_id"]
            signals = cfg["signals"]
            scen = _get_scenario(manifest, sid)
            if scen is None:
                ax.set_visible(False)
                continue

            sig = signals[0]  # primary signal for panel
            if sig not in clean.columns:
                ax.set_visible(False)
                continue

            start_s = int(scen["start_time_s"])
            end_s   = int(scen["end_time_s"])
            pad_s   = max(600, int((end_s - start_s) * 1.5))

            ctx_c = _extract_context(clean,    start_s, end_s, pad_s)
            ctx_a = _extract_context(attacked, start_s, end_s, pad_s)

            ax.plot(ctx_c["time_s"].values, ctx_c[sig].values,
                    color=CLEAN_COLOR, linewidth=1.4, label="Clean baseline")
            ax.plot(ctx_a["time_s"].values, ctx_a[sig].values,
                    color=ATTACK_COLOR, linewidth=1.2, linestyle="--",
                    label="Zero-day attacked")
            ax.axvspan(start_s, end_s, color=SHADE_COLOR, alpha=SHADE_ALPHA,
                       label="Attack window")
            ax.axvline(start_s, color=ATTACK_COLOR,
                       linewidth=0.7, linestyle=":", alpha=0.7)
            ax.axvline(end_s,   color=ATTACK_COLOR,
                       linewidth=0.7, linestyle=":", alpha=0.7)

            ax.set_title(f"{bundle.upper()}\n{scen['scenario_id']}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(SIGNAL_LABELS.get(sig, sig), fontsize=9)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
            ax.tick_params(labelsize=8)

        # Shared legend outside the grid
        legend_elements = [
            Line2D([0], [0], color=CLEAN_COLOR, linewidth=1.6, label="Clean baseline"),
            Line2D([0], [0], color=ATTACK_COLOR, linewidth=1.4,
                   linestyle="--", label="Zero-day attacked"),
            mpatches.Patch(color=SHADE_COLOR, alpha=0.5, label="Attack window"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=3, fontsize=10, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.04))

        fig.suptitle(
            "Phase 2 Zero-Day: Cross-Bundle Normal vs Attack Comparison\n"
            "(One representative scenario per LLM bundle)",
            fontsize=13, y=1.01)

        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  [OK] {outpath.name}")


# ── Scenario-family comparison ────────────────────────────────────────────────

def plot_family_comparison(clean: pd.DataFrame, attacked: pd.DataFrame,
                           manifest: pd.DataFrame) -> None:
    outpath = FIGURES_DIR / "scenario_family_comparison.png"
    families = list(FAMILY_REPS.keys())

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                                  constrained_layout=True)
        axes_flat = axes.flatten()

        for idx, family in enumerate(families):
            ax = axes_flat[idx]
            cfg = FAMILY_REPS[family]
            sid = cfg["scenario_id"]
            sig = cfg["signal"]
            scen = _get_scenario(manifest, sid)
            if scen is None or sig not in clean.columns:
                ax.set_visible(False)
                continue

            start_s = int(scen["start_time_s"])
            end_s   = int(scen["end_time_s"])
            pad_s   = max(600, int((end_s - start_s) * 1.5))

            ctx_c = _extract_context(clean,    start_s, end_s, pad_s)
            ctx_a = _extract_context(attacked, start_s, end_s, pad_s)

            ax.plot(ctx_c["time_s"].values, ctx_c[sig].values,
                    color=CLEAN_COLOR, linewidth=1.4, label="Clean baseline")
            ax.plot(ctx_a["time_s"].values, ctx_a[sig].values,
                    color=ATTACK_COLOR, linewidth=1.2, linestyle="--",
                    label="Zero-day attacked")
            ax.axvspan(start_s, end_s, color=SHADE_COLOR, alpha=SHADE_ALPHA,
                       label="Attack window")
            ax.axvline(start_s, color=ATTACK_COLOR,
                       linewidth=0.7, linestyle=":", alpha=0.7)
            ax.axvline(end_s,   color=ATTACK_COLOR,
                       linewidth=0.7, linestyle=":", alpha=0.7)

            ax.set_title(f"{cfg['title']}\n{sid}  ({cfg['bundle']})", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(SIGNAL_LABELS.get(sig, sig), fontsize=9)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
            ax.tick_params(labelsize=8)

        legend_elements = [
            Line2D([0], [0], color=CLEAN_COLOR, linewidth=1.6, label="Clean baseline"),
            Line2D([0], [0], color=ATTACK_COLOR, linewidth=1.4,
                   linestyle="--", label="Zero-day attacked"),
            mpatches.Patch(color=SHADE_COLOR, alpha=0.5, label="Attack window"),
        ]
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=3, fontsize=10, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.04))

        fig.suptitle(
            "Phase 2 Zero-Day: Scenario Family Comparison\n"
            "(Representative scenario per anomaly family)",
            fontsize=13, y=1.01)

        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  [OK] {outpath.name}")


# ── Figure index report ───────────────────────────────────────────────────────

def _write_figure_index(index_rows: list) -> None:
    lines = [
        "# ZERO_DAY_FIGURE_INDEX",
        "",
        f"Generated: {NOW}",
        f"Figure directory: `{FIGURES_DIR}`",
        "",
        "## All Generated Figures",
        "",
        "| Filename | Scenario ID | Bundle | Signal(s) | Start–End (s) | Selection reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in index_rows:
        lines.append(
            f"| `{r['filename']}` | {r['scenario_id']} | {r['bundle']} | "
            f"{r['signals']} | {r['start_s']}–{r['end_s']} | {r['reason']} |"
        )
    lines += ["", "---", "*Phase 2 Zero-Day Evaluation — Figure index*"]
    path = REPORTS_DIR / "ZERO_DAY_FIGURE_INDEX.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ATTACKED_CSV.exists():
        print(f"[ERROR] Attacked CSV not found: {ATTACKED_CSV}")
        sys.exit(1)
    if not MANIFEST_CSV.exists():
        print(f"[ERROR] Manifest CSV not found: {MANIFEST_CSV}")
        sys.exit(1)

    clean, attacked, manifest = _load_data()
    index_rows = []

    # ── Per-bundle representative plots ──────────────────────────────────────
    print("\n[INFO] Generating per-bundle plots...")
    for bundle, cfg in BUNDLE_REPS.items():
        sid    = cfg["scenario_id"]
        sigs   = cfg["signals"]
        title  = cfg["title"]
        reason = cfg["reason"]
        scen   = _get_scenario(manifest, sid)
        if scen is None:
            print(f"  [SKIP] {sid} not in manifest")
            continue

        outpath = FIGURES_DIR / f"{bundle}_normal_vs_attack.png"
        plot_multi_signal(clean, attacked, scen, sigs, title, outpath)

        start_s = int(scen["start_time_s"])
        end_s   = int(scen["end_time_s"])
        index_rows.append({
            "filename":   outpath.name,
            "scenario_id": sid,
            "bundle":     bundle,
            "signals":    ", ".join(sigs),
            "start_s":    start_s,
            "end_s":      end_s,
            "reason":     reason,
        })

    # ── Extra per-variable plots ──────────────────────────────────────────────
    print("\n[INFO] Generating per-variable plots...")
    for ep in EXTRA_PLOTS:
        sid   = ep["scenario_id"]
        sig   = ep["signal"]
        scen  = _get_scenario(manifest, sid)
        if scen is None:
            continue
        outpath = FIGURES_DIR / ep["filename"]
        title   = (f"{ep['bundle'].upper()} — {SIGNAL_LABELS.get(sig, sig)}\n"
                   f"Zero-Day Normal vs Attack")
        plot_single_signal(clean, attacked, scen, sig, title, outpath)
        index_rows.append({
            "filename":   ep["filename"],
            "scenario_id": sid,
            "bundle":     ep["bundle"],
            "signals":    sig,
            "start_s":    int(scen["start_time_s"]),
            "end_s":      int(scen["end_time_s"]),
            "reason":     f"Per-variable detail for {sig}",
        })

    # ── Cross-bundle 2×2 panel ────────────────────────────────────────────────
    print("\n[INFO] Generating cross-bundle panel...")
    plot_cross_bundle_panel(clean, attacked, manifest)
    index_rows.append({
        "filename":   "cross_bundle_comparison_panel.png",
        "scenario_id": "multiple",
        "bundle":     "all",
        "signals":    "one per bundle",
        "start_s":    "varies",
        "end_s":      "varies",
        "reason":     "2×2 panel showing one representative scenario per LLM bundle",
    })

    # ── Scenario-family comparison ────────────────────────────────────────────
    print("\n[INFO] Generating scenario family comparison...")
    plot_family_comparison(clean, attacked, manifest)
    index_rows.append({
        "filename":   "scenario_family_comparison.png",
        "scenario_id": "multiple",
        "bundle":     "multiple",
        "signals":    "one per family",
        "start_s":    "varies",
        "end_s":      "varies",
        "reason":     "2×2 panel showing representative scenario per anomaly family",
    })

    # ── Figure index ──────────────────────────────────────────────────────────
    print("\n[INFO] Writing figure index...")
    _write_figure_index(index_rows)

    print(f"\n[DONE] {len(index_rows)} figures generated in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
