"""
Build event-specific context windows for each scenario.
Each window contains pre-event, event, and post-event physical rows with
relative timing and cyber-physical correlation columns.
Writes: data_updated/processed/event_specific_context_windows_7d.csv
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
    CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, LIFECYCLE_MAP_CSV,
    CONTEXT_WINDOWS_CSV,
)
from config import PRE_EVENT_BUFFER_S, POST_EVENT_BUFFER_S

START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")


def utc_to_s(utc_str: str) -> int:
    """Convert UTC ISO string to seconds offset."""
    if not utc_str or pd.isna(utc_str):
        return -1
    try:
        t = pd.Timestamp(utc_str)
        return int((t - START_UTC).total_seconds())
    except Exception:
        return -1


def main() -> pd.DataFrame:
    CONTEXT_WINDOWS_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not LIFECYCLE_MAP_CSV.exists():
        print("ERROR: Lifecycle map not found. Run 06a first.")
        sys.exit(1)
    if not ATTACKED_PHYSICAL_CSV.exists():
        print("ERROR: Attacked physical CSV not found.")
        sys.exit(1)
    if not CLEAN_PHYSICAL_CSV.exists():
        print("ERROR: Clean physical CSV not found.")
        sys.exit(1)

    print("Loading physical data...")
    lifecycle = pd.read_csv(LIFECYCLE_MAP_CSV)

    # Load only relevant columns from physical CSVs to save memory
    phys_cols = [
        "time_s", "timestamp_utc", "der_site_id", "pcc_id",
        "pv_actual_p_kw", "pv_actual_q_kvar", "bess_actual_p_kw",
        "bess_actual_q_kvar", "bess_soc_percent",
        "pcc_v_a_pu", "pcc_v_b_pu", "pcc_v_c_pu",
        "pcc_p_kw", "pcc_q_kvar",
        "pcc_voltage_mean_pu", "pcc_voltage_unbalance_pu",
        "physical_scenario_id", "physical_effect_active_flag",
        "physical_effect_type", "generation_method",
    ]

    clean = pd.read_csv(CLEAN_PHYSICAL_CSV, usecols=[c for c in phys_cols if c != "physical_scenario_id"])
    attacked = pd.read_csv(ATTACKED_PHYSICAL_CSV, usecols=phys_cols)

    clean_dict = {col: clean[col].values for col in clean.columns}
    attacked_dict = {col: attacked[col].values for col in attacked.columns}
    time_s_arr = clean["time_s"].values

    all_windows = []

    print(f"Building context windows for {len(lifecycle)} scenarios...")
    for _, sc in lifecycle.iterrows():
        scenario_id = sc["scenario_id"]
        cmd_apply_s = utc_to_s(str(sc.get("command_apply_time_utc", "")))
        phys_start_s = utc_to_s(str(sc.get("physical_effect_start_time_utc", "")))
        phys_end_s = utc_to_s(str(sc.get("physical_effect_end_time_utc", "")))
        cyber_onset_s = utc_to_s(str(sc.get("cyber_onset_time_utc", "")))
        cmd_sent_s = utc_to_s(str(sc.get("command_sent_time_utc", "")))

        # Window bounds
        anchor_s = phys_start_s if phys_start_s > 0 else cmd_apply_s
        if anchor_s < 0:
            # Try to get start from scenario timing
            try:
                anchor_s = utc_to_s(str(sc.get("physical_effect_start_time_utc", "")))
            except Exception:
                anchor_s = 0

        win_start = max(0, anchor_s - PRE_EVENT_BUFFER_S)
        win_end = min(604_799, anchor_s + POST_EVENT_BUFFER_S)
        if phys_end_s > 0:
            win_end = min(604_799, phys_end_s + PRE_EVENT_BUFFER_S)

        # Extract row indices in window
        idx_mask = (time_s_arr >= win_start) & (time_s_arr <= win_end)
        idx_arr = np.where(idx_mask)[0]
        if len(idx_arr) == 0:
            continue

        for idx in idx_arr:
            t_s = int(time_s_arr[idx])
            row = {
                "scenario_id": scenario_id,
                "event_id": "",
                "message_mrid": "",
                "transaction_id": "",
                "asset_id": sc["asset_id"],
                "asset_type": sc["asset_type"],
                "bus_id": sc.get("bus_id", "65"),
                "phase": "ABC",
                "pcc_id": sc["pcc_id"],
                "window_start_utc": (START_UTC + pd.Timedelta(seconds=win_start)).isoformat() + "Z",
                "window_end_utc": (START_UTC + pd.Timedelta(seconds=win_end)).isoformat() + "Z",
                "relative_time_to_command_s": (t_s - cmd_sent_s) if cmd_sent_s > 0 else 0,
                "relative_time_to_apply_s": (t_s - cmd_apply_s) if cmd_apply_s > 0 else 0,
                "relative_time_to_physical_effect_s": (t_s - phys_start_s) if phys_start_s > 0 else 0,
                # Cyber flag (active during cyber lifecycle window)
                "cyber_active_flag": int(
                    cyber_onset_s > 0 and cmd_sent_s <= t_s <= cmd_sent_s + 120
                ),
                "command_active_flag": int(
                    cmd_sent_s > 0 and cmd_sent_s <= t_s <= cmd_apply_s + 60
                    if cmd_apply_s > 0 else (cmd_sent_s <= t_s <= cmd_sent_s + 120)
                ),
                "physical_effect_active_flag": int(
                    phys_start_s > 0 and phys_start_s <= t_s <= phys_end_s
                ) if phys_end_s > 0 and phys_start_s > 0 else 0,
                "attack_active_flag": int(
                    sc["label_anomaly"] == 1 and (
                        (phys_start_s > 0 and phys_start_s <= t_s <= phys_end_s) or
                        (cyber_onset_s > 0 and cyber_onset_s <= t_s <= cyber_onset_s + 60)
                    )
                ),
                # Physical baselines (from clean dataset)
                "baseline_p_kw": float(clean_dict["pv_actual_p_kw"][idx]),
                "actual_p_kw": float(attacked_dict["pv_actual_p_kw"][idx]),
                "residual_p_kw": float(
                    attacked_dict["pv_actual_p_kw"][idx] - clean_dict["pv_actual_p_kw"][idx]
                ),
                "baseline_q_kvar": float(clean_dict["pv_actual_q_kvar"][idx]),
                "actual_q_kvar": float(attacked_dict["pv_actual_q_kvar"][idx]),
                "residual_q_kvar": float(
                    attacked_dict["pv_actual_q_kvar"][idx] - clean_dict["pv_actual_q_kvar"][idx]
                ),
                "baseline_voltage_pu": float(clean_dict["pcc_voltage_mean_pu"][idx]),
                "actual_voltage_pu": float(attacked_dict["pcc_voltage_mean_pu"][idx]),
                "residual_voltage_pu": float(
                    attacked_dict["pcc_voltage_mean_pu"][idx] - clean_dict["pcc_voltage_mean_pu"][idx]
                ),
                "baseline_soc_percent": float(clean_dict["bess_soc_percent"][idx]),
                "actual_soc_percent": float(attacked_dict["bess_soc_percent"][idx]),
                "residual_soc_percent": float(
                    attacked_dict["bess_soc_percent"][idx] - clean_dict["bess_soc_percent"][idx]
                ),
                "constraint_violation_flag": 0,
                "violated_constraint_name": "",
                "evidence_strength_score": float(abs(
                    attacked_dict["pv_actual_p_kw"][idx] - clean_dict["pv_actual_p_kw"][idx]
                ) / (abs(clean_dict["pv_actual_p_kw"][idx]) + 1.0)),
                "final_event_class": sc["scenario_class"],
                "anomaly_type": sc["anomaly_type"],
                "generation_method": sc.get("generation_method", "physics_constrained_surrogate"),
                "protocol_claim_level": "semantic_ieee2030_5_style",
                # Timing evidence
                "cyber_onset_time_utc": str(sc.get("cyber_onset_time_utc", "")),
                "command_sent_time_utc": str(sc.get("command_sent_time_utc", "")),
                "command_recv_time_utc": str(sc.get("command_recv_time_utc", "")),
                "command_accept_time_utc": str(sc.get("command_accept_time_utc", "")),
                "command_apply_time_utc": str(sc.get("command_apply_time_utc", "")),
                "command_response_time_utc": str(sc.get("command_response_time_utc", "")),
                "physical_effect_start_time_utc": str(sc.get("physical_effect_start_time_utc", "")),
                "physical_effect_peak_time_utc": str(sc.get("physical_effect_peak_time_utc", "")),
                "physical_effect_end_time_utc": str(sc.get("physical_effect_end_time_utc", "")),
            }
            all_windows.append(row)

    df = pd.DataFrame(all_windows)
    df.to_csv(CONTEXT_WINDOWS_CSV, index=False)
    print(f"Context windows: {len(df)} rows across {len(lifecycle)} scenarios")
    print(f"Saved: {CONTEXT_WINDOWS_CSV}")
    return df


if __name__ == "__main__":
    main()
