"""Generate attacked 7-day physical timeseries via event-window simulation."""
import sys
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, DER_METADATA_JSON, SCENARIO_MANIFEST_CSV, REPORTS
from config import DER_SITE_ID, PCC_ID, NOMINAL_VOLTAGE_KV
from physics_utils import compute_soc, compute_voltage_unbalance

TOTAL_ROWS = 604_800
TOLERANCE_KVA = 0.5


def load_metadata() -> dict:
    if DER_METADATA_JSON.exists():
        with open(DER_METADATA_JSON) as f:
            return json.load(f)
    return {"pv_p_rated_kw": 100.0, "pv_s_rated_kva": 111.11,
            "bess_p_rated_kw": 50.0, "bess_s_rated_kva": 55.56,
            "bess_capacity_kwh": 200.0, "bess_soc_min_percent": 10.0,
            "bess_soc_max_percent": 90.0, "bess_initial_soc_percent": 50.0,
            "bess_eff_charge_percent": 95.0, "bess_eff_discharge_percent": 95.0}


def apply_irradiance_drop(df, s, e, meta, rng):
    mask = (df["time_s"] >= s) & (df["time_s"] < e)
    drop = rng.uniform(0.1, 0.5)
    df.loc[mask, "irradiance_pu"] *= drop
    df.loc[mask, "pv_available_kw"] = df.loc[mask, "irradiance_pu"] * meta["pv_p_rated_kw"]
    df.loc[mask, "pv_actual_p_kw"] = np.minimum(df.loc[mask, "pv_commanded_p_kw"].values, df.loc[mask, "pv_available_kw"].values)
    df.loc[mask, "pv_p_kw"] = df.loc[mask, "pv_actual_p_kw"]
    df.loc[mask, "pv_curtailment_kw"] = np.maximum(df.loc[mask, "pv_available_kw"].values - df.loc[mask, "pv_actual_p_kw"].values, 0.0)
    df.loc[mask, "physical_effect_active_flag"] = 1
    df.loc[mask, "physical_effect_type"] = "irradiance_drop"


def apply_load_step(df, s, e, meta, rng):
    mask = (df["time_s"] >= s) & (df["time_s"] < e)
    step_kw = rng.choice([-1, 1]) * rng.uniform(30, 80)
    df.loc[mask, "pcc_p_kw"] = df.loc[mask, "pcc_p_kw"] + step_kw
    v_mean = df.loc[mask, "pcc_voltage_mean_pu"].values
    v_ll_kv = v_mean * NOMINAL_VOLTAGE_KV
    i_mag = np.abs(df.loc[mask, "pcc_p_kw"].values) / (math.sqrt(3) * v_ll_kv + 1e-9) * 1000
    df.loc[mask, "pcc_i_a_amp"] = np.clip(i_mag, 0, 500)
    df.loc[mask, "pcc_i_b_amp"] = np.clip(i_mag, 0, 500)
    df.loc[mask, "pcc_i_c_amp"] = np.clip(i_mag, 0, 500)
    df.loc[mask, "physical_effect_active_flag"] = 1
    df.loc[mask, "physical_effect_type"] = "load_step"


def apply_delayed_pv_limit(df, s, e, meta, delay_s, rng):
    apply_start = int(s + delay_s)
    mask = (df["time_s"] >= apply_start) & (df["time_s"] < e)
    cmd_p = rng.uniform(0, meta["pv_p_rated_kw"] * 0.7)
    df.loc[mask, "pv_commanded_p_kw"] = cmd_p
    df.loc[mask, "pv_actual_p_kw"] = np.minimum(cmd_p, df.loc[mask, "pv_available_kw"].values)
    df.loc[mask, "pv_p_kw"] = df.loc[mask, "pv_actual_p_kw"]
    df.loc[mask, "physical_effect_active_flag"] = 1
    df.loc[mask, "physical_effect_type"] = "delayed_response"


def apply_wrong_pv_setpoint(df, s, e, meta, rng):
    mask = (df["time_s"] >= s) & (df["time_s"] < e)
    expected = rng.uniform(40, meta["pv_p_rated_kw"])
    wrong = rng.uniform(0, meta["pv_p_rated_kw"])
    df.loc[mask, "pv_commanded_p_kw"] = expected
    df.loc[mask, "pv_actual_p_kw"] = np.minimum(wrong, df.loc[mask, "pv_available_kw"].values)
    df.loc[mask, "pv_p_kw"] = df.loc[mask, "pv_actual_p_kw"]
    df.loc[mask, "physical_effect_active_flag"] = 1
    df.loc[mask, "physical_effect_type"] = "wrong_setpoint"


def apply_bess_wrong_direction(df, s, e, meta, rng):
    mask = (df["time_s"] >= s) & (df["time_s"] < e)
    cmd_p = rng.uniform(10, meta["bess_p_rated_kw"])
    df.loc[mask, "bess_commanded_p_kw"] = cmd_p
    df.loc[mask, "bess_actual_p_kw"] = -cmd_p
    df.loc[mask, "bess_p_kw"] = -cmd_p
    df.loc[mask, "physical_effect_active_flag"] = 1
    df.loc[mask, "physical_effect_type"] = "wrong_dispatch"


def apply_replay_command(df, s, e, meta, rng):
    mask = (df["time_s"] >= s) & (df["time_s"] < e)
    stale_p = rng.uniform(0, meta["pv_p_rated_kw"] * 0.5)
    df.loc[mask, "pv_actual_p_kw"] = np.minimum(stale_p, df.loc[mask, "pv_available_kw"].values)
    df.loc[mask, "pv_p_kw"] = df.loc[mask, "pv_actual_p_kw"]
    df.loc[mask, "physical_effect_active_flag"] = 1
    df.loc[mask, "physical_effect_type"] = "stale_setpoint"


def apply_high_rate_burst(df, s, e, meta, rng):
    mask_idx = df.index[(df["time_s"] >= s) & (df["time_s"] < e)]
    period = 5
    for i, idx in enumerate(mask_idx):
        phase = (i // period) % 2
        new_p = meta["pv_p_rated_kw"] * (0.8 if phase == 0 else 0.2)
        av = df.at[idx, "pv_available_kw"]
        df.at[idx, "pv_actual_p_kw"] = min(new_p, av)
        df.at[idx, "pv_p_kw"] = min(new_p, av)
        df.at[idx, "physical_effect_active_flag"] = 1
        df.at[idx, "physical_effect_type"] = "oscillating_output"


def apply_soc_constraint_violation(df, s, e, meta, rng):
    mask = (df["time_s"] >= s) & (df["time_s"] < e)
    cmd_p = meta["bess_p_rated_kw"]
    df.loc[mask, "bess_commanded_p_kw"] = cmd_p
    soc_vals = df.loc[mask, "bess_soc_percent"].values
    actual_p = np.where(soc_vals <= meta["bess_soc_min_percent"] + 2.0, 0.0, cmd_p)
    df.loc[mask, "bess_actual_p_kw"] = actual_p
    df.loc[mask, "bess_p_kw"] = actual_p
    df.loc[mask, "bess_constraint_violation_flag"] = (soc_vals <= meta["bess_soc_min_percent"] + 2.0).astype(int)
    df.loc[mask, "physical_effect_active_flag"] = 1
    df.loc[mask, "physical_effect_type"] = "soc_violation"
    df.loc[mask, "physical_constraint_status"] = "soc_min_violation"


def apply_voltage_sag(df, s, e, meta, rng):
    mask = (df["time_s"] >= s) & (df["time_s"] < e)
    sag = rng.uniform(0.08, 0.15)
    for col in ["pcc_v_a_pu", "pcc_v_b_pu", "pcc_v_c_pu"]:
        df.loc[mask, col] = np.clip(df.loc[mask, col] - sag, 0.50, 1.15)
    va = df.loc[mask, "pcc_v_a_pu"].values
    vb = df.loc[mask, "pcc_v_b_pu"].values
    vc = df.loc[mask, "pcc_v_c_pu"].values
    mean_v, unb, unb_status = compute_voltage_unbalance(va, vb, vc)
    df.loc[mask, "pcc_voltage_mean_pu"] = mean_v
    df.loc[mask, "pcc_voltage_unbalance_pu"] = unb
    df.loc[mask, "voltage_unbalance_status"] = unb_status
    df.loc[mask, "voltage_min_pu"] = np.minimum.reduce([va, vb, vc])
    df.loc[mask, "voltage_max_pu"] = np.maximum.reduce([va, vb, vc])
    i_adj = df.loc[mask, "pcc_i_a_amp"].values * (1.0 / np.maximum(mean_v, 0.01))
    for col in ["pcc_i_a_amp", "pcc_i_b_amp", "pcc_i_c_amp"]:
        df.loc[mask, col] = np.clip(i_adj, 0, 800)
    df.loc[mask, "physical_effect_active_flag"] = 1
    df.loc[mask, "physical_effect_type"] = "voltage_sag"


def recompute_soc_from_actual(df, meta):
    soc = compute_soc(df["bess_actual_p_kw"].values, meta, initial_soc=meta.get("bess_initial_soc_percent", 50.0))
    df["bess_soc_percent"] = soc
    return df


def recompute_voltage_unbalance_global(df):
    va = df["pcc_v_a_pu"].values
    vb = df["pcc_v_b_pu"].values
    vc = df["pcc_v_c_pu"].values
    mean_v, unb, unb_status = compute_voltage_unbalance(va, vb, vc)
    df["pcc_voltage_mean_pu"] = mean_v
    df["pcc_voltage_unbalance_pu"] = unb
    df["voltage_unbalance_status"] = unb_status
    df["voltage_min_pu"] = np.minimum.reduce([va, vb, vc])
    df["voltage_max_pu"] = np.maximum.reduce([va, vb, vc])
    return df


def main() -> pd.DataFrame:
    meta = load_metadata()
    rng = np.random.default_rng(99)

    if not CLEAN_PHYSICAL_CSV.exists():
        print("ERROR: Clean physical CSV not found. Run 04b first.")
        sys.exit(1)
    if not SCENARIO_MANIFEST_CSV.exists():
        print("ERROR: Scenario manifest not found. Run 02 first.")
        sys.exit(1)

    print("Loading clean physical timeseries...")
    df = pd.read_csv(CLEAN_PHYSICAL_CSV, dtype={"voltage_unbalance_status": str, "pv_inverter_mode": str})
    df = df.copy()
    scenarios = pd.read_csv(SCENARIO_MANIFEST_CSV)
    print(f"Applying {len(scenarios)} scenarios...")

    surrogate_count = 0
    skipped = 0

    for _, row in scenarios.iterrows():
        s = int(row["start_time_s"])
        e = int(row["end_time_s"])
        sname = row["scenario_name"]
        delay_s = float(row.get("delay_s", 0))
        mask = (df["time_s"] >= s) & (df["time_s"] < e)
        df.loc[mask, "physical_scenario_id"] = row["scenario_id"]

        if sname in ("normal_legit_command", "unauthorized_blocked"):
            df.loc[mask, "generation_method"] = "physics_constrained_surrogate"
            surrogate_count += 1
            continue

        try:
            if sname == "physical_irradiance_drop":
                apply_irradiance_drop(df, s, e, meta, rng)
            elif sname == "physical_load_step":
                apply_load_step(df, s, e, meta, rng)
            elif sname == "delayed_pv_limit":
                apply_delayed_pv_limit(df, s, e, meta, delay_s, rng)
            elif sname == "wrong_pv_setpoint":
                apply_wrong_pv_setpoint(df, s, e, meta, rng)
            elif sname == "bess_wrong_direction":
                apply_bess_wrong_direction(df, s, e, meta, rng)
            elif sname == "replay_command":
                apply_replay_command(df, s, e, meta, rng)
            elif sname == "high_rate_command_burst":
                apply_high_rate_burst(df, s, e, meta, rng)
            elif sname == "soc_constraint_violation":
                apply_soc_constraint_violation(df, s, e, meta, rng)
            elif sname == "voltage_sag":
                apply_voltage_sag(df, s, e, meta, rng)
            else:
                skipped += 1
                continue
            df.loc[mask, "generation_method"] = "physics_constrained_surrogate"
            surrogate_count += 1
        except Exception as exc:
            print(f"  WARN: Failed {sname}: {exc}")
            skipped += 1

    df = recompute_soc_from_actual(df, meta)
    df = recompute_voltage_unbalance_global(df)
    df["pv_ramp_rate_kw_per_s"] = np.concatenate([[0.0], np.diff(df["pv_actual_p_kw"].values)])
    df["bess_ramp_rate_kw_per_s"] = np.concatenate([[0.0], np.diff(df["bess_actual_p_kw"].values)])
    df["pv_curtailment_kw"] = np.maximum(df["pv_available_kw"] - df["pv_actual_p_kw"], 0.0)
    pv_s = np.sqrt(df["pv_actual_p_kw"] ** 2 + df["pv_actual_q_kvar"] ** 2)
    df["pv_constraint_violation_flag"] = (pv_s > meta["pv_s_rated_kva"] + TOLERANCE_KVA).astype(int)
    bess_s = np.sqrt(df["bess_actual_p_kw"] ** 2 + df["bess_actual_q_kvar"] ** 2)
    df["bess_constraint_violation_flag"] = (bess_s > meta["bess_s_rated_kva"] + TOLERANCE_KVA).astype(int)

    ATTACKED_PHYSICAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ATTACKED_PHYSICAL_CSV, index=False)
    print(f"Attacked timeseries: {len(df)} rows, surrogate={surrogate_count}, skipped={skipped}")
    print(f"Saved: {ATTACKED_PHYSICAL_CSV}")

    report_path = REPORTS / "02_physical_layer_improvement_report.md"
    with open(report_path, "a") as f:
        f.write(f"\n\n## Attacked Physical Data\n\n"
                f"- Scenarios applied: {len(scenarios)}\n"
                f"- OpenDSS event-window: 0\n"
                f"- Surrogate fallback: {surrogate_count}\n"
                f"- Skipped: {skipped}\n"
                f"\n> **NOTE:** OpenDSS event-window simulation not run. "
                f"Default final claim applies.\n")
    return df


if __name__ == "__main__":
    main()
