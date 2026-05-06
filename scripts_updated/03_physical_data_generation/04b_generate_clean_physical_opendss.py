"""Generate clean 7-day physical timeseries (OpenDSS or surrogate)."""
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

from paths import CLEAN_PHYSICAL_CSV, DER_METADATA_JSON, ENV_CHECK_JSON, REPORTS
from config import DER_SITE_ID, PCC_ID, NOMINAL_VOLTAGE_KV, NOMINAL_FREQUENCY_HZ, BESS_INITIAL_SOC_PERCENT
from time_utils import build_time_index, build_time_s_array, solar_irradiance_pu, temperature_c, load_profile_kw
from physics_utils import compute_soc, compute_voltage_unbalance

TOTAL_ROWS = 604_800


def load_metadata() -> dict:
    if DER_METADATA_JSON.exists():
        with open(DER_METADATA_JSON) as f:
            return json.load(f)
    return {"pv_p_rated_kw": 100.0, "pv_s_rated_kva": 111.11,
            "bess_p_rated_kw": 50.0, "bess_s_rated_kva": 55.56,
            "bess_capacity_kwh": 200.0, "bess_soc_min_percent": 10.0,
            "bess_soc_max_percent": 90.0, "bess_initial_soc_percent": 50.0,
            "bess_eff_charge_percent": 95.0, "bess_eff_discharge_percent": 95.0}


def bess_schedule_kw(time_s: np.ndarray, irradiance: np.ndarray, meta: dict) -> np.ndarray:
    bess_max = meta["bess_p_rated_kw"]
    tod = time_s % 86400
    charge_mask = (tod >= 36000) & (tod < 50400) & (irradiance > 0.5)
    discharge_mask = (tod >= 61200) & (tod < 75600)
    schedule = np.zeros(len(time_s))
    schedule[charge_mask] = -bess_max * 0.7 * irradiance[charge_mask]
    schedule[discharge_mask] = bess_max * 0.7
    noise = np.random.default_rng(101).normal(0, 1.5, len(time_s))
    return np.clip(schedule + noise, -bess_max, bess_max)


def compute_pcc_physics(pv_p, bess_p, load_p, meta, rng):
    n = len(pv_p)
    net_inj_kw = pv_p + bess_p - load_p
    dv_per_kw = 0.00015
    v_mean = np.clip(1.0 + net_inj_kw * dv_per_kw, 0.90, 1.10)
    unb_a = rng.normal(0.002, 0.0005, n)
    unb_b = rng.normal(0.000, 0.0003, n)
    unb_c = rng.normal(-0.002, 0.0005, n)
    v_a = np.clip(v_mean + unb_a, 0.85, 1.15)
    v_b = np.clip(v_mean + unb_b, 0.85, 1.15)
    v_c = np.clip(v_mean + unb_c, 0.85, 1.15)
    load_pf = 0.87
    load_q = load_p * math.tan(math.acos(load_pf))
    pcc_q = load_q + rng.normal(0, 1.5, n)
    pcc_p = load_p - pv_p - bess_p + rng.normal(0, 0.8, n)
    v_ll_kv = v_mean * NOMINAL_VOLTAGE_KV
    i_mag = np.abs(pcc_p) / (math.sqrt(3) * v_ll_kv + 1e-9) * 1000
    i_a = np.clip(i_mag * (1 + rng.normal(0, 0.015, n)), 0, 500)
    i_b = np.clip(i_mag * (1 + rng.normal(0, 0.015, n)), 0, 500)
    i_c = np.clip(i_mag * (1 + rng.normal(0, 0.015, n)), 0, 500)
    feeder_head_p = load_p * 8.0 - pv_p * 0.8 + rng.normal(0, 5.0, n)
    feeder_head_q = feeder_head_p * math.tan(math.acos(load_pf))
    freq = np.full(n, NOMINAL_FREQUENCY_HZ) + rng.normal(0, 0.005, n)
    rated_a = 200.0
    line_loading = np.clip((i_mag / rated_a) * 100.0, 0, 150)
    tap = np.ones(n, dtype=int)
    cap_status = np.where(v_mean < 0.97, 1, 0)
    return {"pcc_v_a_pu": v_a, "pcc_v_b_pu": v_b, "pcc_v_c_pu": v_c,
            "pcc_i_a_amp": i_a, "pcc_i_b_amp": i_b, "pcc_i_c_amp": i_c,
            "pcc_p_kw": pcc_p, "pcc_q_kvar": pcc_q, "pcc_frequency_hz": freq,
            "feeder_head_p_kw": feeder_head_p, "feeder_head_q_kvar": feeder_head_q,
            "line_loading_max_percent": line_loading,
            "regulator_tap_position": tap, "capacitor_status": cap_status}


def build_surrogate_clean(meta: dict) -> pd.DataFrame:
    print("  Building physics-constrained surrogate for clean 7-day timeseries...")
    rng = np.random.default_rng(42)
    rng2 = np.random.default_rng(43)
    time_s = build_time_s_array()
    time_idx = build_time_index()
    irr = solar_irradiance_pu(time_s, rng=np.random.default_rng(10))
    temp = temperature_c(time_s, rng=np.random.default_rng(11))
    load_p = load_profile_kw(time_s, rng=np.random.default_rng(12))
    pv_max_kw = meta["pv_p_rated_kw"]
    pv_s_kva = meta["pv_s_rated_kva"]
    pv_available = irr * pv_max_kw
    pv_actual_p = np.clip(pv_available.copy(), 0, pv_available)
    pv_actual_q = np.zeros(TOTAL_ROWS)
    pv_curtailment = np.maximum(pv_available - pv_actual_p, 0.0)
    pv_s = np.sqrt(pv_actual_p ** 2 + pv_actual_q ** 2)
    pv_violation = (pv_s > pv_s_kva + 0.5).astype(int)
    pv_ramp = np.concatenate([[0.0], np.diff(pv_actual_p)])
    bess_max = meta["bess_p_rated_kw"]
    bess_cmd_p = bess_schedule_kw(time_s, irr, meta)
    bess_actual_p = np.clip(bess_cmd_p, -bess_max, bess_max)
    bess_actual_q = np.zeros(TOTAL_ROWS)
    bess_soc = compute_soc(bess_actual_p, meta, initial_soc=BESS_INITIAL_SOC_PERCENT)
    bess_s_kva = meta["bess_s_rated_kva"]
    bess_s = np.sqrt(bess_actual_p ** 2 + bess_actual_q ** 2)
    bess_violation = (bess_s > bess_s_kva + 0.5).astype(int)
    bess_ramp = np.concatenate([[0.0], np.diff(bess_actual_p)])
    pcc = compute_pcc_physics(pv_actual_p, bess_actual_p, load_p, meta, rng2)
    mean_v, unb, unb_status = compute_voltage_unbalance(pcc["pcc_v_a_pu"], pcc["pcc_v_b_pu"], pcc["pcc_v_c_pu"])
    v_min = np.minimum.reduce([pcc["pcc_v_a_pu"], pcc["pcc_v_b_pu"], pcc["pcc_v_c_pu"]])
    v_max = np.maximum.reduce([pcc["pcc_v_a_pu"], pcc["pcc_v_b_pu"], pcc["pcc_v_c_pu"]])
    df = pd.DataFrame({
        "timestamp_utc": time_idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_s": time_s, "der_site_id": DER_SITE_ID, "pcc_id": PCC_ID,
        "pv_p_kw": pv_actual_p, "pv_q_kvar": pv_actual_q,
        "bess_p_kw": bess_actual_p, "bess_q_kvar": bess_actual_q,
        "bess_soc_percent": bess_soc,
        "pcc_v_a_pu": pcc["pcc_v_a_pu"], "pcc_v_b_pu": pcc["pcc_v_b_pu"], "pcc_v_c_pu": pcc["pcc_v_c_pu"],
        "pcc_i_a_amp": pcc["pcc_i_a_amp"], "pcc_i_b_amp": pcc["pcc_i_b_amp"], "pcc_i_c_amp": pcc["pcc_i_c_amp"],
        "pcc_p_kw": pcc["pcc_p_kw"], "pcc_q_kvar": pcc["pcc_q_kvar"],
        "irradiance_pu": irr, "temperature_c": temp,
        "pv_available_kw": pv_available, "pv_commanded_p_kw": pv_actual_p,
        "pv_actual_p_kw": pv_actual_p, "pv_commanded_q_kvar": pv_actual_q,
        "pv_actual_q_kvar": pv_actual_q, "pv_curtailment_kw": pv_curtailment,
        "pv_s_rated_kva": meta["pv_s_rated_kva"], "pv_inverter_mode": "active_power_follow",
        "pv_ramp_rate_kw_per_s": pv_ramp, "pv_constraint_violation_flag": pv_violation,
        "bess_commanded_p_kw": bess_cmd_p, "bess_actual_p_kw": bess_actual_p,
        "bess_commanded_q_kvar": bess_actual_q, "bess_actual_q_kvar": bess_actual_q,
        "bess_capacity_kwh": meta["bess_capacity_kwh"], "bess_s_rated_kva": meta["bess_s_rated_kva"],
        "bess_soc_min_percent": meta["bess_soc_min_percent"], "bess_soc_max_percent": meta["bess_soc_max_percent"],
        "bess_ramp_rate_kw_per_s": bess_ramp, "bess_constraint_violation_flag": bess_violation,
        "pcc_frequency_hz": pcc["pcc_frequency_hz"],
        "feeder_head_p_kw": pcc["feeder_head_p_kw"], "feeder_head_q_kvar": pcc["feeder_head_q_kvar"],
        "voltage_min_pu": v_min, "voltage_max_pu": v_max,
        "pcc_voltage_mean_pu": mean_v, "pcc_voltage_unbalance_pu": unb,
        "voltage_unbalance_status": unb_status,
        "line_loading_max_percent": pcc["line_loading_max_percent"],
        "regulator_tap_position": pcc["regulator_tap_position"],
        "capacitor_status": pcc["capacitor_status"],
        "physical_scenario_id": "clean_baseline",
        "physical_effect_active_flag": 0, "physical_effect_type": "none",
        "physical_constraint_status": "normal",
        "generation_method": "physics_constrained_surrogate",
    })
    return df


def main() -> pd.DataFrame:
    meta = load_metadata()
    print(f"Using PV={meta['pv_p_rated_kw']} kW, BESS={meta['bess_p_rated_kw']} kW")
    df = None
    if ENV_CHECK_JSON.exists():
        with open(ENV_CHECK_JSON) as f:
            env = json.load(f)
        if env.get("opendss_available", False):
            print("  OpenDSS available but full 7-day QSTS skipped for performance. "
                  "Using physics-constrained surrogate.")
    df = build_surrogate_clean(meta)
    CLEAN_PHYSICAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PHYSICAL_CSV, index=False)
    print(f"Clean physical timeseries: {len(df)} rows, generation_method=physics_constrained_surrogate")
    print(f"Saved: {CLEAN_PHYSICAL_CSV}")
    return df


if __name__ == "__main__":
    main()
