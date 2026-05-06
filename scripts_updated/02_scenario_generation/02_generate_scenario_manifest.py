"""
Generate the scenario manifest for 7-day DER cyber-physical dataset.
"""
import sys
import json
import random
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import SCENARIO_MANIFEST_CSV, SCENARIO_MANIFEST_JSON, REPORTS, SCENARIOS
from config import (
    DER_SITE_ID, PV_ASSET_ID, BESS_ASSET_ID, PCC_ID,
    PV_P_RATED_KW, PV_S_RATED_KVA, BESS_P_RATED_KW, BESS_S_RATED_KVA,
    RANDOM_SEED,
)
from validation_utils import write_report

START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")
TOTAL_S = 604_800
rng = random.Random(RANDOM_SEED)

SCENARIO_TYPES = [
    ("normal_legit_command",     "normal",         "pv_001",   0, 0, 0, "none",  "none",              "none",            "none",   (10, 60),   20),
    ("unauthorized_blocked",     "cyber_only",     "pv_001",   1, 1, 0, "CIA_A", "none",              "blocked_command", "medium", (5, 30),    15),
    ("physical_irradiance_drop", "physical_only",  "pv_001",   1, 0, 1, "CIA_A", "irradiance_drop",   "none",            "medium", (120, 600), 15),
    ("physical_load_step",       "physical_only",  "pcc_001",  1, 0, 1, "CIA_A", "load_step",         "none",            "low",    (60, 300),  15),
    ("delayed_pv_limit",         "cyber_physical", "pv_001",   1, 1, 1, "CIA_A", "delayed_response",  "delayed_command", "medium", (60, 300),  15),
    ("wrong_pv_setpoint",        "cyber_physical", "pv_001",   1, 1, 1, "CIA_I", "wrong_setpoint",    "integrity_attack","high",   (30, 180),  15),
    ("bess_wrong_direction",     "cyber_physical", "bess_001", 1, 1, 1, "CIA_I", "wrong_dispatch",    "integrity_attack","high",   (60, 300),  15),
    ("replay_command",           "cyber_physical", "pv_001",   1, 1, 1, "CIA_I", "stale_setpoint",    "replay_attack",   "medium", (10, 60),   15),
    ("high_rate_command_burst",  "cyber_physical", "pv_001",   1, 1, 1, "CIA_A", "oscillating_output","dos_burst",       "medium", (30, 180),  15),
    ("soc_constraint_violation", "cyber_physical", "bess_001", 1, 1, 1, "CIA_I", "soc_violation",     "constraint_bypass","high", (30, 120),  15),
    ("voltage_sag",              "physical_only",  "pcc_001",  1, 0, 1, "CIA_A", "voltage_sag",       "none",            "medium", (30, 300),  15),
]

COLS = [
    "scenario_id", "scenario_name", "scenario_class", "asset_id", "asset_type",
    "der_site_id", "pcc_id", "start_time_utc", "end_time_utc", "start_time_s", "end_time_s",
    "duration_s", "command_type", "expected_p_kw", "expected_q_kvar", "applied_p_kw",
    "applied_q_kvar", "delay_s", "label_anomaly", "label_cyber_anomaly", "label_physical_anomaly",
    "cia_dimension", "physical_effect_type", "cyber_effect_type", "severity_level", "description",
]


def asset_type_str(asset_id):
    return "PVSystem" if "pv" in asset_id else "BESS" if "bess" in asset_id else "PCC"


def random_p_command(asset_id, sname):
    if "pv" in asset_id:
        exp = round(rng.uniform(0, PV_P_RATED_KW), 2)
        if "wrong_setpoint" in sname:
            return exp, 0.0, round(rng.uniform(0, PV_P_RATED_KW), 2), 0.0
        if "replay" in sname:
            return exp, 0.0, round(rng.uniform(0, PV_P_RATED_KW * 0.5), 2), 0.0
        return exp, 0.0, exp, 0.0
    if "bess" in asset_id:
        exp = round(rng.uniform(5, BESS_P_RATED_KW), 2)
        if "wrong_direction" in sname:
            return exp, 0.0, -exp, 0.0
        return exp, 0.0, exp, 0.0
    return 0.0, 0.0, 0.0, 0.0


def cmd_type(sname, asset_id):
    if "bess" in asset_id:
        return "storage_dispatch"
    if "setpoint" in sname or "limit" in sname or "pv" in sname:
        return "active_power_limit"
    return "active_power_limit"


def build_scenarios():
    rows = []
    used = []

    def no_overlap(s, e):
        for us, ue in used:
            if not (e <= us or s >= ue):
                return False
        return True

    idx = 0
    for (sname, sclass, asset_id, l_a, l_c, l_p,
         cia, phys_eff, cyber_eff, sev, dur_range, count) in SCENARIO_TYPES:
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 30:
            attempts += 1
            dur = rng.randint(*dur_range)
            max_start = TOTAL_S - dur - 600
            if max_start < 600:
                break
            start_s = rng.randint(600, max_start)
            end_s = start_s + dur
            if not no_overlap(start_s, end_s):
                continue
            used.append((start_s, end_s))
            idx += 1
            sid = f"S{idx:04d}_{sname[:20]}"
            s_utc = (START_UTC + pd.Timedelta(seconds=start_s)).isoformat() + "Z"
            e_utc = (START_UTC + pd.Timedelta(seconds=end_s)).isoformat() + "Z"
            exp_p, exp_q, app_p, app_q = random_p_command(asset_id, sname)
            delay_s = 0.0
            if "delayed" in sname:
                delay_s = round(rng.uniform(5, 30), 1)
            elif "replay" in sname:
                delay_s = round(rng.uniform(60, 600), 1)
            rows.append({
                "scenario_id": sid, "scenario_name": sname, "scenario_class": sclass,
                "asset_id": asset_id, "asset_type": asset_type_str(asset_id),
                "der_site_id": DER_SITE_ID, "pcc_id": PCC_ID,
                "start_time_utc": s_utc, "end_time_utc": e_utc,
                "start_time_s": start_s, "end_time_s": end_s, "duration_s": dur,
                "command_type": cmd_type(sname, asset_id),
                "expected_p_kw": exp_p, "expected_q_kvar": exp_q,
                "applied_p_kw": app_p, "applied_q_kvar": app_q,
                "delay_s": delay_s, "label_anomaly": l_a,
                "label_cyber_anomaly": l_c, "label_physical_anomaly": l_p,
                "cia_dimension": cia, "physical_effect_type": phys_eff,
                "cyber_effect_type": cyber_eff, "severity_level": sev,
                "description": f"{sname} on {asset_id} t={start_s}–{end_s}s ({dur}s)",
            })
            placed += 1
    return rows


def main():
    SCENARIOS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = build_scenarios()
    df = pd.DataFrame(rows, columns=COLS).sort_values("start_time_s").reset_index(drop=True)
    df.to_csv(SCENARIO_MANIFEST_CSV, index=False)
    with open(SCENARIO_MANIFEST_JSON, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
    counts = df.groupby("scenario_name").size().to_dict()
    lines = [
        "# Scenario Manifest Report", "",
        f"**Total scenarios:** {len(df)}", "",
        "## Counts by Scenario Name", "",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"- `{k}`: {v}")
    write_report(lines, REPORTS / "01_scenario_manifest_report.md")
    print(f"Generated {len(df)} scenarios across {len(counts)} types.")
    return df


if __name__ == "__main__":
    main()
