"""
Build cyber-physical lifecycle response map.
Joins cyber anomalous log with physical scenario windows.
Writes: data_updated/processed/cyber_physical_lifecycle_map_7d.csv
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import (
    CYBER_ANOMALOUS_CSV, ATTACKED_PHYSICAL_CSV, SCENARIO_MANIFEST_CSV,
    LIFECYCLE_MAP_CSV, REPORTS,
)
from validation_utils import write_report


def main() -> pd.DataFrame:
    LIFECYCLE_MAP_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not CYBER_ANOMALOUS_CSV.exists():
        print("ERROR: Anomalous cyber log not found.")
        sys.exit(1)
    if not SCENARIO_MANIFEST_CSV.exists():
        print("ERROR: Scenario manifest not found.")
        sys.exit(1)

    cyber = pd.read_csv(CYBER_ANOMALOUS_CSV)
    scenarios = pd.read_csv(SCENARIO_MANIFEST_CSV)

    # Build a per-scenario timing dictionary from cyber log
    timing_rows = []
    for _, sc in scenarios.iterrows():
        sid = sc["scenario_id"]
        sc_cyber = cyber[cyber["scenario_id"] == sid]

        if len(sc_cyber) == 0:
            # No cyber events for this scenario (physical-only)
            timing_rows.append({
                "scenario_id": sid,
                "scenario_name": sc["scenario_name"],
                "scenario_class": sc["scenario_class"],
                "asset_id": sc["asset_id"],
                "asset_type": sc["asset_type"],
                "bus_id": "65",
                "phase": "ABC",
                "pcc_id": sc["pcc_id"],
                "cyber_onset_time_utc": "",
                "command_sent_time_utc": "",
                "command_recv_time_utc": "",
                "command_accept_time_utc": "",
                "command_apply_time_utc": "",
                "command_response_time_utc": "",
                "physical_effect_start_time_utc": sc["start_time_utc"],
                "physical_effect_peak_time_utc": _estimate_peak(sc),
                "physical_effect_end_time_utc": sc["end_time_utc"],
                "label_anomaly": sc["label_anomaly"],
                "label_cyber_anomaly": sc["label_cyber_anomaly"],
                "label_physical_anomaly": sc["label_physical_anomaly"],
                "cia_dimension": sc["cia_dimension"],
                "anomaly_type": sc["scenario_name"],
                "generation_method": "physics_constrained_surrogate",
                "protocol_claim_level": "semantic_ieee2030_5_style",
            })
            continue

        # Get lifecycle stage times from cyber log
        def get_stage_time(stage):
            rows = sc_cyber[sc_cyber["lifecycle_stage"] == stage]
            return rows["event_time_utc"].iloc[0] if len(rows) > 0 else ""

        cyber_onset = get_stage_time("DER_CONTROL_CREATED")
        cmd_sent = get_stage_time("DER_CONTROL_SENT")
        cmd_recv = get_stage_time("DER_CONTROL_RECEIVED")
        cmd_accept = get_stage_time("DER_CONTROL_ACCEPTED")
        cmd_apply = get_stage_time("DER_CONTROL_APPLIED")
        cmd_response = get_stage_time("DER_CONTROL_RESPONSE")

        # Physical effect timing (from scenario)
        phys_start = sc["start_time_utc"]
        phys_end = sc["end_time_utc"]

        # For delayed scenarios, physical start is after command_apply
        if sc["scenario_name"] == "delayed_pv_limit" and cmd_apply:
            phys_start = cmd_apply
        # For blocked: no physical effect
        if sc["scenario_class"] == "cyber_only":
            phys_start = ""
            phys_end = ""

        gen_method = sc_cyber["generation_method"].iloc[0] if "generation_method" in sc_cyber.columns else "physics_constrained_surrogate"

        timing_rows.append({
            "scenario_id": sid,
            "scenario_name": sc["scenario_name"],
            "scenario_class": sc["scenario_class"],
            "asset_id": sc["asset_id"],
            "asset_type": sc["asset_type"],
            "bus_id": "65",
            "phase": "ABC",
            "pcc_id": sc["pcc_id"],
            "cyber_onset_time_utc": cyber_onset,
            "command_sent_time_utc": cmd_sent,
            "command_recv_time_utc": cmd_recv,
            "command_accept_time_utc": cmd_accept,
            "command_apply_time_utc": cmd_apply,
            "command_response_time_utc": cmd_response,
            "physical_effect_start_time_utc": phys_start,
            "physical_effect_peak_time_utc": _estimate_peak(sc),
            "physical_effect_end_time_utc": phys_end,
            "label_anomaly": sc["label_anomaly"],
            "label_cyber_anomaly": sc["label_cyber_anomaly"],
            "label_physical_anomaly": sc["label_physical_anomaly"],
            "cia_dimension": sc["cia_dimension"],
            "anomaly_type": sc["scenario_name"],
            "generation_method": gen_method,
            "protocol_claim_level": "semantic_ieee2030_5_style",
        })

    df = pd.DataFrame(timing_rows)
    df.to_csv(LIFECYCLE_MAP_CSV, index=False)

    lines = [
        "# Context Mapping Improvement Report",
        "",
        "## Lifecycle Response Map",
        "",
        f"- Total scenarios mapped: {len(df)}",
        f"- Cyber-physical scenarios: {(df['scenario_class'] == 'cyber_physical').sum()}",
        f"- Physical-only scenarios: {(df['scenario_class'] == 'physical_only').sum()}",
        f"- Cyber-only scenarios: {(df['scenario_class'] == 'cyber_only').sum()}",
        f"- Normal scenarios: {(df['scenario_class'] == 'normal').sum()}",
        "",
        f"**Output:** `{LIFECYCLE_MAP_CSV}`",
    ]
    write_report(lines, REPORTS / "04_context_mapping_improvement_report.md")

    print(f"Lifecycle map: {len(df)} rows")
    print(f"Saved: {LIFECYCLE_MAP_CSV}")
    return df


def _estimate_peak(sc) -> str:
    """Estimate physical effect peak time as midpoint of scenario."""
    try:
        s = int(sc["start_time_s"])
        e = int(sc["end_time_s"])
        mid = (s + e) // 2
        return (pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(seconds=mid)).isoformat() + "Z"
    except Exception:
        return str(sc.get("start_time_utc", ""))


if __name__ == "__main__":
    main()
