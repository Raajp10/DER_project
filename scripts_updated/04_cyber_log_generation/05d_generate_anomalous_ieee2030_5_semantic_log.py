"""
Generate anomalous IEEE 2030.5-style semantic cyber event log.
One lifecycle per anomalous scenario, with appropriate anomaly flags.
Writes: data_updated/raw/cyber_event_log_anomalous_ieee2030_5_semantic_7d.csv
        reports/03_cyber_layer_improvement_report.md
"""
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import (
    CYBER_ANOMALOUS_CSV, SCENARIO_MANIFEST_CSV, REPORTS,
)
from config import DER_SITE_ID, PCC_ID, RANDOM_SEED
from validation_utils import write_report

_05b_path = ROOT / "scripts_updated" / "04_cyber_log_generation" / "05b_generate_der_command_lifecycle.py"
if "generate_der_command_lifecycle" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("generate_der_command_lifecycle", _05b_path)
    _lifecycle_mod = importlib.util.module_from_spec(_spec)
    sys.modules["generate_der_command_lifecycle"] = _lifecycle_mod
    _spec.loader.exec_module(_lifecycle_mod)
else:
    _lifecycle_mod = sys.modules["generate_der_command_lifecycle"]
build_control_lifecycle = _lifecycle_mod.build_control_lifecycle
build_meter_reading = _lifecycle_mod.build_meter_reading
make_mrid = _lifecycle_mod.make_mrid

rng = np.random.default_rng(RANDOM_SEED + 1)

SCENARIO_LIFECYCLE_MAP = {
    "normal_legit_command":     "normal",
    "unauthorized_blocked":     "blocked",
    "physical_irradiance_drop": "normal",       # cyber shows normal monitoring
    "physical_load_step":       "normal",
    "delayed_pv_limit":         "delayed",
    "wrong_pv_setpoint":        "wrong_setpoint",
    "bess_wrong_direction":     "wrong_setpoint",
    "replay_command":           "replay",
    "high_rate_command_burst":  "burst",
    "soc_constraint_violation": "soc_violation",
    "voltage_sag":              "normal",
}


def prev_mrid_for_replay(history: list) -> str:
    """Return an old message MRID for replay scenarios."""
    if len(history) > 5:
        return history[-5].get("message_mrid", "")
    return make_mrid()


def main() -> pd.DataFrame:
    CYBER_ANOMALOUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not SCENARIO_MANIFEST_CSV.exists():
        print("ERROR: Scenario manifest not found.")
        sys.exit(1)

    scenarios = pd.read_csv(SCENARIO_MANIFEST_CSV)
    all_rows = []
    history = []

    for _, s in scenarios.iterrows():
        sname = s["scenario_name"]
        lifecycle_type = SCENARIO_LIFECYCLE_MAP.get(sname, "normal")
        start_s = int(s["start_time_s"])
        delay_s = float(s.get("delay_s", 0))
        asset_id = s["asset_id"]
        target_p = float(s.get("expected_p_kw", 0))
        target_q = float(s.get("expected_q_kvar", 0))
        applied_p = float(s.get("applied_p_kw", target_p))
        applied_q = float(s.get("applied_q_kvar", target_q))

        prev_mrid = ""
        if lifecycle_type == "replay":
            prev_mrid = prev_mrid_for_replay(history)

        if lifecycle_type == "burst":
            # Generate 5-10 rapid duplicate commands
            burst_n = int(rng.integers(5, 11))
            for b in range(burst_n):
                burst_rows = build_control_lifecycle(
                    scenario_id=s["scenario_id"] + f"_burst{b}",
                    start_time_s=start_s + b * 3,
                    command_type=s["command_type"],
                    asset_id=asset_id,
                    target_p_kw=target_p,
                    target_q_kvar=target_q,
                    applied_p_kw=applied_p,
                    applied_q_kvar=applied_q,
                    label_anomaly=int(s["label_anomaly"]),
                    label_cyber=int(s["label_cyber_anomaly"]),
                    label_physical=int(s["label_physical_anomaly"]),
                    cia=s["cia_dimension"],
                    lifecycle_type="burst",
                    delay_s=0.0,
                    rng=rng,
                    burst_count=burst_n,
                )
                all_rows.extend(burst_rows)
                history.extend(burst_rows)
        else:
            lifecycle_rows = build_control_lifecycle(
                scenario_id=s["scenario_id"],
                start_time_s=start_s,
                command_type=s["command_type"],
                asset_id=asset_id,
                target_p_kw=target_p,
                target_q_kvar=target_q,
                applied_p_kw=applied_p,
                applied_q_kvar=applied_q,
                label_anomaly=int(s["label_anomaly"]),
                label_cyber=int(s["label_cyber_anomaly"]),
                label_physical=int(s["label_physical_anomaly"]),
                cia=s["cia_dimension"],
                lifecycle_type=lifecycle_type,
                delay_s=delay_s,
                rng=rng,
                prev_mrid=prev_mrid,
            )
            all_rows.extend(lifecycle_rows)
            history.extend(lifecycle_rows)

        # Add a metering reading after each scenario
        meter_s = int(s["end_time_s"])
        meter_row = build_meter_reading(
            PCC_ID, meter_s,
            float(rng.uniform(-20, 80)),
            float(rng.uniform(5, 25)),
            rng
        )
        meter_row["scenario_id"] = s["scenario_id"]
        meter_row["label_anomaly"] = int(s["label_anomaly"])
        meter_row["label_cyber_anomaly"] = int(s["label_cyber_anomaly"])
        meter_row["label_physical_anomaly"] = int(s["label_physical_anomaly"])
        all_rows.append(meter_row)

    df = pd.DataFrame(all_rows)
    df = df.sort_values("time_s").reset_index(drop=True)
    df.to_csv(CYBER_ANOMALOUS_CSV, index=False)

    # Counts
    lifecycle_counts = df.groupby("lifecycle_stage").size().to_dict() if "lifecycle_stage" in df.columns else {}
    scenario_counts = df[df["label_cyber_anomaly"] == 1].groupby("scenario_id").size().to_dict() if "scenario_id" in df.columns else {}
    cia_counts = df.groupby("cia_dimension").size().to_dict() if "cia_dimension" in df.columns else {}
    anomaly_flag_counts = {
        "label_anomaly=1": int((df["label_anomaly"] == 1).sum()),
        "label_cyber=1": int((df["label_cyber_anomaly"] == 1).sum()),
        "label_physical=1": int((df["label_physical_anomaly"] == 1).sum()),
        "blocked": int(df["blocked_flag"].sum()),
        "replay": int(df["replay_flag"].sum()),
        "mismatch": int(df["mismatch_flag"].sum()),
        "duplicate": int(df["duplicate_flag"].sum()),
    }

    lines = [
        "# Cyber Layer Improvement Report",
        "",
        "## Overview",
        "",
        f"- Protocol claim level: `semantic_ieee2030_5_style`",
        f"- Total anomalous events: {len(df)}",
        f"- Scenarios covered: {len(scenarios)}",
        "",
        "## Lifecycle Stage Counts",
        "",
    ]
    for k, v in sorted(lifecycle_counts.items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## CIA Dimension Counts", ""]
    for k, v in sorted(cia_counts.items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Anomaly Flag Counts", ""]
    for k, v in anomaly_flag_counts.items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Protocol Claim",
        "",
        "All events use `protocol_claim_level = semantic_ieee2030_5_style`.",
        "No actual serialized IEEE 2030.5 XML, EXI, or packet captures are present.",
        "This dataset models the IEEE 2030.5 DER control lifecycle at the semantic level.",
        "",
        "## What Is NOT Claimed",
        "- Real utility field telemetry",
        "- Official IEEE 2030.5 compliance",
        "- Packet-level protocol compliance",
        "- Serialized EXI messages",
        "- Real network packet captures",
    ]
    write_report(lines, REPORTS / "03_cyber_layer_improvement_report.md")

    print(f"Anomalous cyber log: {len(df)} events covering {len(scenarios)} scenarios")
    print(f"Saved: {CYBER_ANOMALOUS_CSV}")
    return df


if __name__ == "__main__":
    main()
