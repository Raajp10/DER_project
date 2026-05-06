"""
Generate normal IEEE 2030.5-style semantic cyber event log for 7 days.
Includes: routine DER controls, metering readings, DER status reports.
Writes: data_updated/raw/cyber_event_log_normal_ieee2030_5_semantic_7d.csv
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

from paths import CYBER_NORMAL_CSV, REPORTS, METADATA
from config import (
    DER_SITE_ID, PCC_ID, PV_ASSET_ID, BESS_ASSET_ID, RANDOM_SEED, PROTOCOL_CLAIM_LEVEL,
)
from validation_utils import write_report

_05b_path = ROOT / "scripts_updated" / "04_cyber_log_generation" / "05b_generate_der_command_lifecycle.py"
_spec = importlib.util.spec_from_file_location("generate_der_command_lifecycle", _05b_path)
_lifecycle_mod = importlib.util.module_from_spec(_spec)
sys.modules["generate_der_command_lifecycle"] = _lifecycle_mod
_spec.loader.exec_module(_lifecycle_mod)
build_control_lifecycle = _lifecycle_mod.build_control_lifecycle
build_meter_reading = _lifecycle_mod.build_meter_reading
make_mrid = _lifecycle_mod.make_mrid
ts_str = _lifecycle_mod.ts_str
time_s_to_ts = _lifecycle_mod.time_s_to_ts

TOTAL_S = 604_800
rng = np.random.default_rng(RANDOM_SEED)


def generate_routine_pv_commands() -> list:
    """
    Generate routine PV curtailment commands throughout 7 days.
    Approximately every 2-4 hours during daylight, then clear at night.
    """
    rows = []
    # Generate roughly 84 commands (12 per day)
    for day in range(7):
        for hour in [7, 9, 11, 12, 13, 15, 17, 18]:
            base_s = day * 86400 + hour * 3600
            # Small offset
            start_s = base_s + int(rng.integers(0, 300))
            target_p = float(rng.uniform(40, 100))
            lifecycle = build_control_lifecycle(
                scenario_id=f"normal_d{day}_h{hour}",
                start_time_s=start_s,
                command_type="active_power_limit",
                asset_id=PV_ASSET_ID,
                target_p_kw=target_p,
                target_q_kvar=0.0,
                applied_p_kw=target_p,
                applied_q_kvar=0.0,
                label_anomaly=0,
                label_cyber=0,
                label_physical=0,
                cia="none",
                lifecycle_type="normal",
                delay_s=0.0,
                rng=rng,
            )
            rows.extend(lifecycle)
    return rows


def generate_routine_bess_commands() -> list:
    """Generate routine BESS dispatch commands — charge mid-day, discharge evening."""
    rows = []
    for day in range(7):
        # Charge at noon
        start_s = day * 86400 + 12 * 3600 + int(rng.integers(0, 600))
        lifecycle = build_control_lifecycle(
            scenario_id=f"normal_bess_charge_d{day}",
            start_time_s=start_s,
            command_type="storage_dispatch",
            asset_id=BESS_ASSET_ID,
            target_p_kw=-35.0,
            target_q_kvar=0.0,
            applied_p_kw=-35.0,
            applied_q_kvar=0.0,
            label_anomaly=0, label_cyber=0, label_physical=0,
            cia="none", lifecycle_type="normal", delay_s=0.0, rng=rng,
        )
        rows.extend(lifecycle)
        # Discharge in evening
        start_s2 = day * 86400 + 18 * 3600 + int(rng.integers(0, 600))
        lifecycle2 = build_control_lifecycle(
            scenario_id=f"normal_bess_discharge_d{day}",
            start_time_s=start_s2,
            command_type="storage_dispatch",
            asset_id=BESS_ASSET_ID,
            target_p_kw=35.0,
            target_q_kvar=0.0,
            applied_p_kw=35.0,
            applied_q_kvar=0.0,
            label_anomaly=0, label_cyber=0, label_physical=0,
            cia="none", lifecycle_type="normal", delay_s=0.0, rng=rng,
        )
        rows.extend(lifecycle2)
    return rows


def generate_background_metering() -> list:
    """Generate 15-minute interval PCC metering readings for 7 days."""
    rows = []
    p_values = np.random.default_rng(55).uniform(-30, 80, 672)  # 7*24*4 = 672 readings
    q_values = np.random.default_rng(56).uniform(5, 30, 672)
    for i, (p, q) in enumerate(zip(p_values, q_values)):
        time_s = i * 900  # every 15 minutes
        if time_s >= TOTAL_S:
            break
        row = build_meter_reading(PCC_ID, time_s, float(p), float(q), rng)
        row["scenario_id"] = f"metering_bg_{i}"
        rows.append(row)
    return rows


def generate_der_status_reports() -> list:
    """Generate hourly DER status reports."""
    rows = []
    for hour in range(7 * 24):
        time_s = hour * 3600
        if time_s >= TOTAL_S:
            break
        PCL = PROTOCOL_CLAIM_LEVEL
        t = time_s_to_ts(time_s)
        row = {
            "event_id": make_mrid(),
            "event_time_utc": ts_str(t),
            "time_s": time_s,
            "der_site_id": DER_SITE_ID,
            "pcc_id": PCC_ID,
            "asset_id": PV_ASSET_ID,
            "asset_type": "PVSystem",
            "source_system_id": f"DER-{PV_ASSET_ID}",
            "destination_system_id": "DERMS-001",
            "source_role": "DER_Client",
            "destination_role": "DERMS",
            "command_type": "status_report",
            **{k: ts_str(t) for k in [
                "command_created_time_utc", "command_sent_time_utc",
                "command_recv_time_utc", "command_accept_time_utc",
                "command_apply_time_utc", "command_response_time_utc",
            ]},
            "command_expire_time_utc": ts_str(t + pd.Timedelta(hours=2)),
            "target_p_kw": 0.0, "target_q_kvar": 0.0, "target_pf": 1.0,
            "applied_p_kw": 0.0, "applied_q_kvar": 0.0, "applied_pf": 1.0,
            "authn_status": "passed", "authz_status": "passed",
            "integrity_status": "passed", "availability_status": "normal",
            "confidentiality_status": "passed",
            "delivery_status": "delivered", "communication_outcome": "success",
            "delay_s": 0.0, "network_latency_ms": 20.0,
            "processing_latency_ms": 5.0, "queue_latency_ms": 2.0, "total_delay_s": 0.027,
            "retry_count": 0, "timeout_flag": 0, "duplicate_flag": 0,
            "stale_command_flag": 0, "replay_flag": 0, "blocked_flag": 0, "mismatch_flag": 0,
            "label_anomaly": 0, "label_cyber_anomaly": 0, "label_physical_anomaly": 0,
            "cia_dimension": "none",
            "protocol_profile": "IEEE_2030_5_DER",
            "protocol_claim_level": PCL,
            "message_type": "DER_STATUS_REPORT",
            "cyber_event_category": "status",
            "message_mrid": make_mrid(),
            "transaction_id": make_mrid(),
            "session_id": make_mrid()[:8],
            "previous_message_mrid": "", "related_control_mrid": "", "related_metering_mrid": "",
            "ieee2030_5_profile": "DER",
            "ieee2030_5_resource_type": "DERStatus",
            "ieee2030_5_function_set": "DER",
            "ieee2030_5_der_program_id": f"DERProgram-{DER_SITE_ID}",
            "ieee2030_5_der_control_id": "",
            "ieee2030_5_control_mode": "status_report",
            "ieee2030_5_control_status": "reported",
            "ieee2030_5_response_required": 0,
            "ieee2030_5_response_status": "no_reply",
            "ieee2030_5_der_curve_id": "", "ieee2030_5_client_id": f"DER-{PV_ASSET_ID}",
            "ieee2030_5_server_id": "DERMS-001",
            "ieee2030_5_security_context": "TLS_1.2_client_cert",
            "lifecycle_stage": "DER_STATUS_REPORT",
            "lifecycle_order": 0,
            "mapped_physical_asset": PV_ASSET_ID,
            "related_physical_variable": "pv_actual_p_kw",
            "is_control_event": 0, "is_monitoring_event": 0, "is_status_event": 1, "is_security_event": 0,
            "scenario_id": f"status_h{hour}",
        }
        rows.append(row)
    return rows


def main() -> pd.DataFrame:
    CYBER_NORMAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    all_rows = []
    print("Generating routine PV commands...")
    all_rows.extend(generate_routine_pv_commands())
    print("Generating routine BESS commands...")
    all_rows.extend(generate_routine_bess_commands())
    print("Generating background metering readings...")
    all_rows.extend(generate_background_metering())
    print("Generating DER status reports...")
    all_rows.extend(generate_der_status_reports())

    df = pd.DataFrame(all_rows)
    df = df.sort_values("time_s").reset_index(drop=True)
    df.to_csv(CYBER_NORMAL_CSV, index=False)

    counts = df.groupby("lifecycle_stage").size().to_dict() if "lifecycle_stage" in df.columns else {}
    print(f"Normal cyber log: {len(df)} events")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"Saved: {CYBER_NORMAL_CSV}")
    return df


if __name__ == "__main__":
    main()
