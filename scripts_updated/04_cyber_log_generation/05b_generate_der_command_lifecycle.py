"""
Utilities for generating IEEE 2030.5-style DER command lifecycle events.
Used by both normal and anomalous log generators.
"""
import sys
import uuid
import math
import random
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from config import (
    DER_SITE_ID, PCC_ID, PV_ASSET_ID, BESS_ASSET_ID,
    NORMAL_LATENCY_MS_MEAN, NORMAL_LATENCY_MS_STD,
    NORMAL_PROCESSING_MS_MEAN, NORMAL_PROCESSING_MS_STD,
    DELAYED_LATENCY_MS_MEAN, PROTOCOL_CLAIM_LEVEL,
)

START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")


def make_mrid() -> str:
    return str(uuid.uuid4())


def ts_str(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def time_s_to_ts(time_s: int) -> pd.Timestamp:
    return START_UTC + pd.Timedelta(seconds=int(time_s))


def resource_type_and_mode(command_type: str, asset_id: str) -> tuple:
    if "active_power" in command_type:
        return "DERControl", "active_power_limit"
    if "reactive_power" in command_type:
        return "DERControl", "reactive_power_control"
    if "storage_dispatch" in command_type:
        return "DERControl", "storage_dispatch"
    if "storage_reactive" in command_type:
        return "DERControl", "storage_reactive_support"
    if "pcc" in asset_id.lower() or "meter" in command_type:
        return "MirrorMeterReading", "metering"
    return "DERControl", "active_power_limit"


def normal_latencies(rng: np.random.Generator) -> dict:
    net = float(rng.normal(NORMAL_LATENCY_MS_MEAN, NORMAL_LATENCY_MS_STD))
    proc = float(rng.normal(NORMAL_PROCESSING_MS_MEAN, NORMAL_PROCESSING_MS_STD))
    queue = float(rng.normal(5.0, 2.0))
    return {
        "network_latency_ms": max(net, 5.0),
        "processing_latency_ms": max(proc, 5.0),
        "queue_latency_ms": max(queue, 0.5),
        "total_delay_s": (net + proc + queue) / 1000.0,
    }


def delayed_latencies(rng: np.random.Generator, extra_delay_s: float = 10.0) -> dict:
    net = float(rng.normal(DELAYED_LATENCY_MS_MEAN, 2000.0))
    proc = float(rng.normal(NORMAL_PROCESSING_MS_MEAN * 2, 20.0))
    queue = float(rng.normal(500.0, 100.0))
    return {
        "network_latency_ms": max(net, 100.0),
        "processing_latency_ms": max(proc, 10.0),
        "queue_latency_ms": max(queue, 50.0),
        "total_delay_s": extra_delay_s + (net + proc + queue) / 1000.0,
    }


def build_control_lifecycle(
    scenario_id: str,
    start_time_s: int,
    command_type: str,
    asset_id: str,
    target_p_kw: float,
    target_q_kvar: float,
    applied_p_kw: float,
    applied_q_kvar: float,
    label_anomaly: int,
    label_cyber: int,
    label_physical: int,
    cia: str,
    lifecycle_type: str,  # 'normal' | 'delayed' | 'blocked' | 'wrong_setpoint' | 'replay' | 'burst' | 'soc_violation'
    delay_s: float,
    rng: np.random.Generator,
    prev_mrid: str = "",
    burst_count: int = 1,
) -> list:
    """
    Build a list of cyber event row dicts for one command lifecycle.
    lifecycle_type controls which anomaly flags are set.
    """
    rows = []
    t0 = time_s_to_ts(start_time_s)
    mrid = make_mrid()
    tx_id = make_mrid()
    session_id = make_mrid()[:8]
    prog_id = f"DERProgram-{DER_SITE_ID}"
    ctrl_id = f"DERControl-{mrid[:8]}"
    client_id = "DERMS-001"
    server_id = f"DER-{asset_id}"

    res_type, ctrl_mode = resource_type_and_mode(command_type, asset_id)
    asset_type_str = ("PVSystem" if "pv" in asset_id else
                      "BESS" if "bess" in asset_id else "PCC")

    lat = (delayed_latencies(rng, delay_s)
           if lifecycle_type == "delayed"
           else normal_latencies(rng))

    # Timing offsets
    t_created = t0
    t_sent = t_created + pd.Timedelta(milliseconds=lat["queue_latency_ms"])
    t_recv = t_sent + pd.Timedelta(milliseconds=lat["network_latency_ms"])
    t_accept = t_recv + pd.Timedelta(milliseconds=lat["processing_latency_ms"] * 0.5)
    t_apply = t_recv + pd.Timedelta(milliseconds=lat["processing_latency_ms"])
    if lifecycle_type == "delayed":
        t_apply = t0 + pd.Timedelta(seconds=delay_s)
    t_response = t_apply + pd.Timedelta(milliseconds=50.0)
    t_expire = t0 + pd.Timedelta(seconds=3600)

    # CIA and anomaly flags
    is_blocked = lifecycle_type == "blocked"
    is_replay = lifecycle_type == "replay"
    is_mismatch = lifecycle_type in ("wrong_setpoint",)
    is_delayed = lifecycle_type == "delayed"
    is_burst = lifecycle_type == "burst"
    is_soc = lifecycle_type == "soc_violation"

    authn_status = "passed"
    authz_status = "failed" if is_blocked else "passed"
    integrity_status = "compromised" if is_mismatch else "passed"
    availability_status = ("degraded" if (is_delayed or is_burst) else "normal")
    confidentiality_status = "passed"
    delivery_status = ("blocked" if is_blocked else
                       "delayed" if is_delayed else
                       "delivered")
    comm_outcome = ("blocked" if is_blocked else
                    "replay_detected" if is_replay else
                    "integrity_mismatch" if is_mismatch else
                    "success")

    # Lifecycle stages
    if is_blocked:
        stages = [
            ("DER_CONTROL_CREATED", 0),
            ("DER_CONTROL_SENT", 1),
            ("SECURITY_AUTH_FAILURE", 2),
            ("SECURITY_BLOCKED_COMMAND", 3),
        ]
    elif is_replay:
        stages = [
            ("DER_CONTROL_CREATED", 0),
            ("DER_CONTROL_SENT", 1),
            ("SECURITY_REPLAY_DETECTED", 2),
            ("DER_CONTROL_REJECTED", 3),
        ]
    elif is_mismatch:
        stages = [
            ("DER_CONTROL_CREATED", 0),
            ("DER_CONTROL_SENT", 1),
            ("DER_CONTROL_RECEIVED", 2),
            ("SECURITY_INTEGRITY_MISMATCH", 3),
            ("DER_CONTROL_APPLIED", 4),
            ("DER_CONTROL_RESPONSE", 5),
        ]
    elif is_soc:
        stages = [
            ("DER_CONTROL_CREATED", 0),
            ("DER_CONTROL_SENT", 1),
            ("DER_CONTROL_RECEIVED", 2),
            ("DER_CONTROL_ACCEPTED", 3),
            ("DER_CONTROL_APPLIED", 4),
            ("DER_CONTROL_RESPONSE", 5),
        ]
    else:
        stages = [
            ("DER_CONTROL_CREATED", 0),
            ("DER_CONTROL_SENT", 1),
            ("DER_CONTROL_RECEIVED", 2),
            ("DER_CONTROL_ACCEPTED", 3),
            ("DER_CONTROL_APPLIED", 4),
            ("DER_CONTROL_RESPONSE", 5),
        ]

    stage_times = [t_created, t_sent, t_recv, t_accept, t_apply, t_response]

    for i, (stage, order) in enumerate(stages):
        t_event = stage_times[min(i, len(stage_times) - 1)]
        t_s = int((t_event - START_UTC).total_seconds())

        is_security = stage.startswith("SECURITY_")
        is_control = stage.startswith("DER_CONTROL")
        is_status = stage == "DER_STATUS_REPORT"
        is_meter = stage == "DER_METER_READING"

        row = {
            "event_id": make_mrid(),
            "event_time_utc": ts_str(t_event),
            "time_s": max(0, t_s),
            "der_site_id": DER_SITE_ID,
            "pcc_id": PCC_ID,
            "asset_id": asset_id,
            "asset_type": asset_type_str,
            "source_system_id": client_id,
            "destination_system_id": server_id,
            "source_role": "DERMS",
            "destination_role": "DER_Client",
            "command_type": command_type,
            "command_created_time_utc": ts_str(t_created),
            "command_sent_time_utc": ts_str(t_sent),
            "command_recv_time_utc": ts_str(t_recv),
            "command_accept_time_utc": ts_str(t_accept),
            "command_apply_time_utc": ts_str(t_apply),
            "command_response_time_utc": ts_str(t_response),
            "command_expire_time_utc": ts_str(t_expire),
            "target_p_kw": target_p_kw,
            "target_q_kvar": target_q_kvar,
            "target_pf": 1.0,
            "applied_p_kw": applied_p_kw,
            "applied_q_kvar": applied_q_kvar,
            "applied_pf": 1.0,
            "authn_status": authn_status,
            "authz_status": authz_status,
            "integrity_status": integrity_status,
            "availability_status": availability_status,
            "confidentiality_status": confidentiality_status,
            "delivery_status": delivery_status,
            "communication_outcome": comm_outcome,
            "delay_s": delay_s,
            "network_latency_ms": round(lat["network_latency_ms"], 2),
            "processing_latency_ms": round(lat["processing_latency_ms"], 2),
            "queue_latency_ms": round(lat["queue_latency_ms"], 2),
            "total_delay_s": round(lat["total_delay_s"], 4),
            "retry_count": int(rng.integers(0, 2)) if is_delayed else 0,
            "timeout_flag": 1 if lifecycle_type == "timeout" else 0,
            "duplicate_flag": 1 if is_replay or is_burst else 0,
            "stale_command_flag": 1 if is_replay else 0,
            "replay_flag": 1 if is_replay else 0,
            "blocked_flag": 1 if is_blocked else 0,
            "mismatch_flag": 1 if is_mismatch else 0,
            "label_anomaly": label_anomaly,
            "label_cyber_anomaly": label_cyber,
            "label_physical_anomaly": label_physical,
            "cia_dimension": cia,
            "protocol_profile": "IEEE_2030_5_DER",
            "protocol_claim_level": PROTOCOL_CLAIM_LEVEL,
            "message_type": stage,
            "cyber_event_category": (
                "security" if is_security else
                "control" if is_control else
                "status" if is_status else "metering"
            ),
            "message_mrid": mrid,
            "transaction_id": tx_id,
            "session_id": session_id,
            "previous_message_mrid": prev_mrid,
            "related_control_mrid": ctrl_id,
            "related_metering_mrid": "",
            "ieee2030_5_profile": "DER",
            "ieee2030_5_resource_type": res_type,
            "ieee2030_5_function_set": "DER",
            "ieee2030_5_der_program_id": prog_id,
            "ieee2030_5_der_control_id": ctrl_id,
            "ieee2030_5_control_mode": ctrl_mode,
            "ieee2030_5_control_status": (
                "rejected" if is_blocked or is_replay else "applied"
            ),
            "ieee2030_5_response_required": 1,
            "ieee2030_5_response_status": (
                "event_received" if stage == "DER_CONTROL_RECEIVED" else
                "event_started" if stage == "DER_CONTROL_APPLIED" else
                "event_completed" if stage == "DER_CONTROL_RESPONSE" else
                "no_reply"
            ),
            "ieee2030_5_der_curve_id": "",
            "ieee2030_5_client_id": client_id,
            "ieee2030_5_server_id": server_id,
            "ieee2030_5_security_context": (
                "TLS_1.2_client_cert" if not is_blocked else
                "TLS_1.2_cert_mismatch"
            ),
            "lifecycle_stage": stage,
            "lifecycle_order": order,
            "mapped_physical_asset": asset_id,
            "related_physical_variable": (
                "pv_actual_p_kw" if "pv" in asset_id else
                "bess_actual_p_kw" if "bess" in asset_id else "pcc_p_kw"
            ),
            "is_control_event": 1 if is_control else 0,
            "is_monitoring_event": 1 if is_meter else 0,
            "is_status_event": 1 if is_status else 0,
            "is_security_event": 1 if is_security else 0,
            "scenario_id": scenario_id,
        }
        rows.append(row)

    return rows


def build_meter_reading(asset_id: str, time_s: int, p_kw: float, q_kvar: float,
                         rng: np.random.Generator) -> dict:
    """Build a single MirrorMeterReading event row."""
    t_event = time_s_to_ts(time_s)
    mrid = make_mrid()
    return {
        "event_id": make_mrid(),
        "event_time_utc": ts_str(t_event),
        "time_s": time_s,
        "der_site_id": DER_SITE_ID,
        "pcc_id": PCC_ID,
        "asset_id": asset_id,
        "asset_type": "PCC",
        "source_system_id": f"DER-{asset_id}",
        "destination_system_id": "DERMS-001",
        "source_role": "DER_Client",
        "destination_role": "DERMS",
        "command_type": "meter_reading",
        "command_created_time_utc": ts_str(t_event),
        "command_sent_time_utc": ts_str(t_event),
        "command_recv_time_utc": ts_str(t_event + pd.Timedelta(milliseconds=60)),
        "command_accept_time_utc": ts_str(t_event + pd.Timedelta(milliseconds=70)),
        "command_apply_time_utc": ts_str(t_event + pd.Timedelta(milliseconds=80)),
        "command_response_time_utc": ts_str(t_event + pd.Timedelta(milliseconds=90)),
        "command_expire_time_utc": ts_str(t_event + pd.Timedelta(seconds=300)),
        "target_p_kw": p_kw, "target_q_kvar": q_kvar, "target_pf": 1.0,
        "applied_p_kw": p_kw, "applied_q_kvar": q_kvar, "applied_pf": 1.0,
        "authn_status": "passed", "authz_status": "passed",
        "integrity_status": "passed", "availability_status": "normal",
        "confidentiality_status": "passed",
        "delivery_status": "delivered", "communication_outcome": "success",
        "delay_s": 0.0, "network_latency_ms": 30.0, "processing_latency_ms": 10.0,
        "queue_latency_ms": 5.0, "total_delay_s": 0.045,
        "retry_count": 0, "timeout_flag": 0, "duplicate_flag": 0,
        "stale_command_flag": 0, "replay_flag": 0, "blocked_flag": 0, "mismatch_flag": 0,
        "label_anomaly": 0, "label_cyber_anomaly": 0, "label_physical_anomaly": 0,
        "cia_dimension": "none",
        "protocol_profile": "IEEE_2030_5_DER",
        "protocol_claim_level": PROTOCOL_CLAIM_LEVEL,
        "message_type": "DER_METER_READING",
        "cyber_event_category": "metering",
        "message_mrid": mrid,
        "transaction_id": make_mrid(),
        "session_id": mrid[:8],
        "previous_message_mrid": "", "related_control_mrid": "", "related_metering_mrid": mrid,
        "ieee2030_5_profile": "Metering",
        "ieee2030_5_resource_type": "MirrorMeterReading",
        "ieee2030_5_function_set": "Metering",
        "ieee2030_5_der_program_id": "", "ieee2030_5_der_control_id": "",
        "ieee2030_5_control_mode": "metering",
        "ieee2030_5_control_status": "reported",
        "ieee2030_5_response_required": 0,
        "ieee2030_5_response_status": "no_reply",
        "ieee2030_5_der_curve_id": "",
        "ieee2030_5_client_id": f"DER-{asset_id}",
        "ieee2030_5_server_id": "DERMS-001",
        "ieee2030_5_security_context": "TLS_1.2_client_cert",
        "lifecycle_stage": "DER_METER_READING",
        "lifecycle_order": 0,
        "mapped_physical_asset": asset_id,
        "related_physical_variable": "pcc_p_kw",
        "is_control_event": 0, "is_monitoring_event": 1, "is_status_event": 0, "is_security_event": 0,
        "scenario_id": "metering_background",
    }
