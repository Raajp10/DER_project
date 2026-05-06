"""
Build per-scenario cyber-physical evidence packets (JSONL).
Writes: data_updated/processed/cyber_physical_evidence_packets_7d.jsonl
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

from paths import (
    CONTEXT_WINDOWS_CSV, LIFECYCLE_MAP_CSV, CYBER_ANOMALOUS_CSV,
    EVIDENCE_PACKETS_JSONL,
)
from config import (
    PV_S_RATED_KVA, BESS_S_RATED_KVA, BESS_SOC_MIN_PERCENT, BESS_SOC_MAX_PERCENT,
)


def build_constraint_checks(window_df: pd.DataFrame) -> list:
    checks = []
    if "actual_p_kw" in window_df.columns:
        max_p = float(window_df["actual_p_kw"].abs().max())
        checks.append({
            "constraint": "pv_apparent_power_limit",
            "status": "pass" if max_p <= PV_S_RATED_KVA else "fail",
            "value": round(max_p, 2),
            "limit": PV_S_RATED_KVA,
        })
    if "actual_soc_percent" in window_df.columns:
        min_soc = float(window_df["actual_soc_percent"].min())
        max_soc = float(window_df["actual_soc_percent"].max())
        checks.append({
            "constraint": "bess_soc_min",
            "status": "pass" if min_soc >= BESS_SOC_MIN_PERCENT - 0.5 else "fail",
            "value": round(min_soc, 2),
            "limit": BESS_SOC_MIN_PERCENT,
        })
        checks.append({
            "constraint": "bess_soc_max",
            "status": "pass" if max_soc <= BESS_SOC_MAX_PERCENT + 0.5 else "fail",
            "value": round(max_soc, 2),
            "limit": BESS_SOC_MAX_PERCENT,
        })
    if "residual_voltage_pu" in window_df.columns:
        max_residual = float(window_df["residual_voltage_pu"].abs().max())
        checks.append({
            "constraint": "voltage_residual_threshold",
            "status": "pass" if max_residual < 0.05 else "warning",
            "value": round(max_residual, 4),
            "limit": 0.05,
        })
    return checks


def build_physical_evidence(window_df: pd.DataFrame) -> list:
    evidence = []
    for col, bcol, rcol, meaning in [
        ("actual_p_kw", "baseline_p_kw", "residual_p_kw", "PV active power output"),
        ("actual_q_kvar", "baseline_q_kvar", "residual_q_kvar", "PV reactive power"),
        ("actual_voltage_pu", "baseline_voltage_pu", "residual_voltage_pu", "PCC mean voltage"),
        ("actual_soc_percent", "baseline_soc_percent", "residual_soc_percent", "BESS state of charge"),
    ]:
        if col in window_df.columns:
            evidence.append({
                "field": col,
                "baseline": round(float(window_df[bcol].mean()), 3) if bcol in window_df.columns else None,
                "actual": round(float(window_df[col].mean()), 3),
                "residual": round(float(window_df[rcol].mean()), 3) if rcol in window_df.columns else None,
                "meaning": meaning,
            })
    return evidence


def main() -> int:
    EVIDENCE_PACKETS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    if not CONTEXT_WINDOWS_CSV.exists():
        print("ERROR: Context windows CSV not found. Run 06b first.")
        sys.exit(1)
    if not LIFECYCLE_MAP_CSV.exists():
        print("ERROR: Lifecycle map not found.")
        sys.exit(1)

    print("Loading context windows...")
    ctx = pd.read_csv(CONTEXT_WINDOWS_CSV)
    lifecycle = pd.read_csv(LIFECYCLE_MAP_CSV)

    # Load cyber evidence
    cyber_available = CYBER_ANOMALOUS_CSV.exists()
    if cyber_available:
        cyber = pd.read_csv(CYBER_ANOMALOUS_CSV)
    else:
        cyber = pd.DataFrame()

    count = 0
    with open(EVIDENCE_PACKETS_JSONL, "w") as f:
        for _, sc in lifecycle.iterrows():
            sid = sc["scenario_id"]

            # Window rows for this scenario
            w = ctx[ctx["scenario_id"] == sid]

            # Cyber evidence
            cyber_evidence = []
            if cyber_available and len(cyber) > 0 and "scenario_id" in cyber.columns:
                sc_cyber = cyber[cyber["scenario_id"] == sid]
                if len(sc_cyber) > 0:
                    for field in ["authn_status", "authz_status", "integrity_status",
                                  "blocked_flag", "replay_flag", "mismatch_flag",
                                  "duplicate_flag", "delivery_status", "communication_outcome"]:
                        if field in sc_cyber.columns:
                            val = sc_cyber[field].iloc[0]
                            cyber_evidence.append({
                                "field": field,
                                "value": str(val),
                                "meaning": field.replace("_", " "),
                            })

            # Physical evidence from context window
            phys_evidence = build_physical_evidence(w) if len(w) > 0 else []
            constraints = build_constraint_checks(w) if len(w) > 0 else []

            timing = {
                "command_sent_time_utc": str(sc.get("command_sent_time_utc", "")),
                "command_apply_time_utc": str(sc.get("command_apply_time_utc", "")),
                "physical_effect_start_time_utc": str(sc.get("physical_effect_start_time_utc", "")),
                "delay_s": float(0),
            }

            # Determine limitations
            limitations = []
            gen_method = str(sc.get("generation_method", "physics_constrained_surrogate"))
            if "surrogate" in gen_method:
                limitations.append(
                    "Physical data generated by physics-constrained surrogate, "
                    "not actual OpenDSS event-window simulation."
                )
            claim_level = str(sc.get("protocol_claim_level", "semantic_ieee2030_5_style"))
            if claim_level == "semantic_ieee2030_5_style":
                limitations.append(
                    "Cyber log is semantic IEEE 2030.5-style model only. "
                    "No actual protocol serialization or packet capture."
                )

            packet = {
                "scenario_id": sid,
                "event_id": sid,
                "asset_id": str(sc.get("asset_id", "")),
                "anomaly_type": str(sc.get("anomaly_type", "")),
                "final_event_class": str(sc.get("scenario_class", "")),
                "generation_method": gen_method,
                "protocol_claim_level": claim_level,
                "cyber_evidence": cyber_evidence,
                "physical_evidence": phys_evidence,
                "constraints_checked": constraints,
                "timing_evidence": timing,
                "claim_support": str(sc.get("scenario_class", "normal")),
                "limitations": limitations,
            }
            f.write(json.dumps(packet) + "\n")
            count += 1

    print(f"Evidence packets: {count} scenarios written to {EVIDENCE_PACKETS_JSONL}")
    return count


if __name__ == "__main__":
    main()
