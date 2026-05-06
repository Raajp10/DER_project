"""
Validate the IEEE 2030.5-style semantic cyber event layer.
Writes:
  data_updated/validation/ieee2030_5_semantic_validation_report.md
  data_updated/validation/ieee2030_5_semantic_validation_summary.json
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
    CYBER_NORMAL_CSV, CYBER_ANOMALOUS_CSV, IEEE2030_5_XML_DIR,
    CYBER_VALIDATION_REPORT, CYBER_VALIDATION_JSON, VALIDATION,
)
from validation_utils import (
    aggregate_results, save_json, write_report,
)

REQUIRED_COLS = [
    "event_id", "event_time_utc", "time_s", "der_site_id", "pcc_id",
    "asset_id", "asset_type", "lifecycle_stage", "lifecycle_order",
    "protocol_claim_level", "label_anomaly", "label_cyber_anomaly",
    "label_physical_anomaly", "cia_dimension",
    "authn_status", "authz_status", "integrity_status",
    "blocked_flag", "replay_flag", "mismatch_flag",
    "delivery_status", "communication_outcome",
    "ieee2030_5_resource_type", "ieee2030_5_function_set",
]

VALID_STAGES = {
    "DER_CONTROL_CREATED", "DER_CONTROL_SENT", "DER_CONTROL_RECEIVED",
    "DER_CONTROL_ACCEPTED", "DER_CONTROL_REJECTED", "DER_CONTROL_APPLIED",
    "DER_CONTROL_RESPONSE", "DER_STATUS_REPORT", "DER_METER_READING",
    "SECURITY_AUTH_FAILURE", "SECURITY_REPLAY_DETECTED", "SECURITY_TIMEOUT",
    "SECURITY_BLOCKED_COMMAND", "SECURITY_INTEGRITY_MISMATCH",
}

VALID_CLAIM_LEVELS = {
    "semantic_ieee2030_5_style", "serialized_ieee2030_5_xml",
    "serialized_ieee2030_5_exi", "packet_capture",
}


def validate_df(df: pd.DataFrame, name: str) -> list:
    results = []

    # Required columns
    for col in REQUIRED_COLS:
        ok = col in df.columns
        results.append({"check": f"{name}_{col}_exists",
                       "status": "PASS" if ok else "FAIL"})

    if len(df) == 0:
        results.append({"check": f"{name}_not_empty", "status": "FAIL"})
        return results

    results.append({"check": f"{name}_not_empty", "status": "PASS",
                   "row_count": len(df)})

    # Protocol claim level validity
    if "protocol_claim_level" in df.columns:
        invalid = df[~df["protocol_claim_level"].isin(VALID_CLAIM_LEVELS)]
        results.append({"check": f"{name}_protocol_claim_level_valid",
                       "status": "PASS" if len(invalid) == 0 else "FAIL",
                       "invalid_rows": int(len(invalid))})

        # No false packet-level claim without artifacts
        packet_claim = (df["protocol_claim_level"] == "packet_capture")
        if packet_claim.any():
            results.append({"check": f"{name}_no_false_packet_claim",
                           "status": "FAIL",
                           "message": "packet_capture claim found but no actual packet captures exist"})
        else:
            results.append({"check": f"{name}_no_false_packet_claim", "status": "PASS"})

        # If serialized_ieee2030_5_xml claimed, check XML artifacts exist
        xml_claim = (df["protocol_claim_level"] == "serialized_ieee2030_5_xml")
        if xml_claim.any():
            xml_ok = IEEE2030_5_XML_DIR.exists() and len(list(IEEE2030_5_XML_DIR.glob("*.xml"))) > 0
            results.append({"check": f"{name}_xml_artifacts_exist_for_xml_claim",
                           "status": "PASS" if xml_ok else "FAIL"})

    # Lifecycle stage validity
    if "lifecycle_stage" in df.columns:
        invalid_stages = df[~df["lifecycle_stage"].isin(VALID_STAGES)]
        results.append({"check": f"{name}_lifecycle_stages_valid",
                       "status": "PASS" if len(invalid_stages) == 0 else "FAIL",
                       "invalid": int(len(invalid_stages))})

    # CIA labels exist
    if "cia_dimension" in df.columns:
        missing_cia = df["cia_dimension"].isna().sum()
        results.append({"check": f"{name}_cia_dimension_no_null",
                       "status": "PASS" if missing_cia == 0 else "FAIL",
                       "null_count": int(missing_cia)})

    # Blocked logic: blocked events must have authz_status=failed
    if all(c in df.columns for c in ["blocked_flag", "authz_status"]):
        blocked_rows = df[df["blocked_flag"] == 1]
        if len(blocked_rows) > 0:
            authz_ok = (blocked_rows["authz_status"] == "failed").all()
            results.append({"check": f"{name}_blocked_flag_authz_failed",
                           "status": "PASS" if authz_ok else "FAIL",
                           "blocked_count": int(len(blocked_rows))})

    # Replay logic: replay events must have previous_message_mrid non-empty
    if all(c in df.columns for c in ["replay_flag", "previous_message_mrid"]):
        replay_rows = df[df["replay_flag"] == 1]
        if len(replay_rows) > 0:
            has_prev = replay_rows["previous_message_mrid"].fillna("").astype(str).str.len() > 0
            ok = has_prev.all()
            results.append({"check": f"{name}_replay_has_previous_mrid",
                           "status": "PASS" if ok else "WARN",
                           "replay_without_prev": int((~has_prev).sum())})

    # Mismatch logic: mismatch flag → integrity_status=compromised
    if all(c in df.columns for c in ["mismatch_flag", "integrity_status"]):
        mismatch_rows = df[df["mismatch_flag"] == 1]
        if len(mismatch_rows) > 0:
            integ_ok = (mismatch_rows["integrity_status"] == "compromised").all()
            results.append({"check": f"{name}_mismatch_flag_integrity_compromised",
                           "status": "PASS" if integ_ok else "FAIL",
                           "mismatch_count": int(len(mismatch_rows))})

    # Delay logic: delayed events must have higher total_delay_s
    if all(c in df.columns for c in ["availability_status", "total_delay_s"]):
        degraded = df[df["availability_status"] == "degraded"]
        if len(degraded) > 0:
            normal = df[df["availability_status"] == "normal"]
            if len(normal) > 0:
                avg_degraded = degraded["total_delay_s"].mean()
                avg_normal = normal["total_delay_s"].mean()
                ok = avg_degraded > avg_normal
                results.append({"check": f"{name}_degraded_has_higher_delay",
                               "status": "PASS" if ok else "WARN",
                               "avg_degraded_s": round(avg_degraded, 3),
                               "avg_normal_s": round(avg_normal, 3)})

    return results


def main() -> dict:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    all_results = []

    for csv_path, label in [(CYBER_NORMAL_CSV, "normal"), (CYBER_ANOMALOUS_CSV, "anomalous")]:
        if not csv_path.exists():
            all_results.append({"check": f"{label}_log_exists",
                               "status": "FAIL", "error": f"{csv_path.name} not found"})
        else:
            df = pd.read_csv(csv_path)
            all_results.append({"check": f"{label}_log_exists",
                               "status": "PASS", "rows": len(df)})
            all_results.extend(validate_df(df, label))

    summary = aggregate_results(all_results)
    summary["results"] = all_results
    save_json(summary, CYBER_VALIDATION_JSON)

    lines = [
        "# IEEE 2030.5 Semantic Layer Validation Report",
        "",
        f"**Overall:** `{summary['overall']}`",
        f"**Passed:** {summary['passed']} / {summary['total']}",
        f"**Failed:** {summary['failed']}",
        "",
        "## Important Claim Boundary",
        "",
        "- protocol_claim_level = `semantic_ieee2030_5_style`",
        "- No official IEEE 2030.5 compliance is claimed.",
        "- No EXI encoding or packet capture is present.",
        "",
        "## Check Details",
        "",
    ]
    for r in all_results:
        icon = "✓" if r.get("status") == "PASS" else ("⚠" if r.get("status") == "WARN" else "✗")
        lines.append(f"- {icon} `{r.get('check')}`: **{r.get('status')}**")
    write_report(lines, CYBER_VALIDATION_REPORT)

    print(f"Cyber validation: {summary['overall']}")
    return summary


if __name__ == "__main__":
    main()
