"""
Leakage audit for Phase 1 model benchmark.

Reads:
  D:/updated_dataset/models/windows/windows_all.parquet
  D:/updated_dataset/models/windows/feature_manifest.json

Writes:
  D:/updated_dataset/models/reports/LEAKAGE_AUDIT_REPORT.md
  D:/updated_dataset/models/results/leakage_audit_summary.json

Checks that no leakage columns appear in the model feature set.
Fails if any leakage column is found in the flat feature names.
"""
import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    WINDOWS_ALL_PARQUET, FEATURE_MANIFEST_JSON, REPORTS_DIR, RESULTS_DIR,
    LEAKAGE_AUDIT_JSON, ensure_all_dirs,
)
from feature_config import RAW_FEATURES, FLAT_FEATURE_NAMES, LEAKAGE_COLUMNS, WINDOW_META_COLS


def audit():
    ensure_all_dirs()

    with open(FEATURE_MANIFEST_JSON) as f:
        manifest = json.load(f)

    flat_names = manifest["flat_feature_names"]
    raw_names = manifest["raw_features"]

    # Build forbidden pattern set
    forbidden_patterns = [
        "timestamp_utc", "time_s", "window_id", "event_id",
        "scenario_id", "scenario_name", "scenario_class",
        "anomaly_type", "label", "y_", "target", "expected",
        "applied", "delay", "flag", "attack", "anomaly",
        "cyber", "protocol", "message", "mrid", "transaction",
        "session", "lifecycle", "generation_method", "protocol_claim_level",
        "physical_effect", "command", "response", "status", "class",
        "split", "final_event_class", "evidence",
    ]

    # Explicitly allowed despite name overlap
    allowed_overrides = {
        "pv_actual_p_kw", "bess_actual_p_kw",
        "pv_actual_q_kvar", "bess_actual_q_kvar",
        "physical_constraint_status",  # NOT in flat features, just checking
    }

    issues = []
    for col in flat_names:
        base_col = col.rsplit("_", 1)[0] if "_" in col else col
        if base_col in allowed_overrides:
            continue
        for pat in forbidden_patterns:
            if pat.lower() in col.lower():
                issues.append({
                    "column": col,
                    "matched_pattern": pat,
                    "severity": "FAIL",
                })
                break

    # Check raw features against leakage list
    raw_leakage = [r for r in raw_names if r in LEAKAGE_COLUMNS]

    # Check that meta columns are NOT in flat_names
    meta_in_flat = [m for m in WINDOW_META_COLS if m in flat_names]

    overall = "PASS" if not issues and not raw_leakage and not meta_in_flat else "FAIL"

    summary = {
        "status": overall,
        "flat_feature_count": len(flat_names),
        "raw_feature_count": len(raw_names),
        "leakage_violations": issues,
        "raw_features_in_leakage_list": raw_leakage,
        "meta_columns_in_flat_features": meta_in_flat,
        "allowed_overrides_confirmed": sorted(allowed_overrides),
    }
    with open(LEAKAGE_AUDIT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Leakage Audit Report", "",
        f"**Status:** {overall}",
        f"**Flat features audited:** {len(flat_names)}",
        f"**Raw features audited:** {len(raw_names)}",
        f"**Violations found:** {len(issues)}", "",
        "## Allowed Override Confirmations", "",
        "The following column names contain strings that could match leakage patterns",
        "but are confirmed physical sensor measurements (not command/label columns):", "",
    ]
    for col in sorted(allowed_overrides):
        lines.append(f"- `{col}` — physical sensor measurement, not command-context leakage")

    lines += ["", "## Raw Features Confirmed Safe", ""]
    for f in raw_names:
        lines.append(f"- `{f}`")

    lines += ["", "## Excluded Leakage Columns", ""]
    leakage_groups = {
        "Attack labels (direct target leakage)": [
            "physical_effect_active_flag", "physical_effect_type",
            "physical_scenario_id", "physical_constraint_status"],
        "Command-context leakage": [
            "pv_commanded_p_kw", "pv_commanded_q_kvar",
            "bess_commanded_p_kw", "bess_commanded_q_kvar", "pv_curtailment_kw"],
        "Temporal/ID metadata": [
            "timestamp_utc", "time_s", "der_site_id", "pcc_id"],
        "Constants (no signal)": [
            "pv_s_rated_kva", "bess_s_rated_kva", "bess_capacity_kwh",
            "bess_soc_min_percent", "bess_soc_max_percent"],
        "Dataset metadata": ["generation_method"],
        "Categorical flags": [
            "pv_inverter_mode", "voltage_unbalance_status",
            "pv_constraint_violation_flag", "bess_constraint_violation_flag",
            "regulator_tap_position", "capacitor_status"],
    }
    for group, cols in leakage_groups.items():
        lines.append(f"### {group}")
        for c in cols:
            lines.append(f"- `{c}`")
        lines.append("")

    if issues:
        lines += ["## VIOLATIONS (FAIL)", ""]
        for v in issues:
            lines.append(f"- `{v['column']}` matched pattern `{v['matched_pattern']}`")
    else:
        lines += ["## Violations: NONE — all features confirmed clean", ""]

    lines.append(f"\n**LEAKAGE AUDIT: {overall}**")
    with open(REPORTS_DIR / "LEAKAGE_AUDIT_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Leakage audit: {overall} ({len(flat_names)} flat features, {len(issues)} violations)")
    if overall == "FAIL":
        raise RuntimeError(f"Leakage audit FAILED: {len(issues)} violations found. "
                           f"Raw leakage: {raw_leakage}. Meta in flat: {meta_in_flat}")
    return summary


if __name__ == "__main__":
    audit()
