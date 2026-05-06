"""
Validate physical constraints on clean and attacked datasets.
Writes: data_updated/validation/physical_constraints_report.md
        data_updated/validation/physical_constraints_summary.json
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
    CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, DER_METADATA_JSON,
    PHYSICAL_CONSTRAINTS_REPORT, PHYSICAL_CONSTRAINTS_JSON, VALIDATION, METADATA,
)
from config import (
    VOLTAGE_UNBALANCE_WARN_PU, VOLTAGE_UNBALANCE_HARD_PU, VOLTAGE_UNBALANCE_FAIL_PU,
    VOLTAGE_MIN_PLAUSIBLE_PU, VOLTAGE_MAX_PLAUSIBLE_PU,
)
from validation_utils import (
    check_row_count, check_no_duplicates, check_no_gaps,
    check_column_exists, aggregate_results, save_json, write_report,
)

ALLOWED_GENERATION_METHODS = {
    "opendss_clean_baseline",
    "opendss_event_window_resolved",
    "physics_constrained_surrogate",
    "csv_rule_legacy",
}

TOTAL_ROWS = 604_800
TOLERANCE_KVA = 0.5
START_TS = "2026-01-01T00:00:00Z"
END_TS = "2026-01-07T23:59:59Z"

REQUIRED_COLS = [
    "timestamp_utc", "time_s", "der_site_id", "pcc_id",
    "pv_actual_p_kw", "pv_available_kw", "pv_actual_q_kvar",
    "bess_actual_p_kw", "bess_actual_q_kvar", "bess_soc_percent",
    "pcc_v_a_pu", "pcc_v_b_pu", "pcc_v_c_pu",
    "pcc_voltage_mean_pu", "pcc_voltage_unbalance_pu",
    "pcc_p_kw", "pcc_q_kvar",
    "generation_method",
]


def load_metadata() -> dict:
    if DER_METADATA_JSON.exists():
        with open(DER_METADATA_JSON) as f:
            return json.load(f)
    return {"pv_p_rated_kw": 100.0, "pv_s_rated_kva": 111.11,
            "bess_p_rated_kw": 50.0, "bess_s_rated_kva": 55.56,
            "bess_capacity_kwh": 200.0, "bess_soc_min_percent": 10.0,
            "bess_soc_max_percent": 90.0}


def validate_dataset(df: pd.DataFrame, name: str, meta: dict) -> list:
    results = []

    # Row count
    results.append(check_row_count(df, TOTAL_ROWS, name))

    # Duplicate timestamps
    if "time_s" in df.columns:
        results.append(check_no_duplicates(df, "time_s", name))

    # Time gaps
    if "time_s" in df.columns:
        results.append(check_no_gaps(df["time_s"].values, name))

    # Timestamp bounds
    if "timestamp_utc" in df.columns:
        first_ts = str(df["timestamp_utc"].iloc[0])
        last_ts = str(df["timestamp_utc"].iloc[-1])
        ok_start = first_ts.startswith("2026-01-01T00:00:00")
        ok_end = last_ts.startswith("2026-01-07T23:59:59")
        results.append({"check": f"{name}_start_timestamp",
                        "status": "PASS" if ok_start else "FAIL",
                        "expected": START_TS, "actual": first_ts})
        results.append({"check": f"{name}_end_timestamp",
                        "status": "PASS" if ok_end else "FAIL",
                        "expected": END_TS, "actual": last_ts})

    # Required columns
    for col in REQUIRED_COLS:
        results.append(check_column_exists(df, col, name))

    # PV: actual <= available + tolerance
    if "pv_actual_p_kw" in df.columns and "pv_available_kw" in df.columns:
        viol = (df["pv_actual_p_kw"] > df["pv_available_kw"] + 1.0).sum()
        results.append({"check": f"{name}_pv_not_exceed_available",
                        "status": "PASS" if viol == 0 else "FAIL",
                        "violations": int(viol)})

    # PV apparent power limit
    if "pv_actual_p_kw" in df.columns and "pv_actual_q_kvar" in df.columns:
        pv_s_kva = meta["pv_s_rated_kva"]
        pv_s = np.sqrt(df["pv_actual_p_kw"] ** 2 + df["pv_actual_q_kvar"] ** 2)
        viol = (pv_s > pv_s_kva + TOLERANCE_KVA).sum()
        results.append({"check": f"{name}_pv_apparent_power_limit",
                        "status": "PASS" if viol == 0 else "FAIL",
                        "violations": int(viol), "limit_kva": pv_s_kva})

    # BESS apparent power limit
    if "bess_actual_p_kw" in df.columns and "bess_actual_q_kvar" in df.columns:
        bess_s_kva = meta["bess_s_rated_kva"]
        bess_s = np.sqrt(df["bess_actual_p_kw"] ** 2 + df["bess_actual_q_kvar"] ** 2)
        viol = (bess_s > bess_s_kva + TOLERANCE_KVA).sum()
        results.append({"check": f"{name}_bess_apparent_power_limit",
                        "status": "PASS" if viol == 0 else "FAIL",
                        "violations": int(viol), "limit_kva": bess_s_kva})

    # BESS SOC min/max for clean dataset (normal rows only)
    if "bess_soc_percent" in df.columns:
        soc_min = meta["bess_soc_min_percent"]
        soc_max = meta["bess_soc_max_percent"]
        viol_min = (df["bess_soc_percent"] < soc_min - 0.5).sum()
        viol_max = (df["bess_soc_percent"] > soc_max + 0.5).sum()
        # For clean, no violations allowed
        if name == "clean":
            results.append({"check": f"{name}_bess_soc_min",
                            "status": "PASS" if viol_min == 0 else "FAIL",
                            "violations": int(viol_min), "limit": soc_min})
            results.append({"check": f"{name}_bess_soc_max",
                            "status": "PASS" if viol_max == 0 else "FAIL",
                            "violations": int(viol_max), "limit": soc_max})
        else:
            # For attacked, only report (SOC violations may be intentional in scenarios)
            results.append({"check": f"{name}_bess_soc_min",
                            "status": "PASS" if viol_min < 1000 else "WARN",
                            "violations": int(viol_min), "limit": soc_min})

    # Voltage plausibility
    if "pcc_v_a_pu" in df.columns:
        viol_lo = (df["pcc_v_a_pu"] < VOLTAGE_MIN_PLAUSIBLE_PU).sum()
        viol_hi = (df["pcc_v_a_pu"] > VOLTAGE_MAX_PLAUSIBLE_PU).sum()
        results.append({"check": f"{name}_voltage_plausible",
                        "status": "PASS" if (viol_lo + viol_hi) == 0 else "WARN",
                        "below_min": int(viol_lo), "above_max": int(viol_hi)})

    # Voltage unbalance
    if "pcc_voltage_unbalance_pu" in df.columns:
        unb = df["pcc_voltage_unbalance_pu"].values
        n_warn = int((unb >= VOLTAGE_UNBALANCE_WARN_PU).sum())
        n_hard = int((unb >= VOLTAGE_UNBALANCE_HARD_PU).sum())
        n_fail = int((unb >= VOLTAGE_UNBALANCE_FAIL_PU).sum())
        max_unb = float(unb.max())

        # For clean data: hard threshold check
        if name == "clean":
            status = "PASS" if n_hard == 0 else "FAIL"
        else:
            # For attacked: only FAIL if unexplained extreme violations
            status = "PASS" if n_fail < 100 else "FAIL"

        results.append({
            "check": f"{name}_voltage_unbalance",
            "status": status,
            "max_unbalance_pu": round(max_unb, 5),
            "rows_above_warn_0012": n_warn,
            "rows_above_hard_0020": n_hard,
            "rows_above_fail_0030": n_fail,
        })

    # Rating contamination check (must not use 450 kW or 300 kW)
    if "pv_s_rated_kva" in df.columns:
        bad_ratings = df["pv_s_rated_kva"].isin([500.0, 450.0 * 1.111]).sum()
        results.append({"check": f"{name}_no_generic_pv_rating_contamination",
                        "status": "PASS" if bad_ratings == 0 else "FAIL",
                        "contaminated_rows": int(bad_ratings)})

    # generation_method exists, summarized, and only allowed values
    if "generation_method" in df.columns:
        gm_counts = df["generation_method"].value_counts().to_dict()
        results.append({"check": f"{name}_generation_method_summary",
                        "status": "PASS",
                        "counts": {str(k): int(v) for k, v in gm_counts.items()}})
        bad_methods = df[~df["generation_method"].isin(ALLOWED_GENERATION_METHODS)]
        results.append({
            "check": f"{name}_generation_method_allowed_values",
            "status": "PASS" if len(bad_methods) == 0 else "FAIL",
            "unknown_count": int(len(bad_methods)),
            "allowed": sorted(ALLOWED_GENERATION_METHODS),
        })

    return results


def main() -> dict:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    all_results = []

    if not CLEAN_PHYSICAL_CSV.exists():
        all_results.append({"check": "clean_csv_exists", "status": "FAIL",
                            "error": "Clean physical CSV not found"})
    else:
        print("Validating clean physical dataset...")
        clean = pd.read_csv(CLEAN_PHYSICAL_CSV)
        all_results.extend(validate_dataset(clean, "clean", meta))

    if not ATTACKED_PHYSICAL_CSV.exists():
        all_results.append({"check": "attacked_csv_exists", "status": "FAIL",
                            "error": "Attacked physical CSV not found"})
    else:
        print("Validating attacked physical dataset...")
        attacked = pd.read_csv(ATTACKED_PHYSICAL_CSV,
                               dtype={"voltage_unbalance_status": str, "pv_inverter_mode": str})
        all_results.extend(validate_dataset(attacked, "attacked", meta))

    # Alignment check: time_s values match between clean and attacked
    if CLEAN_PHYSICAL_CSV.exists() and ATTACKED_PHYSICAL_CSV.exists():
        clean_ts = pd.read_csv(CLEAN_PHYSICAL_CSV, usecols=["time_s"])["time_s"].values
        att_ts = pd.read_csv(ATTACKED_PHYSICAL_CSV, usecols=["time_s"])["time_s"].values
        aligned = (len(clean_ts) == len(att_ts) and
                   len(clean_ts) > 0 and
                   (clean_ts == att_ts).all())
        all_results.append({"check": "clean_attacked_time_s_alignment",
                            "status": "PASS" if aligned else "FAIL"})

    # Load OpenDSS event-window results if available
    ods_results_path = METADATA / "opendss_event_window_results.json"
    ods_stats = {}
    if ods_results_path.exists():
        with open(ods_results_path, encoding="utf-8") as f:
            ods_stats = json.load(f)
    ods_successes = int(ods_stats.get("successes", 0))
    ods_failures = int(ods_stats.get("failures", 0))
    ods_resolved_rows = int(ods_stats.get("resolved_rows", 0))
    ods_stage = int(ods_stats.get("stage", 0))

    summary = aggregate_results(all_results)
    summary["results"] = all_results
    summary["opendss_event_window_successes"] = ods_successes
    summary["opendss_event_window_failures"] = ods_failures
    summary["opendss_event_window_resolved_rows"] = ods_resolved_rows
    summary["opendss_event_window_stage"] = ods_stage
    save_json(summary, PHYSICAL_CONSTRAINTS_JSON)

    # Generate markdown report
    lines = [
        "# Physical Constraints Validation Report",
        "",
        f"**Overall:** `{summary['overall']}`",
        f"**Passed:** {summary['passed']} / {summary['total']}",
        f"**Warned:** {summary['warned']}",
        f"**Failed:** {summary['failed']}",
        "",
        "## OpenDSS Event-Window Resolution Stats",
        "",
        f"- Stage: {ods_stage if ods_stage else 'not yet run'}",
        f"- Scenarios succeeded: {ods_successes}",
        f"- Scenarios failed: {ods_failures}",
        f"- Rows resolved to opendss_event_window_resolved: {ods_resolved_rows:,}",
        "",
        "## Check Details",
        "",
    ]
    for r in all_results:
        icon = "✓" if r.get("status") == "PASS" else ("⚠" if r.get("status") == "WARN" else "✗")
        lines.append(f"- {icon} `{r.get('check')}`: **{r.get('status')}**"
                     + (f" — violations={r.get('violations', '')}" if "violations" in r else "")
                     + (f" — counts={r.get('counts', '')}" if "counts" in r else ""))

    write_report(lines, PHYSICAL_CONSTRAINTS_REPORT)
    print(f"Physical validation: {summary['overall']} "
          f"({summary['passed']}/{summary['total']} checks passed)")
    return summary


if __name__ == "__main__":
    r = main()
    if r["overall"] == "FAIL":
        sys.exit(1)
