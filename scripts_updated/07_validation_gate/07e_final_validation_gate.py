"""
Final validation gate — aggregates all prior validation results.
Writes: data_updated/validation/final_updated_dataset_validation_report.md
        data_updated/validation/final_updated_dataset_validation_summary.json
        reports/05_validation_summary_report.md
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import (
    PHYSICAL_CONSTRAINTS_JSON, CYBER_VALIDATION_JSON, CONTEXT_VALIDATION_JSON,
    FINAL_VALIDATION_REPORT, FINAL_VALIDATION_JSON, REPORTS, VALIDATION,
    FIGURES, CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, CYBER_ANOMALOUS_CSV,
    EVIDENCE_PACKETS_JSONL, LIFECYCLE_MAP_CSV, CONTEXT_WINDOWS_CSV, METADATA,
)
from validation_utils import write_report

ALLOWED_GENERATION_METHODS = {
    "opendss_clean_baseline",
    "opendss_event_window_resolved",
    "physics_constrained_surrogate",
    "csv_rule_legacy",
}

REQUIRED_FIGURES = [
    "physical_clean_vs_attacked_overview.png",
    "scenario_physical_effect_examples.png",
    "cyber_lifecycle_timeline_examples.png",
    "cyber_anomaly_distribution.png",
    "context_command_to_physical_response_alignment.png",
    "generation_method_summary.png",
    "protocol_claim_level_summary.png",
    "final_validation_dashboard.png",
    # DER component visualizations
    "der_component_overview.png",
    "der_pv_output_profile.png",
    "der_bess_soc_profile.png",
    "der_bess_power_profile.png",
    "der_pcc_voltage_profile.png",
    # OpenDSS event-window summary
    "opendss_event_window_resolution_summary.png",
    "scenario_resolution_method_by_type.png",
]


def load_sub_result(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"overall": "FAIL", "error": f"{path.name} not found"}


def check_figure(fname: str) -> dict:
    fpath = FIGURES / fname
    exists = fpath.exists()
    size = fpath.stat().st_size if exists else 0
    return {"file": fname, "exists": exists, "size_bytes": size,
            "status": "PASS" if (exists and size > 0) else "FAIL"}


def main() -> dict:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    # Load sub-validation results
    phys = load_sub_result(PHYSICAL_CONSTRAINTS_JSON)
    cyber = load_sub_result(CYBER_VALIDATION_JSON)
    context = load_sub_result(CONTEXT_VALIDATION_JSON)

    # Check required output files exist
    file_checks = []
    for path, name in [
        (CLEAN_PHYSICAL_CSV, "clean_physical_csv"),
        (ATTACKED_PHYSICAL_CSV, "attacked_physical_csv"),
        (CYBER_ANOMALOUS_CSV, "anomalous_cyber_log"),
        (EVIDENCE_PACKETS_JSONL, "evidence_packets"),
        (LIFECYCLE_MAP_CSV, "lifecycle_map"),
        (CONTEXT_WINDOWS_CSV, "context_windows"),
    ]:
        ok = path.exists() and path.stat().st_size > 0
        file_checks.append({"check": f"output_{name}_exists",
                            "status": "PASS" if ok else "FAIL",
                            "path": str(path)})

    # Check figures
    fig_checks = [check_figure(f) for f in REQUIRED_FIGURES]
    figs_passed = sum(1 for f in fig_checks if f["status"] == "PASS")
    figs_failed = len(fig_checks) - figs_passed

    # Aggregate gate status
    gates = {
        "physical": phys.get("overall", "FAIL"),
        "cyber": cyber.get("overall", "FAIL"),
        "context": context.get("overall", "FAIL"),
        "files": "PASS" if all(c["status"] == "PASS" for c in file_checks) else "FAIL",
        "figures": "PASS" if figs_failed == 0 else "WARN",
    }

    any_fail = any(v == "FAIL" for v in gates.values())
    overall = "FAIL" if any_fail else "PASS"

    # Determine generation method counts from physical data
    gen_counts = {}
    gm_validation_status = "PASS"
    if ATTACKED_PHYSICAL_CSV.exists():
        import pandas as pd
        df = pd.read_csv(ATTACKED_PHYSICAL_CSV, usecols=["generation_method"])
        gen_counts = df["generation_method"].value_counts().to_dict()
        gen_counts = {str(k): int(v) for k, v in gen_counts.items()}
        unknown_methods = set(gen_counts.keys()) - ALLOWED_GENERATION_METHODS
        if unknown_methods:
            gm_validation_status = "FAIL"
            file_checks.append({
                "check": "generation_method_allowed_values",
                "status": "FAIL",
                "unknown_methods": sorted(unknown_methods),
            })
        else:
            file_checks.append({
                "check": "generation_method_allowed_values",
                "status": "PASS",
                "methods_found": sorted(gen_counts.keys()),
            })

    protocol_counts = {}
    if CYBER_ANOMALOUS_CSV.exists():
        import pandas as pd
        df = pd.read_csv(CYBER_ANOMALOUS_CSV, usecols=["protocol_claim_level"])
        protocol_counts = df["protocol_claim_level"].value_counts().to_dict()
        protocol_counts = {str(k): int(v) for k, v in protocol_counts.items()}

    # Load OpenDSS event-window results
    ods_results_path = METADATA / "opendss_event_window_results.json"
    ods_stats = {}
    if ods_results_path.exists():
        with open(ods_results_path, encoding="utf-8") as f:
            ods_stats = json.load(f)
    ods_successes = int(ods_stats.get("successes", 0))
    ods_failures = int(ods_stats.get("failures", 0))
    ods_resolved_rows = int(ods_stats.get("resolved_rows", 0))
    ods_stage = int(ods_stats.get("stage", 0))

    # Determine final claim — Cases A / B / C
    opendss_ew_rows = int(gen_counts.get("opendss_event_window_resolved", 0))
    surrogate_rows = int(gen_counts.get("physics_constrained_surrogate", 0))
    total_attacked = opendss_ew_rows + surrogate_rows

    if total_attacked > 0 and opendss_ew_rows == total_attacked:
        # Case C: All attacked rows resolved via OpenDSS event-window
        claim_case = "C"
        final_claim = (
            "The updated dataset uses OpenDSS-based IEEE 123-bus clean simulations and "
            "full OpenDSS event-window-resolved anomalous DER simulations for all attack "
            "scenarios, combined with IEEE 1547-informed DER operating constraints and an "
            "IEEE 2030.5-style semantic cyber event lifecycle. Cyber and physical layers are "
            "aligned through event-specific command-to-response context windows for "
            "cyber-physical anomaly detection."
        )
    elif opendss_ew_rows > 0:
        # Case B: Partial OpenDSS event-window resolution
        claim_case = "B"
        final_claim = (
            f"The updated dataset combines OpenDSS-based IEEE 123-bus clean DER simulation outputs "
            f"with partially OpenDSS event-window-resolved anomalous DER responses "
            f"({opendss_ew_rows:,} rows resolved, {surrogate_rows:,} rows physics-constrained "
            f"surrogate), IEEE 1547-informed DER operating constraints, and an IEEE 2030.5-style "
            f"semantic cyber event lifecycle. Cyber and physical layers are aligned through "
            f"event-specific command-to-response context windows for cyber-physical anomaly detection."
        )
    else:
        # Case A: All surrogate — current state
        claim_case = "A"
        final_claim = (
            "The updated dataset combines OpenDSS-based IEEE 123-bus clean DER simulation outputs "
            "with physics-constrained surrogate anomalous DER responses, IEEE 1547-informed DER "
            "operating constraints, and an IEEE 2030.5-style semantic cyber event lifecycle "
            "representing DER control, status, metering, and response behavior. Cyber and physical "
            "layers are aligned through event-specific command-to-response context windows for "
            "cyber-physical anomaly detection."
        )

    # Recompute gate status after adding generation_method check
    gates["files"] = "PASS" if all(c["status"] == "PASS" for c in file_checks) else "FAIL"
    any_fail = any(v == "FAIL" for v in gates.values())
    overall = "FAIL" if any_fail else "PASS"

    summary = {
        "run_time_utc": now,
        "overall": overall,
        "gates": gates,
        "generation_method_counts": gen_counts,
        "protocol_claim_level_counts": protocol_counts,
        "figures_passed": figs_passed,
        "figures_failed": figs_failed,
        "final_claim": final_claim,
        "claim_case": claim_case,
        "opendss_event_window_rows": opendss_ew_rows,
        "surrogate_rows": surrogate_rows,
        "opendss_event_window_successes": ods_successes,
        "opendss_event_window_failures": ods_failures,
        "opendss_event_window_resolved_rows": ods_resolved_rows,
        "opendss_event_window_stage": ods_stage,
    }

    with open(FINAL_VALIDATION_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Final Updated Dataset Validation Report",
        "",
        f"**Generated:** {now}",
        f"**Overall Status:** `{overall}`",
        "",
        "## Validation Gates",
        "",
    ]
    for gate, status in gates.items():
        icon = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        lines.append(f"- {icon} `{gate}`: **{status}**")

    lines += [
        "",
        "## Output File Checks",
        "",
    ]
    for c in file_checks:
        icon = "✓" if c["status"] == "PASS" else "✗"
        lines.append(f"- {icon} `{c['check']}`: **{c['status']}**")

    lines += [
        "",
        "## Figure Checks",
        "",
    ]
    for f in fig_checks:
        icon = "✓" if f["status"] == "PASS" else "✗"
        lines.append(f"- {icon} `{f['file']}` ({f['size_bytes']} bytes): **{f['status']}**")

    lines += [
        "",
        "## Generation Method Counts",
        "",
    ]
    for k, v in gen_counts.items():
        lines.append(f"- `{k}`: {v:,} rows")

    lines += [
        "",
        "## Protocol Claim Level Counts",
        "",
    ]
    for k, v in protocol_counts.items():
        lines.append(f"- `{k}`: {v:,} events")

    lines += [
        "",
        "## OpenDSS Event-Window Resolution",
        "",
        f"- Stage executed: {ods_stage if ods_stage else 'not yet run'}",
        f"- Scenarios succeeded: {ods_successes}",
        f"- Scenarios failed: {ods_failures}",
        f"- Rows resolved to opendss_event_window_resolved: {ods_resolved_rows:,}",
        f"- Rows remaining as physics_constrained_surrogate: {surrogate_rows:,}",
        "",
        "## Final Claim",
        "",
        f"**Claim Case: {claim_case}** "
        + ("(All surrogate)" if claim_case == "A" else
           "(Partial OpenDSS)" if claim_case == "B" else "(All OpenDSS)"),
        "",
        f"> {final_claim}",
        "",
    ]

    if overall == "PASS":
        lines += [
            "```",
            "UPDATED DATASET FULL PASS",
            "```",
        ]
    else:
        lines += [
            "```",
            f"VALIDATION FAILED — see gate details above",
            "```",
        ]

    write_report(lines, FINAL_VALIDATION_REPORT)
    write_report(lines, REPORTS / "05_validation_summary_report.md")

    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION: {overall}")
    for gate, status in gates.items():
        print(f"  {gate}: {status}")
    if overall == "PASS":
        print("\nUPDATED DATASET FULL PASS")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    r = main()
    if r["overall"] == "FAIL":
        sys.exit(1)
