"""Validate context window causality and flag integrity."""
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
    LIFECYCLE_MAP_CSV, CONTEXT_WINDOWS_CSV, EVIDENCE_PACKETS_JSONL,
    CONTEXT_VALIDATION_REPORT, CONTEXT_VALIDATION_JSON, VALIDATION,
)
from validation_utils import (
    aggregate_results, save_json, write_report,
)


def main() -> dict:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    results = []

    # Check required files
    for path, name in [(LIFECYCLE_MAP_CSV, "lifecycle_map"),
                       (CONTEXT_WINDOWS_CSV, "context_windows"),
                       (EVIDENCE_PACKETS_JSONL, "evidence_packets")]:
        exists = path.exists()
        results.append({"check": f"{name}_exists",
                       "status": "PASS" if exists else "FAIL",
                       "path": str(path)})

    if not CONTEXT_WINDOWS_CSV.exists():
        save_json(aggregate_results(results), CONTEXT_VALIDATION_JSON)
        write_report(["# Context Causality Validation", "", "**FAIL: Files missing**"],
                     CONTEXT_VALIDATION_REPORT)
        return aggregate_results(results)

    ctx = pd.read_csv(CONTEXT_WINDOWS_CSV)

    # Required columns
    for col in ["cyber_active_flag", "command_active_flag",
                "physical_effect_active_flag", "attack_active_flag",
                "generation_method", "protocol_claim_level"]:
        results.append({"check": f"ctx_{col}_exists",
                       "status": "PASS" if col in ctx.columns else "FAIL"})

    # Attack flag not blindly copied to every row
    if "attack_active_flag" in ctx.columns and len(ctx) > 0:
        attack_frac = ctx["attack_active_flag"].mean()
        results.append({"check": "attack_flag_not_all_rows",
                       "status": "PASS" if attack_frac < 0.5 else "WARN",
                       "attack_fraction": round(float(attack_frac), 3)})

    # For cyber-only events: physical_effect_active_flag should be 0
    if all(c in ctx.columns for c in ["final_event_class", "physical_effect_active_flag"]):
        cyber_only = ctx[ctx["final_event_class"] == "cyber_only"]
        if len(cyber_only) > 0:
            phys_flag_in_cyber_only = (cyber_only["physical_effect_active_flag"] == 1).mean()
            results.append({"check": "cyber_only_no_physical_flag",
                           "status": "PASS" if phys_flag_in_cyber_only == 0.0 else "WARN",
                           "fraction_with_phys_flag": round(float(phys_flag_in_cyber_only), 3)})

    # For physical-only: cyber_active_flag should be 0 (except for background monitoring)
    if all(c in ctx.columns for c in ["final_event_class", "cyber_active_flag"]):
        phys_only = ctx[ctx["final_event_class"] == "physical_only"]
        if len(phys_only) > 0:
            cyber_in_phys = (phys_only["cyber_active_flag"] == 1).mean()
            results.append({"check": "physical_only_no_malicious_cyber_flag",
                           "status": "PASS" if cyber_in_phys < 0.2 else "WARN",
                           "fraction": round(float(cyber_in_phys), 3)})

    # For delayed scenarios: physical effect must start after command_apply
    if all(c in ctx.columns for c in ["anomaly_type", "relative_time_to_apply_s",
                                       "physical_effect_active_flag"]):
        delayed = ctx[ctx["anomaly_type"] == "delayed_pv_limit"]
        if len(delayed) > 0:
            # Physical effect rows should have positive relative_time_to_apply_s
            phys_rows = delayed[delayed["physical_effect_active_flag"] == 1]
            if len(phys_rows) > 0:
                ok = (phys_rows["relative_time_to_apply_s"] >= -5).all()
                results.append({"check": "delayed_physical_after_apply_time",
                               "status": "PASS" if ok else "FAIL",
                               "phys_rows": int(len(phys_rows))})

    # generation_method preserved in context windows
    if "generation_method" in ctx.columns:
        missing = ctx["generation_method"].isna().sum()
        results.append({"check": "ctx_generation_method_no_null",
                       "status": "PASS" if missing == 0 else "WARN",
                       "null_count": int(missing)})

    # protocol_claim_level preserved
    if "protocol_claim_level" in ctx.columns:
        missing = ctx["protocol_claim_level"].isna().sum()
        results.append({"check": "ctx_protocol_claim_level_no_null",
                       "status": "PASS" if missing == 0 else "WARN",
                       "null_count": int(missing)})

    # Evidence packets file size > 0
    if EVIDENCE_PACKETS_JSONL.exists():
        size = EVIDENCE_PACKETS_JSONL.stat().st_size
        results.append({"check": "evidence_packets_not_empty",
                       "status": "PASS" if size > 0 else "FAIL",
                       "size_bytes": size})

    summary = aggregate_results(results)
    summary["results"] = results
    save_json(summary, CONTEXT_VALIDATION_JSON)

    lines = [
        "# Context Causality Validation Report",
        "",
        f"**Overall:** `{summary['overall']}`",
        f"**Passed:** {summary['passed']} / {summary['total']}",
        "",
        "## Check Details",
        "",
    ]
    for r in results:
        icon = "✓" if r.get("status") == "PASS" else ("⚠" if r.get("status") == "WARN" else "✗")
        lines.append(f"- {icon} `{r.get('check')}`: **{r.get('status')}**")
    write_report(lines, CONTEXT_VALIDATION_REPORT)

    print(f"Context validation: {summary['overall']}")
    return summary


if __name__ == "__main__":
    main()
