"""Validate alignment between clean and attacked physical datasets."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import (
    CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, VALIDATION,
)
from validation_utils import write_report, aggregate_results, save_json

ALIGNMENT_REPORT = VALIDATION / "clean_attacked_alignment_report.md"
ALIGNMENT_JSON = VALIDATION / "clean_attacked_alignment_summary.json"


def main() -> dict:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    results = []

    if not CLEAN_PHYSICAL_CSV.exists() or not ATTACKED_PHYSICAL_CSV.exists():
        results.append({"check": "files_exist", "status": "FAIL",
                       "error": "One or both physical CSVs missing"})
        save_json(aggregate_results(results), ALIGNMENT_JSON)
        write_report(["# Alignment Validation", "", "**FAIL: Files missing**"], ALIGNMENT_REPORT)
        return aggregate_results(results)

    print("Checking clean/attacked alignment...")
    clean = pd.read_csv(CLEAN_PHYSICAL_CSV, usecols=["time_s", "timestamp_utc"])
    attacked = pd.read_csv(ATTACKED_PHYSICAL_CSV, usecols=["time_s", "timestamp_utc"])

    # Row count match
    ok = len(clean) == len(attacked)
    results.append({"check": "row_count_match", "status": "PASS" if ok else "FAIL",
                   "clean": len(clean), "attacked": len(attacked)})

    # time_s exact match
    if len(clean) == len(attacked):
        aligned = (clean["time_s"].values == attacked["time_s"].values).all()
        results.append({"check": "time_s_exact_match",
                       "status": "PASS" if aligned else "FAIL"})

        # Timestamp match (first/last)
        results.append({"check": "first_timestamp_match",
                       "status": "PASS" if (str(clean["timestamp_utc"].iloc[0]) ==
                                           str(attacked["timestamp_utc"].iloc[0])) else "FAIL",
                       "clean": str(clean["timestamp_utc"].iloc[0]),
                       "attacked": str(attacked["timestamp_utc"].iloc[0])})
        results.append({"check": "last_timestamp_match",
                       "status": "PASS" if (str(clean["timestamp_utc"].iloc[-1]) ==
                                           str(attacked["timestamp_utc"].iloc[-1])) else "FAIL",
                       "clean": str(clean["timestamp_utc"].iloc[-1]),
                       "attacked": str(attacked["timestamp_utc"].iloc[-1])})

        # No gaps in clean
        diffs = np.diff(clean["time_s"].values)
        gaps = int((diffs != 1).sum())
        results.append({"check": "clean_no_time_gaps",
                       "status": "PASS" if gaps == 0 else "FAIL", "gaps": gaps})

        # No gaps in attacked
        diffs_a = np.diff(attacked["time_s"].values)
        gaps_a = int((diffs_a != 1).sum())
        results.append({"check": "attacked_no_time_gaps",
                       "status": "PASS" if gaps_a == 0 else "FAIL", "gaps": gaps_a})

    summary = aggregate_results(results)
    summary["results"] = results
    save_json(summary, ALIGNMENT_JSON)

    lines = [
        "# Clean/Attacked Alignment Report",
        "",
        f"**Overall:** `{summary['overall']}`",
        "",
    ]
    for r in results:
        icon = "✓" if r.get("status") == "PASS" else "✗"
        lines.append(f"- {icon} `{r['check']}`: **{r['status']}**")
    write_report(lines, ALIGNMENT_REPORT)

    print(f"Alignment validation: {summary['overall']}")
    return summary


if __name__ == "__main__":
    main()
