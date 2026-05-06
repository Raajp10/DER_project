"""
Phase 2 Zero-Day Evaluation Setup Runner.

Orchestrates all Phase 2 setup steps in order:
  1. Discover raw scenario JSON files
  2. Validate scenario bundles (apply safe repairs, copy accepted)
  3. Compile zero-day dataset (clean CSV + effects → attacked CSV + labels)
  4. Build zero-day windows (same Phase 1 feature order)
  5. Write setup report and verdict

DOES NOT run frozen model evaluation (deferred until Phase 1 is complete).

Usage:
  python run_phase2_zero_day_setup.py
  run_phase2_zero_day_setup.bat
"""
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

PHASE2_ROOT = Path(r"D:\updated_dataset\phase2_zero_day_eval")
SCRIPTS_DIR = PHASE2_ROOT / "scripts"
REPORTS_DIR = PHASE2_ROOT / "reports"
OUTPUTS_DIR = PHASE2_ROOT / "outputs"

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

STEPS = [
    {
        "name": "validate_scenarios",
        "script": SCRIPTS_DIR / "validate_zero_day_scenarios.py",
        "description": "Validate scenario bundles and apply safe repairs",
        "required": True,
    },
    {
        "name": "compile_dataset",
        "script": SCRIPTS_DIR / "compile_zero_day_dataset.py",
        "description": "Compile zero-day attacked dataset from clean CSV",
        "required": True,
    },
    {
        "name": "build_windows",
        "script": SCRIPTS_DIR / "build_zero_day_windows.py",
        "description": "Build zero-day feature windows (Phase 1 feature order)",
        "required": True,
    },
]

NOT_RUN_STEPS = [
    {
        "name": "frozen_model_eval",
        "script": SCRIPTS_DIR / "evaluate_frozen_models_on_zero_day.py",
        "reason": "Deferred until Phase 1 training and evaluation is complete. "
                  "Run manually: python scripts/evaluate_frozen_models_on_zero_day.py",
    },
]


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(step: dict) -> dict:
    script = step["script"]
    name   = step["name"]
    desc   = step["description"]

    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"  {desc}")
    print(f"  Script: {script.name}")
    print(f"{'='*60}")

    if not script.exists():
        msg = f"Script not found: {script}"
        print(f"[ERROR] {msg}")
        return {"step": name, "status": "ERROR", "error": msg, "elapsed_s": 0}

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        return {"step": name, "status": "OK", "elapsed_s": round(elapsed, 1)}
    else:
        print(f"\n[FAIL] {name} exited with code {result.returncode} after {elapsed:.1f}s")
        return {
            "step": name,
            "status": "FAIL",
            "returncode": result.returncode,
            "elapsed_s": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Summarise outputs
# ---------------------------------------------------------------------------

def _collect_output_summary() -> dict:
    summary: dict = {}

    # Validation summary
    val_path = OUTPUTS_DIR / "zero_day_validation_summary.json"
    if val_path.exists():
        with open(val_path) as f:
            summary["validation"] = json.load(f)

    # Window build summary
    win_path = OUTPUTS_DIR / "zero_day_window_build_summary.json"
    if win_path.exists():
        with open(win_path) as f:
            summary["windows"] = json.load(f)

    # Attacked CSV
    atk_csv = OUTPUTS_DIR / "zero_day_physical_attacked.csv"
    if atk_csv.exists():
        summary["attacked_csv_exists"] = True
        summary["attacked_csv_path"] = str(atk_csv)
    else:
        summary["attacked_csv_exists"] = False

    # Validated bundles
    vdir = PHASE2_ROOT / "scenarios" / "scenario_bundles_validated"
    if vdir.exists():
        validated = list(vdir.glob("*.json"))
        summary["validated_bundles"] = [v.name for v in validated]
        summary["n_validated_bundles"] = len(validated)
    else:
        summary["n_validated_bundles"] = 0

    return summary


# ---------------------------------------------------------------------------
# Write reports
# ---------------------------------------------------------------------------

def _write_setup_report(step_results: list, output_summary: dict,
                        total_elapsed: float) -> None:
    all_ok = all(r["status"] == "OK" for r in step_results)
    verdict = "PASS" if all_ok else "PARTIAL" if any(r["status"] == "OK" for r in step_results) else "FAIL"

    lines = [
        "# PHASE2_ZERO_DAY_SETUP_REPORT",
        "",
        f"Generated: {NOW}",
        f"Verdict: **{verdict}**",
        f"Total elapsed: {total_elapsed:.1f}s",
        "",
        "## Step Results",
        "",
        "| Step | Status | Elapsed |",
        "|---|---|---|",
    ]
    for r in step_results:
        status = r["status"]
        symbol = "✓" if status == "OK" else "✗"
        lines.append(f"| {r['step']} | {symbol} {status} | {r.get('elapsed_s', '-')}s |")

    lines += ["", "## Deferred Steps (NOT RUN)", ""]
    for s in NOT_RUN_STEPS:
        lines.append(f"- **{s['name']}**: {s['reason']}")

    # Validation details
    val = output_summary.get("validation", {})
    if val:
        lines += ["", "## Scenario Validation Summary", ""]
        for bundle_name, result in val.get("bundles", {}).items():
            status = result.get("status", "UNKNOWN")
            n_rep = len(result.get("repairs", []))
            n_err = len(result.get("errors", []))
            lines.append(f"- **{bundle_name}**: {status} "
                         f"({n_rep} repairs, {n_err} errors)")

    # Window stats
    win = output_summary.get("windows", {})
    if win:
        lines += ["", "## Zero-Day Window Stats", ""]
        lines.append(f"- Total windows : {win.get('total_windows', '?'):,}")
        lines.append(f"- Active windows: {win.get('n_active_zd_windows', '?'):,}")
        lines.append(f"- Flat shape    : {win.get('flat_shape', '?')}")

    # Validated bundles
    lines += ["", "## Validated Bundles", ""]
    n_val = output_summary.get("n_validated_bundles", 0)
    lines.append(f"- Validated bundles ready: {n_val}")
    for b in output_summary.get("validated_bundles", []):
        lines.append(f"  - `{b}`")

    lines += [
        "",
        "## Next Step",
        "",
        "1. Complete Phase 1 training: `python models/run_phase1_models.py`",
        "2. Then run frozen evaluation: "
        "`python phase2_zero_day_eval/scripts/evaluate_frozen_models_on_zero_day.py`",
        "",
        "---",
        "*Phase 2 Zero-Day Setup complete*",
    ]

    report_path = REPORTS_DIR / "PHASE2_ZERO_DAY_SETUP_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[INFO] Setup report written: {report_path}")


def _write_verdict_report(step_results: list, output_summary: dict) -> None:
    all_ok = all(r["status"] == "OK" for r in step_results)

    lines = [
        "# PHASE2_ZERO_DAY_FINAL_VERDICT",
        "",
        f"Generated: {NOW}",
        "",
    ]

    if all_ok:
        lines += [
            "## Verdict: SETUP COMPLETE — READY FOR FROZEN EVALUATION",
            "",
            "All Phase 2 setup steps passed.  The zero-day attacked dataset and",
            "feature windows are built and validated.  Frozen model evaluation is",
            "deferred until Phase 1 training is confirmed complete.",
            "",
        ]
    else:
        failed = [r["step"] for r in step_results if r["status"] != "OK"]
        lines += [
            "## Verdict: SETUP INCOMPLETE",
            "",
            f"Failed steps: {', '.join(failed)}",
            "",
            "Review step logs above and re-run after fixing reported errors.",
            "",
        ]

    val = output_summary.get("validation", {})
    bundles_status = []
    for b, r in val.get("bundles", {}).items():
        bundles_status.append(f"  - {b}: **{r.get('status', '?')}**")

    if bundles_status:
        lines += ["## Scenario Bundle Acceptance", ""]
        lines.extend(bundles_status)
        lines.append("")

    lines += [
        "## Constraints Honoured Throughout",
        "",
        "- No Phase 1 scripts modified",
        "- No model retraining or threshold recalibration",
        "- Cyber context: event-level metadata only",
        "- No packet-level fields or real telemetry claims",
        "- Physical effects bounded within schema-defined limits",
        "- Frozen model evaluation deferred until Phase 1 complete",
        "",
        "---",
        "*Phase 2 Zero-Day Evaluation Infrastructure*",
    ]

    report_path = REPORTS_DIR / "PHASE2_ZERO_DAY_FINAL_VERDICT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Verdict report written: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 Zero-Day Evaluation Setup")
    print(f"Started: {NOW}")
    print("=" * 60)
    print()
    print("NOTE: Frozen model evaluation is NOT run here.")
    print("      Run it manually after Phase 1 is complete.")
    print()

    t_total = time.time()
    step_results = []

    for step in STEPS:
        result = run_step(step)
        step_results.append(result)
        if result["status"] != "OK" and step.get("required"):
            print(f"\n[ABORT] Required step '{step['name']}' failed. Stopping.")
            break

    total_elapsed = time.time() - t_total

    print(f"\n{'='*60}")
    print("SETUP COMPLETE")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"{'='*60}")
    for r in step_results:
        symbol = "OK  " if r["status"] == "OK" else "FAIL"
        print(f"  [{symbol}] {r['step']}")
    for s in NOT_RUN_STEPS:
        print(f"  [SKIP] {s['name']} (deferred)")

    output_summary = _collect_output_summary()
    _write_setup_report(step_results, output_summary, total_elapsed)
    _write_verdict_report(step_results, output_summary)

    all_ok = all(r["status"] == "OK" for r in step_results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
