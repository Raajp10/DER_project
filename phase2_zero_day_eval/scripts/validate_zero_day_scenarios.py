"""
Phase 2 Zero-Day Scenario Validator.

Recursively finds all .json files under scenario_bundles_raw, validates each
bundle against schema and rules, applies safe documented repairs, copies
accepted bundles to scenario_bundles_validated.

Writes:
  reports/ZERO_DAY_SCENARIO_VALIDATION_REPORT.md
  outputs/zero_day_validation_summary.json
"""
import sys
import json
import re
import copy
import shutil
from pathlib import Path
from datetime import datetime

PHASE2_ROOT = Path(r"D:\updated_dataset\phase2_zero_day_eval")
RAW_DIR = PHASE2_ROOT / "scenarios" / "scenario_bundles_raw"
VALIDATED_DIR = PHASE2_ROOT / "scenarios" / "scenario_bundles_validated"
REPORTS_DIR = PHASE2_ROOT / "reports"
OUTPUTS_DIR = PHASE2_ROOT / "outputs"

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

REQUIRED_CLASS_COUNTS = {"physical_only": 4, "cyber_only": 4, "cyber_physical": 6, "normal": 2}
REQUIRED_SCENARIO_COUNT = 16
ALLOWED_ASSET_IDS = {"der_site_001", "pv_001", "bess_001", "pcc_001"}
FORBIDDEN_ASSET_IDS = {"pv35", "pv60", "pv83", "pv76", "pv49", "pv104", "pv114",
                        "bess48", "bess76", "bess108", "bess114"}
ALLOWED_CLASSES = {"normal", "cyber_only", "physical_only", "cyber_physical"}
ALLOWED_COMPONENTS = {"pv", "bess", "pcc", "measured_layer", "der_site"}
ALLOWED_FAMILIES = {
    "normal_control_variation", "soc_aware_bess_dispatch_anomaly",
    "pv_curtailment_mismatch", "stale_or_delayed_measurement",
    "command_delay", "command_suppression", "oscillatory_bess_control",
    "pcc_voltage_deviation", "coordinated_pv_bess_response",
    "false_data_injection_context_only", "availability_degradation_context_only",
    "physical_irradiance_like_disturbance",
}
ALLOWED_VARS = {
    "pv_p_kw", "pv_q_kvar", "bess_p_kw", "bess_q_kvar", "bess_soc_percent",
    "pcc_v_a_pu", "pcc_v_b_pu", "pcc_v_c_pu",
    "pcc_i_a_amp", "pcc_i_b_amp", "pcc_i_c_amp",
    "pcc_p_kw", "pcc_q_kvar", "irradiance_pu",
}
FORBIDDEN_VARS = {"temperature_c", "pcc_freq_hz", "frequency_hz", "grid_frequency", "bus_id", "ieee123_bus"}
PHYSICAL_BOUNDS = {
    "pv_p_kw": (0, 100), "pv_q_kvar": (-60, 60),
    "bess_p_kw": (-50, 50), "bess_q_kvar": (-50, 50), "bess_soc_percent": (5, 95),
    "pcc_v_a_pu": (0.90, 1.10), "pcc_v_b_pu": (0.90, 1.10), "pcc_v_c_pu": (0.90, 1.10),
    "pcc_i_a_amp": (0, 500), "pcc_i_b_amp": (0, 500), "pcc_i_c_amp": (0, 500),
    "pcc_p_kw": (-250, 250), "pcc_q_kvar": (-150, 150), "irradiance_pu": (0, 1.2),
}
FORBIDDEN_CYBER_FIELDS = {
    "payload", "packet_bytes", "tcp_flags", "exploit", "malware",
    "credential", "password", "hash", "shellcode", "buffer_overflow",
    "sql_injection", "cve_id", "metasploit", "cobalt_strike",
}
COMPAT_STATEMENT = ("I used only the final single-site dataset assets der_site_001, "
                    "pv_001, bess_001, pcc_001; I did not use old multi-DER asset IDs "
                    "or packet-level cyber fields.")
DAY_BOUNDARIES = [(i * 86400, i * 86400 + 86399) for i in range(7)]
AUTHOR_MAP = {
    "chatgpt_sceario": "chatgpt", "claude_sceario": "claude",
    "gemini_sceario": "gemini", "grok_sceario": "grok",
    "chatgpt_scenario": "chatgpt", "claude_scenario": "claude",
    "gemini_scenario": "gemini", "grok_scenario": "grok",
}
VALIDATED_NAMES = {
    "chatgpt": "chatgpt_validated.json",
    "claude": "claude_validated.json",
    "gemini": "gemini_validated.json",
    "grok": "grok_validated.json",
    "other": "other_validated.json",
}


def _infer_author_from_folder(json_path: Path) -> str:
    folder = json_path.parent.name.lower()
    for key, val in AUTHOR_MAP.items():
        if key in folder:
            return val
    return "other"


def _apply_safe_repairs(bundle: dict, repairs_applied: list) -> dict:
    """Apply documented safe repairs to a bundle in-place. Returns repaired copy."""
    b = copy.deepcopy(bundle)

    # 1. compatibility_statement → project_compatibility_statement
    if "compatibility_statement" in b and "project_compatibility_statement" not in b:
        b["project_compatibility_statement"] = b.pop("compatibility_statement")
        repairs_applied.append("field_alias_compatibility_statement: mapped 'compatibility_statement' -> 'project_compatibility_statement'")

    for sc in b.get("scenarios", []):
        sid = sc.get("scenario_id", "?")

        # 2. cyber_physical_type → scenario_class
        if "cyber_physical_type" in sc and "scenario_class" not in sc:
            sc["scenario_class"] = sc.pop("cyber_physical_type")
            repairs_applied.append(f"{sid}: field_alias_scenario_class: mapped 'cyber_physical_type' -> 'scenario_class'")

        # 3. affected_asset → target_asset_id
        if "affected_asset" in sc and "target_asset_id" not in sc:
            sc["target_asset_id"] = sc.pop("affected_asset")
            repairs_applied.append(f"{sid}: field_alias_target_asset_id: mapped 'affected_asset' -> 'target_asset_id'")

        # 4. missing scenario_name
        if "scenario_name" not in sc:
            sc["scenario_name"] = sid.replace("_", " ")
            repairs_applied.append(f"{sid}: missing_scenario_name: generated from scenario_id")

        # 5. missing target_component
        if "target_component" not in sc:
            asset = sc.get("target_asset_id", "")
            comp_map = {"pv_001": "pv", "bess_001": "bess", "pcc_001": "pcc", "der_site_001": "der_site"}
            sc["target_component"] = comp_map.get(asset, "der_site")
            repairs_applied.append(f"{sid}: missing_target_component: inferred '{sc['target_component']}' from asset")

        # 6. missing safety_note
        if "safety_note" not in sc:
            sc["safety_note"] = (
                "Synthetic/simulation-only defensive frozen-model evaluation scenario; "
                "no real-world telemetry or protocol compliance claim."
            )
            repairs_applied.append(f"{sid}: missing_safety_note: generated default safety_note")

        # 7-9. Effect type normalization
        for eff in sc.get("physical_effects", []):
            et = eff.get("effect_type", "")
            if et == "stale_value_hold":
                eff["effect_type"] = "hold_stale"
                repairs_applied.append(f"{sid}: effect_type_stale_value_hold: normalized to 'hold_stale'")
            elif et == "bounded_clip":
                eff["effect_type"] = "clip"
                repairs_applied.append(f"{sid}: effect_type_bounded_clip: normalized to 'clip'")
            elif et == "bounded_oscillation":
                eff["effect_type"] = "smooth_oscillation"
                repairs_applied.append(f"{sid}: effect_type_bounded_oscillation: normalized to 'smooth_oscillation'")

            # 9. negative magnitude with directional intent
            mag = eff.get("magnitude", 0)
            direction = eff.get("direction", "")
            if isinstance(mag, (int, float)) and mag < 0 and direction in ("increase", "decrease"):
                eff["magnitude"] = abs(mag)
                repairs_applied.append(f"{sid}/{eff.get('variable','?')}: neg_magnitude: replaced {mag} with {abs(mag)} (direction='{direction}' encodes sign)")

    return b


def _validate_bundle(bundle: dict, json_path: Path) -> tuple:
    """Returns (errors: list[str], warnings: list[str], repaired_bundle: dict, repairs: list[str])."""
    errors = []
    warnings = []
    repairs = []

    repaired = _apply_safe_repairs(bundle, repairs)

    # Top-level required fields
    for f in ["dataset_id", "author_model", "generation_purpose", "project_compatibility_statement", "scenarios"]:
        if f not in repaired:
            errors.append(f"Missing required top-level field: '{f}'")

    if errors:
        return errors, warnings, repaired, repairs

    # author_model
    am = repaired.get("author_model", "")
    if am not in ("chatgpt", "claude", "gemini", "grok", "other"):
        errors.append(f"Invalid author_model '{am}'. Must be one of: chatgpt, claude, gemini, grok, other")

    # generation_purpose
    gp = repaired.get("generation_purpose", "")
    if gp != "held_out_zero_day_like_frozen_model_evaluation":
        errors.append(f"generation_purpose must be 'held_out_zero_day_like_frozen_model_evaluation', got '{gp}'")

    # project_compatibility_statement
    pcs = repaired.get("project_compatibility_statement", "")
    if pcs != COMPAT_STATEMENT:
        errors.append(f"project_compatibility_statement does not match expected value. Got: '{pcs[:80]}...'")

    scenarios = repaired.get("scenarios", [])

    # Scenario count
    if len(scenarios) != REQUIRED_SCENARIO_COUNT:
        errors.append(f"Expected exactly {REQUIRED_SCENARIO_COUNT} scenarios, got {len(scenarios)}")

    # Unique scenario IDs
    seen_ids = {}
    for i, sc in enumerate(scenarios):
        sid = sc.get("scenario_id", f"<missing_{i}>")
        if sid in seen_ids:
            errors.append(f"Duplicate scenario_id: '{sid}' at index {i} and {seen_ids[sid]}")
        seen_ids[sid] = i

    # Class counts
    class_counts = {}
    for sc in scenarios:
        cls = sc.get("scenario_class", "")
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls, req in REQUIRED_CLASS_COUNTS.items():
        got = class_counts.get(cls, 0)
        if got != req:
            errors.append(f"Class count error: '{cls}' requires {req}, got {got}. "
                          f"Full counts: {class_counts}")

    # Per-scenario validation
    inferred_author = _infer_author_from_folder(json_path)
    for sc in scenarios:
        sid = sc.get("scenario_id", "?")
        cls = sc.get("scenario_class", "")
        family = sc.get("scenario_family", "")

        # scenario_id pattern and prefix
        if not re.match(r"zdl_(chatgpt|claude|gemini|grok|other)_[a-z0-9_]+_\d{3}$", sid):
            errors.append(f"{sid}: scenario_id does not match pattern zdl_<author>_<name>_<NNN>")
        else:
            # Extract only the author token (second segment between zdl_ and the next _)
            sid_author = re.match(r"zdl_(chatgpt|claude|gemini|grok|other)_", sid).group(1)
            if am and sid_author != am:
                errors.append(f"{sid}: scenario_id prefix '{sid_author}' does not match author_model '{am}'")

        # scenario_class
        if cls not in ALLOWED_CLASSES:
            errors.append(f"{sid}: invalid scenario_class '{cls}'")

        # scenario_family
        if family not in ALLOWED_FAMILIES:
            errors.append(f"{sid}: invalid scenario_family '{family}'")

        # target_asset_id and target_component
        asset = sc.get("target_asset_id", "")
        if asset in FORBIDDEN_ASSET_IDS:
            errors.append(f"{sid}: forbidden asset_id '{asset}'")
        elif asset not in ALLOWED_ASSET_IDS:
            errors.append(f"{sid}: unknown asset_id '{asset}'")

        comp = sc.get("target_component", "")
        if comp not in ALLOWED_COMPONENTS:
            errors.append(f"{sid}: invalid target_component '{comp}'")

        # timeline
        t_start = sc.get("start_time_s", -1)
        dur = sc.get("duration_s", -1)
        if not (0 <= t_start <= 604799):
            errors.append(f"{sid}: start_time_s={t_start} out of range [0, 604799]")
        if not (60 <= dur <= 1800):
            errors.append(f"{sid}: duration_s={dur} out of range [60, 1800]")
        if isinstance(t_start, int) and isinstance(dur, int):
            end = t_start + dur - 1
            if end > 604799:
                errors.append(f"{sid}: start_time_s + duration_s - 1 = {end} > 604799")

        # safety_note
        if "safety_note" not in sc:
            errors.append(f"{sid}: missing required 'safety_note' field")

        # labels
        labels = sc.get("labels", {})
        for lf in ["label_anomaly", "label_cyber_anomaly", "label_physical_anomaly"]:
            if lf not in labels:
                errors.append(f"{sid}: missing label '{lf}'")

        # label consistency
        effects = sc.get("physical_effects", [])
        la = labels.get("label_anomaly", -1)
        lca = labels.get("label_cyber_anomaly", -1)
        lpa = labels.get("label_physical_anomaly", -1)

        if cls == "cyber_only" and effects:
            errors.append(f"{sid}: cyber_only scenario must have empty physical_effects (has {len(effects)})")
        if cls == "normal":
            if effects:
                errors.append(f"{sid}: normal scenario must have empty physical_effects (has {len(effects)})")
            if la != 0 or lca != 0 or lpa != 0:
                errors.append(f"{sid}: normal scenario must have all-zero labels, got anomaly={la} cyber={lca} physical={lpa}")
        if cls in ("physical_only", "cyber_physical") and not effects:
            errors.append(f"{sid}: {cls} scenario must have non-empty physical_effects")

        # physical_effects validation
        for eff in effects:
            var = eff.get("variable", "")
            if var in FORBIDDEN_VARS:
                errors.append(f"{sid}: forbidden physical variable '{var}' (temperature_c cannot be modified; also check freq/bus fields)")
            elif var not in ALLOWED_VARS:
                errors.append(f"{sid}: unknown physical variable '{var}'")
            if var == "irradiance_pu" and family != "physical_irradiance_like_disturbance":
                errors.append(f"{sid}: irradiance_pu may only be modified in 'physical_irradiance_like_disturbance' family (got '{family}')")

            # magnitude bounds warning
            mag = eff.get("magnitude", 0)
            if isinstance(mag, (int, float)) and var in PHYSICAL_BOUNDS:
                lo, hi = PHYSICAL_BOUNDS[var]
                if abs(mag) > abs(hi - lo):
                    warnings.append(f"{sid}/{var}: magnitude {mag} may exceed variable range [{lo}, {hi}]")

        # cyber_context
        cc = sc.get("cyber_context", {})
        if cc.get("event_level_only") is not True:
            errors.append(f"{sid}: cyber_context.event_level_only must be true")
        if cc.get("packet_level_protocol_compliance_claimed") is not False:
            errors.append(f"{sid}: cyber_context.packet_level_protocol_compliance_claimed must be false")
        for ff in FORBIDDEN_CYBER_FIELDS:
            if ff in str(cc).lower():
                errors.append(f"{sid}: forbidden cyber field keyword '{ff}' found in cyber_context")

    # Day coverage
    if not errors:
        day_counts = [0] * 7
        for sc in scenarios:
            t = sc.get("start_time_s", -1)
            for d, (lo, hi) in enumerate(DAY_BOUNDARIES):
                if lo <= t <= hi:
                    day_counts[d] += 1
                    break
        for d, cnt in enumerate(day_counts):
            if cnt == 0:
                errors.append(f"No scenarios cover Day {d+1} (seconds {DAY_BOUNDARIES[d][0]}-{DAY_BOUNDARIES[d][1]})")
            if cnt > 4:
                errors.append(f"Day {d+1} has {cnt} scenarios (max 4 allowed)")

    return errors, warnings, repaired, repairs


def run_validation() -> dict:
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(RAW_DIR.rglob("*.json"))
    print(f"\n  Found {len(json_files)} JSON file(s) under {RAW_DIR.name}")

    results = []
    accepted_bundles = []
    accepted_count = 0
    rejected_count = 0

    for jf in json_files:
        rel = jf.relative_to(RAW_DIR)
        print(f"\n  Validating: {rel}")

        try:
            with open(jf, encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            print(f"    FAIL: invalid JSON — {e}")
            results.append({
                "file": str(rel), "status": "REJECTED",
                "errors": [f"Invalid JSON: {e}"], "warnings": [], "repairs": [],
                "author_model": None, "scenario_count": 0, "class_counts": {},
            })
            rejected_count += 1
            continue

        errors, warnings, repaired, repairs = _validate_bundle(raw, jf)

        author = repaired.get("author_model", _infer_author_from_folder(jf))
        scenarios = repaired.get("scenarios", [])
        class_counts = {}
        family_counts = {}
        asset_counts = {}
        var_counts = {}
        for sc in scenarios:
            c = sc.get("scenario_class", "?")
            class_counts[c] = class_counts.get(c, 0) + 1
            family_counts[sc.get("scenario_family", "?")] = family_counts.get(sc.get("scenario_family", "?"), 0) + 1
            a = sc.get("target_asset_id", "?")
            asset_counts[a] = asset_counts.get(a, 0) + 1
            for eff in sc.get("physical_effects", []):
                v = eff.get("variable", "?")
                var_counts[v] = var_counts.get(v, 0) + 1

        day_coverage = {}
        for sc in scenarios:
            t = sc.get("start_time_s", -1)
            for d, (lo, hi) in enumerate(DAY_BOUNDARIES):
                if lo <= t <= hi:
                    day_coverage[f"day_{d+1}"] = day_coverage.get(f"day_{d+1}", 0) + 1
                    break

        if errors:
            status = "REJECTED"
            rejected_count += 1
            print(f"    REJECTED — {len(errors)} error(s), {len(warnings)} warning(s), {len(repairs)} repair(s)")
            for e in errors:
                print(f"      ERROR: {e}")
        else:
            status = "ACCEPTED"
            accepted_count += 1
            out_name = VALIDATED_NAMES.get(author, f"{author}_validated.json")
            out_path = VALIDATED_DIR / out_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(repaired, f, indent=2)
            accepted_bundles.append({"author": author, "path": str(out_path), "scenarios": len(scenarios)})
            print(f"    ACCEPTED -- {len(scenarios)} scenarios, {len(repairs)} repair(s) -> {out_name}")

        results.append({
            "file": str(rel), "status": status,
            "errors": errors, "warnings": warnings, "repairs": repairs,
            "author_model": author,
            "scenario_count": len(scenarios),
            "class_counts": class_counts,
            "family_counts": family_counts,
            "asset_counts": asset_counts,
            "variable_counts": var_counts,
            "day_coverage": day_coverage,
        })

    # Aggregate stats
    accepted_scenario_count = sum(r["scenario_count"] for r in results if r["status"] == "ACCEPTED")
    rejected_scenario_count = sum(r["scenario_count"] for r in results if r["status"] == "REJECTED")

    summary = {
        "timestamp": NOW,
        "files_scanned": len(json_files),
        "accepted_bundle_count": accepted_count,
        "rejected_bundle_count": rejected_count,
        "accepted_scenario_count": accepted_scenario_count,
        "rejected_scenario_count": rejected_scenario_count,
        "accepted_bundles": accepted_bundles,
        "per_file_results": results,
        "safety_status": "event_level_context_only_no_packet_data",
    }

    out_json = OUTPUTS_DIR / "zero_day_validation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_json.name}")

    _write_validation_report(summary, results)
    return summary


def _write_validation_report(summary: dict, results: list):
    lines = [
        "# Zero-Day Scenario Validation Report",
        f"_Generated: {NOW}_",
        "",
        "## Summary",
        f"- Files scanned: {summary['files_scanned']}",
        f"- Bundles accepted: {summary['accepted_bundle_count']}",
        f"- Bundles rejected: {summary['rejected_bundle_count']}",
        f"- Scenarios accepted: {summary['accepted_scenario_count']}",
        f"- Scenarios rejected: {summary['rejected_scenario_count']}",
        f"- Safety status: {summary['safety_status']}",
        "",
        "## Safe Repairs Applied",
        "The following types of safe, documented repairs were applied before validation:",
        "- Negative magnitudes with explicit direction → abs(magnitude) (direction encodes sign)",
        "- `compatibility_statement` → `project_compatibility_statement` (Gemini field alias)",
        "- `cyber_physical_type` → `scenario_class` (Gemini field alias)",
        "- `affected_asset` → `target_asset_id` (Gemini field alias)",
        "- Missing `scenario_name` → generated from scenario_id",
        "- Missing `target_component` → inferred from asset ID",
        "- `stale_value_hold` → `hold_stale` (Gemini effect type alias)",
        "- `bounded_clip` → `clip` (Gemini effect type alias)",
        "- `bounded_oscillation` → `smooth_oscillation` (Gemini effect type alias)",
        "",
        "## Per-Bundle Results",
        "",
    ]
    for r in results:
        icon = "ACCEPTED" if r["status"] == "ACCEPTED" else "REJECTED"
        lines += [
            f"### {r['file']} — {icon}",
            f"- Author model: {r['author_model']}",
            f"- Scenarios: {r['scenario_count']}",
        ]
        if r["class_counts"]:
            lines.append(f"- Class counts: {r['class_counts']}")
        if r["day_coverage"]:
            lines.append(f"- Day coverage: {r['day_coverage']}")
        if r["repairs"]:
            lines += ["- **Repairs applied:**"]
            for rep in r["repairs"]:
                lines.append(f"  - {rep}")
        if r["warnings"]:
            lines += ["- **Warnings:**"]
            for w in r["warnings"]:
                lines.append(f"  - WARNING: {w}")
        if r["errors"]:
            lines += ["- **Errors (REJECTION REASONS):**"]
            for e in r["errors"]:
                lines.append(f"  - ERROR: {e}")
        lines.append("")

    rp = REPORTS_DIR / "ZERO_DAY_SCENARIO_VALIDATION_REPORT.md"
    rp.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {rp.name}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PHASE 2 ZERO-DAY SCENARIO VALIDATION")
    print("=" * 60)
    summary = run_validation()
    print(f"\n  Accepted: {summary['accepted_bundle_count']} bundles, "
          f"{summary['accepted_scenario_count']} scenarios")
    print(f"  Rejected: {summary['rejected_bundle_count']} bundles")
