"""
Phase 2 Zero-Day Dataset Compiler.

Reads validated scenario bundles and applies physical effects to the clean
physical CSV, producing a zero-day-like attacked dataset.  Label columns
are added for frozen-model evaluation.

Inputs:
  configs/phase2_zero_day_config.json
  configs/variable_mapping_updated.json
  scenarios/scenario_bundles_validated/*.json
  data_updated/raw/physical_timeseries_clean_improved_7d.csv

Outputs:
  outputs/zero_day_physical_attacked.csv
  outputs/zero_day_context_windows.csv
  outputs/zero_day_scenario_manifest.csv
  reports/ZERO_DAY_DATASET_COMPILATION_REPORT.md
"""
import sys
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PHASE2_ROOT = Path(r"D:\updated_dataset\phase2_zero_day_eval")
CONFIG_DIR   = PHASE2_ROOT / "configs"
VALIDATED_DIR = PHASE2_ROOT / "scenarios" / "scenario_bundles_validated"
OUTPUTS_DIR  = PHASE2_ROOT / "outputs"
REPORTS_DIR  = PHASE2_ROOT / "reports"

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

PHYSICAL_BOUNDS = {
    "pv_p_kw":         (0.0,    100.0),
    "pv_q_kvar":       (-60.0,   60.0),
    "bess_p_kw":       (-50.0,   50.0),
    "bess_q_kvar":     (-50.0,   50.0),
    "bess_soc_percent":(5.0,     95.0),
    "pcc_v_a_pu":      (0.90,    1.10),
    "pcc_v_b_pu":      (0.90,    1.10),
    "pcc_v_c_pu":      (0.90,    1.10),
    "pcc_i_a_amp":     (0.0,   500.0),
    "pcc_i_b_amp":     (0.0,   500.0),
    "pcc_i_c_amp":     (0.0,   500.0),
    "pcc_p_kw":        (-250.0, 250.0),
    "pcc_q_kvar":      (-150.0, 150.0),
    "irradiance_pu":   (0.0,     1.2),
}

EFFECT_TYPE_ALIASES = {
    "stale_value_hold":   "hold_stale",
    "bounded_clip":       "clip",
    "bounded_oscillation":"smooth_oscillation",
}

ZERO_DAY_LABEL_COLS = [
    "zero_day_active_flag",
    "zero_day_scenario_id",
    "zero_day_scenario_family",
    "zero_day_scenario_class",
    "zero_day_author_model",
    "zero_day_label_anomaly",
    "zero_day_label_cyber_anomaly",
    "zero_day_label_physical_anomaly",
]


# ---------------------------------------------------------------------------
# Variable-mapping helpers
# ---------------------------------------------------------------------------

def _load_var_mapping(config_dir: Path) -> dict:
    vmap_path = config_dir / "variable_mapping_updated.json"
    with open(vmap_path) as f:
        raw = json.load(f)
    return raw  # canonical_name -> [col1, col2, ...]


def _resolve_column(canonical: str, var_mapping: dict, csv_columns: set) -> str | None:
    """Return the first CSV column that matches the canonical variable."""
    candidates = var_mapping.get(canonical, [canonical])
    for col in candidates:
        if col in csv_columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Effect application
# ---------------------------------------------------------------------------

def _normalize_effect_type(et: str) -> str:
    return EFFECT_TYPE_ALIASES.get(et, et)


def _apply_effect(vals: np.ndarray, times: np.ndarray, start_t: int,
                  effect: dict) -> np.ndarray:
    """Return a modified copy of vals with the effect applied.

    vals  : 1-D array of float, shape (n,)
    times : 1-D array of int   timestamps matching vals
    start_t: scenario start_time_s
    """
    etype     = _normalize_effect_type(effect.get("effect_type", "none"))
    magnitude = abs(float(effect.get("magnitude", 0.0)))  # always non-negative
    direction = effect.get("direction", "none")
    shape     = effect.get("shape", "step")
    ramp_s    = int(effect.get("ramp_seconds", 0))
    unit      = effect.get("unit", "")

    result = vals.astype(float).copy()

    if etype == "none" or magnitude == 0.0 and etype not in ("hold_stale", "clip"):
        return result

    if etype == "hold_stale":
        result[:] = result[0]
        return result

    if etype == "clip":
        # Cap at upper bound given by magnitude
        np.clip(result, None, magnitude, out=result)
        return result

    if etype == "smooth_oscillation":
        n = len(times)
        if n == 0:
            return result
        period = max(30.0, min(120.0, float(times[-1] - start_t + 1)))
        t_rel = (times - start_t).astype(float)
        oscillation = magnitude * np.sin(2.0 * math.pi * t_rel / period)
        result += oscillation
        return result

    # absolute_delta / relative_delta
    t_rel_arr = (times - start_t).astype(float)
    if shape == "ramp" and ramp_s > 0:
        ramp_factors = np.clip(t_rel_arr / float(ramp_s), 0.0, 1.0)
    else:
        ramp_factors = np.ones(len(times), dtype=float)

    sign = 1.0 if direction == "increase" else -1.0

    if etype == "absolute_delta":
        result += sign * magnitude * ramp_factors

    elif etype == "relative_delta":
        factor = (magnitude / 100.0) if unit == "%" else magnitude
        result *= (1.0 + sign * factor * ramp_factors)

    return result


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------

def _load_validated_bundles(validated_dir: Path) -> list[dict]:
    bundles = []
    for path in sorted(validated_dir.glob("*.json")):
        with open(path) as f:
            bundle = json.load(f)
        bundle["_source_file"] = path.name
        bundles.append(bundle)
    return bundles


# ---------------------------------------------------------------------------
# Main compilation
# ---------------------------------------------------------------------------

def compile_zero_day_dataset() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- load config & variable mapping ---
    with open(CONFIG_DIR / "phase2_zero_day_config.json") as f:
        cfg = json.load(f)

    var_mapping = _load_var_mapping(CONFIG_DIR)

    clean_csv = Path(cfg["clean_physical_csv"])
    if not clean_csv.exists():
        print(f"[ERROR] Clean CSV not found: {clean_csv}")
        sys.exit(1)

    print(f"[INFO] Loading clean CSV: {clean_csv}")
    df = pd.read_csv(clean_csv)
    print(f"[INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Verify time_s column
    if "time_s" not in df.columns:
        print("[ERROR] time_s column missing from clean CSV")
        sys.exit(1)

    # Add zero-day label columns (defaults)
    df["zero_day_active_flag"]          = 0
    df["zero_day_scenario_id"]          = ""
    df["zero_day_scenario_family"]      = ""
    df["zero_day_scenario_class"]       = ""
    df["zero_day_author_model"]         = ""
    df["zero_day_label_anomaly"]        = 0
    df["zero_day_label_cyber_anomaly"]  = 0
    df["zero_day_label_physical_anomaly"] = 0

    csv_cols = set(df.columns)
    time_arr = df["time_s"].values.astype(int)

    # --- load validated bundles ---
    bundles = _load_validated_bundles(VALIDATED_DIR)
    if not bundles:
        print(f"[WARN] No validated bundles found in {VALIDATED_DIR}")

    manifest_rows = []
    total_affected_rows = 0
    warnings_log = []

    for bundle in bundles:
        author = bundle.get("author_model", "unknown")
        print(f"[INFO] Processing bundle: {bundle.get('_source_file')} (author={author})")
        scenarios = bundle.get("scenarios", [])

        for scen in scenarios:
            sid     = scen.get("scenario_id", "")
            family  = scen.get("scenario_family", "")
            sclass  = scen.get("scenario_class", "")
            start_t = int(scen.get("start_time_s", 0))
            dur_s   = int(scen.get("duration_s", 0))
            end_t   = start_t + dur_s - 1

            labels  = scen.get("labels", {})
            la      = int(labels.get("label_anomaly", 0))
            lca     = int(labels.get("label_cyber_anomaly", 0))
            lpa     = int(labels.get("label_physical_anomaly", 0))

            # row mask for this scenario window
            mask = (time_arr >= start_t) & (time_arr <= end_t)
            n_rows = int(mask.sum())
            total_affected_rows += n_rows

            # Update label columns (OR logic for overlaps)
            existing_id = df.loc[mask, "zero_day_scenario_id"].values
            has_prior = existing_id != ""

            df.loc[mask, "zero_day_active_flag"] = 1
            df.loc[mask, "zero_day_label_anomaly"] = np.maximum(
                df.loc[mask, "zero_day_label_anomaly"].values, la)
            df.loc[mask, "zero_day_label_cyber_anomaly"] = np.maximum(
                df.loc[mask, "zero_day_label_cyber_anomaly"].values, lca)
            df.loc[mask, "zero_day_label_physical_anomaly"] = np.maximum(
                df.loc[mask, "zero_day_label_physical_anomaly"].values, lpa)

            # scenario_id / class / family / author: first-write wins, "MULTIPLE" on conflict
            new_id = np.where(has_prior, "MULTIPLE", sid)
            df.loc[mask, "zero_day_scenario_id"] = new_id
            new_fam = np.where(has_prior, df.loc[mask, "zero_day_scenario_family"].values, family)
            df.loc[mask, "zero_day_scenario_family"] = new_fam
            new_cls = np.where(has_prior, df.loc[mask, "zero_day_scenario_class"].values, sclass)
            df.loc[mask, "zero_day_scenario_class"] = new_cls
            new_auth = np.where(has_prior, df.loc[mask, "zero_day_author_model"].values, author)
            df.loc[mask, "zero_day_author_model"] = new_auth

            # Apply physical effects
            phys_effects = scen.get("physical_effects", [])
            for effect in phys_effects:
                canonical = effect.get("variable", "")
                col = _resolve_column(canonical, var_mapping, csv_cols)
                if col is None:
                    msg = f"[WARN] {sid}: variable '{canonical}' not found in CSV, skipping."
                    warnings_log.append(msg)
                    print(msg)
                    continue

                window_times = time_arr[mask]
                window_vals  = df.loc[mask, col].values.astype(float)

                modified = _apply_effect(window_vals, window_times, start_t, effect)

                # Clip to physical bounds (using canonical name)
                if canonical in PHYSICAL_BOUNDS:
                    lo, hi = PHYSICAL_BOUNDS[canonical]
                    np.clip(modified, lo, hi, out=modified)

                df.loc[mask, col] = modified

            manifest_rows.append({
                "scenario_id":             sid,
                "scenario_name":           scen.get("scenario_name", ""),
                "scenario_family":         family,
                "scenario_class":          sclass,
                "author_model":            author,
                "start_time_s":            start_t,
                "end_time_s":              end_t,
                "duration_s":              dur_s,
                "target_asset_id":         scen.get("target_asset_id", ""),
                "target_component":        scen.get("target_component", ""),
                "n_physical_effects":      len(phys_effects),
                "label_anomaly":           la,
                "label_cyber_anomaly":     lca,
                "label_physical_anomaly":  lpa,
                "n_rows_affected":         n_rows,
                "source_bundle":           bundle.get("_source_file", ""),
            })

    # --- save outputs ---
    out_csv = OUTPUTS_DIR / "zero_day_physical_attacked.csv"
    print(f"[INFO] Saving attacked CSV: {out_csv}")
    df.to_csv(out_csv, index=False)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUTS_DIR / "zero_day_scenario_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[INFO] Saved scenario manifest: {manifest_path} ({len(manifest_df)} rows)")

    # context_windows: same as manifest with window_row_start / window_row_end for index lookup
    ctx_df = manifest_df.copy()
    if not ctx_df.empty:
        ctx_df["window_row_start"] = ctx_df["start_time_s"]
        ctx_df["window_row_end"]   = ctx_df["end_time_s"]
    else:
        ctx_df["window_row_start"] = pd.Series(dtype=int)
        ctx_df["window_row_end"]   = pd.Series(dtype=int)
    ctx_path = OUTPUTS_DIR / "zero_day_context_windows.csv"
    ctx_df.to_csv(ctx_path, index=False)
    print(f"[INFO] Saved context windows: {ctx_path}")

    # --- summary statistics ---
    total_scenarios = len(manifest_rows)
    class_counts = manifest_df["scenario_class"].value_counts().to_dict() if total_scenarios > 0 else {}
    author_counts = manifest_df["author_model"].value_counts().to_dict() if total_scenarios > 0 else {}

    summary = {
        "timestamp":          NOW,
        "bundles_processed":  len(bundles),
        "total_scenarios":    total_scenarios,
        "class_counts":       class_counts,
        "author_counts":      author_counts,
        "total_affected_rows":total_affected_rows,
        "output_csv_rows":    len(df),
        "output_csv_cols":    len(df.columns),
        "warnings":           warnings_log,
        "outputs": {
            "attacked_csv":      str(out_csv),
            "manifest_csv":      str(manifest_path),
            "context_windows_csv": str(ctx_path),
        },
    }

    _write_report(summary, bundles, manifest_df, warnings_log)
    return summary


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(summary: dict, bundles: list, manifest_df: pd.DataFrame,
                  warnings: list) -> None:
    lines = [
        "# ZERO_DAY_DATASET_COMPILATION_REPORT",
        "",
        f"Generated: {summary['timestamp']}",
        "",
        "## Summary",
        "",
        f"- Validated bundles processed: **{summary['bundles_processed']}**",
        f"- Total scenarios compiled:    **{summary['total_scenarios']}**",
        f"- Total CSV rows affected:     **{summary['total_affected_rows']:,}**",
        f"- Output CSV rows:             **{summary['output_csv_rows']:,}**",
        f"- Output CSV columns:          **{summary['output_csv_cols']}**",
        "",
        "## Scenario Class Breakdown",
        "",
    ]
    for cls, cnt in sorted(summary["class_counts"].items()):
        lines.append(f"- {cls}: {cnt}")
    lines += [
        "",
        "## Author Model Breakdown",
        "",
    ]
    for auth, cnt in sorted(summary["author_counts"].items()):
        lines.append(f"- {auth}: {cnt}")

    lines += [
        "",
        "## Bundles Processed",
        "",
    ]
    for b in bundles:
        lines.append(f"- `{b.get('_source_file')}` (author={b.get('author_model')}, "
                     f"scenarios={len(b.get('scenarios', []))})")

    if not manifest_df.empty:
        lines += [
            "",
            "## Scenario Manifest (summary)",
            "",
            "| scenario_id | class | author | start_s | end_s | la | lca | lpa |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, row in manifest_df.iterrows():
            lines.append(
                f"| {row.scenario_id} | {row.scenario_class} | {row.author_model} | "
                f"{row.start_time_s} | {row.end_time_s} | "
                f"{row.label_anomaly} | {row.label_cyber_anomaly} | {row.label_physical_anomaly} |"
            )

    lines += [
        "",
        "## Physical Effects Applied",
        "",
        "Effects are applied in bundle-then-scenario order.  When windows overlap, "
        "label columns are OR-merged; physical effects accumulate additively.  "
        "All values are clipped to schema-defined physical bounds after each effect.",
        "",
        "## Outputs",
        "",
        f"- `{summary['outputs']['attacked_csv']}`",
        f"- `{summary['outputs']['manifest_csv']}`",
        f"- `{summary['outputs']['context_windows_csv']}`",
    ]

    if warnings:
        lines += ["", "## Warnings", ""]
        for w in warnings:
            lines.append(f"- {w}")

    lines += [
        "",
        "## Constraints Honoured",
        "",
        "- Source: clean physical CSV only; no Phase 1 attacked CSV used",
        "- No model retraining or threshold recalibration",
        "- Cyber context is event-level metadata only; no packet-level fields",
        "- All physical effects bounded within schema-defined physical bounds",
        "- temperature_c column not modified",
        "",
        "---",
        "*Phase 2 Zero-Day Evaluation — Compilation complete*",
    ]

    report_path = REPORTS_DIR / "ZERO_DAY_DATASET_COMPILATION_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote compilation report: {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = compile_zero_day_dataset()
    print("\n=== COMPILATION SUMMARY ===")
    print(f"Bundles processed : {summary['bundles_processed']}")
    print(f"Scenarios compiled: {summary['total_scenarios']}")
    print(f"Rows affected     : {summary['total_affected_rows']:,}")
    print(f"Warnings          : {len(summary['warnings'])}")
    for w in summary["warnings"]:
        print(f"  {w}")
    print("===========================")
