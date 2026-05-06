"""
Phase 3 Step 2: Build grounded evidence packets for each smoke detection.
All physical/cyber/context evidence comes from real Phase 2 files.
"""

import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "phase3_explanation_config.json"
OUT_DIR = ROOT / "outputs"
RPT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)
RPT_DIR.mkdir(exist_ok=True)

with open(CFG_PATH) as f:
    cfg = json.load(f)

P = lambda k: Path(cfg[k])

print("[INFO] Loading Phase 2 data files ...")
detections_df = pd.read_csv(cfg["smoke_detection_file"])
windows_df    = pd.read_parquet(P("zero_day_windows"))
manifest_df   = pd.read_csv(P("zero_day_scenario_manifest"))
ctx_df        = pd.read_csv(P("zero_day_context_windows"))
event_log_df  = pd.read_csv(P("zero_day_cyber_event_log"))

print(f"  Detections : {len(detections_df)}")
print(f"  Windows    : {len(windows_df)}")
print(f"  Manifest   : {len(manifest_df)}")
print(f"  Context    : {len(ctx_df)}")
print(f"  Event log  : {len(event_log_df)}")

# Load evidence packets as lookup
ev_packets = {}
ev_path = P("zero_day_phase3_evidence_packets")
if ev_path.exists():
    with open(ev_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                sid = obj.get("scenario_id")
                if sid:
                    ev_packets[sid] = obj

print(f"  Evidence packets loaded: {len(ev_packets)}")

# Load attacked CSV (for physical signal stats)
attacked_path = P("zero_day_physical_attacked")
print(f"[INFO] Loading attacked CSV (large) ...")
attacked_df = pd.read_csv(attacked_path, low_memory=False)
print(f"  Attacked CSV: {attacked_df.shape}")

# Load aligned 1s table
aligned_path = P("zero_day_aligned_1s")
print(f"[INFO] Loading aligned 1s table ...")
aligned_df = pd.read_csv(aligned_path, low_memory=False)
print(f"  Aligned 1s: {aligned_df.shape}")

# ── Physical signal columns ───────────────────────────────────────────────────
PHYS_SIGNALS = [
    "pv_p_kw", "pv_q_kvar",
    "bess_p_kw", "bess_q_kvar", "bess_soc_percent",
    "pcc_v_a_pu", "pcc_v_b_pu", "pcc_v_c_pu",
    "pcc_i_a_amp", "pcc_i_b_amp", "pcc_i_c_amp",
    "pcc_p_kw", "pcc_q_kvar",
    "irradiance_pu", "temperature_c",
]
phys_cols_available = [c for c in PHYS_SIGNALS if c in attacked_df.columns]

# Cyber flag columns available in aligned table
CYBER_FLAGS = [
    "cyber_anomaly_active", "physical_effect_active",
    "command_created_flag", "command_sent_flag", "command_recv_flag",
    "command_accept_flag", "command_apply_flag", "command_response_flag",
    "blocked_flag", "replay_flag", "mismatch_flag",
    "stale_command_flag", "timeout_flag",
]
cyber_flag_cols = [c for c in CYBER_FLAGS if c in aligned_df.columns]

# Time column in aligned table
time_col_aligned = None
for c in ["time_s", "timestamp_s", "second"]:
    if c in aligned_df.columns:
        time_col_aligned = c
        break

# Cyber state column
cyber_state_col = None
for c in ["cyber_state", "cyber_context_state"]:
    if c in aligned_df.columns:
        cyber_state_col = c
        break

lifecycle_col = None
for c in ["cyber_lifecycle_stage", "lifecycle_stage"]:
    if c in aligned_df.columns:
        lifecycle_col = c
        break

# time_s in attacked_df
time_col_attacked = None
for c in ["time_s", "timestamp_s"]:
    if c in attacked_df.columns:
        time_col_attacked = c
        break

print(f"  Physical cols: {len(phys_cols_available)}")
print(f"  Cyber flag cols: {len(cyber_flag_cols)}")

# ── Manifest helpers ─────────────────────────────────────────────────────────
def get_manifest_row(scenario_id):
    sub = manifest_df[manifest_df["scenario_id"] == scenario_id]
    if len(sub) == 0:
        return {}
    return sub.iloc[0].to_dict()

def get_event_rows(scenario_id):
    sub = event_log_df[event_log_df["scenario_id"] == scenario_id] if "scenario_id" in event_log_df.columns else pd.DataFrame()
    return sub.to_dict("records")

def coerce(v):
    """Make value JSON-serializable."""
    if isinstance(v, (np.integer,)):   return int(v)
    if isinstance(v, (np.floating,)):  return float(v) if not math.isnan(v) else None
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, (np.bool_,)):     return bool(v)
    if pd.isna(v) if not isinstance(v, (list, dict, bool)) else False: return None
    return v

# ── Core evidence builder ─────────────────────────────────────────────────────
def build_evidence_packet(det_row):
    det = det_row.to_dict()
    wstart = int(det.get("window_start_s", 0))
    wend   = int(det.get("window_end_s", wstart + 59))
    sid    = str(det.get("scenario_id", "none"))
    wid    = det.get("window_id", f"w_{wstart}")

    # ── A. Detection block ────────────────────────────────────────────────────
    detection = {
        "detection_id":       det.get("detection_id"),
        "model_name":         det.get("model_name"),
        "window_id":          wid,
        "window_start_s":     wstart,
        "window_end_s":       wend,
        "anomaly_score":      coerce(det.get("anomaly_score")),
        "threshold":          coerce(det.get("threshold")),
        "predicted_label":    int(det.get("predicted_label", 0)),
        "smoke_detection_only": bool(det.get("smoke_detection_only", True)),
    }

    # ── B. Scenario block ─────────────────────────────────────────────────────
    mrow = get_manifest_row(sid)
    scenario = {
        "scenario_id":      sid,
        "scenario_class":   det.get("scenario_class", mrow.get("scenario_class")),
        "scenario_family":  det.get("scenario_family", mrow.get("scenario_family")),
        "author_model":     coerce(mrow.get("author_model", mrow.get("bundle_source"))),
        "target_asset_id":  coerce(mrow.get("target_asset_id")),
        "target_component": coerce(mrow.get("target_component")),
        "labels": {
            "label_anomaly":          coerce(mrow.get("label_anomaly")),
            "label_cyber_anomaly":    coerce(mrow.get("label_cyber_anomaly")),
            "label_physical_anomaly": coerce(mrow.get("label_physical_anomaly")),
        }
    }

    # ── C. Physical evidence ──────────────────────────────────────────────────
    phys_ev = build_physical_evidence(sid, wstart, wend, mrow)

    # ── D. Cyber evidence ─────────────────────────────────────────────────────
    cyber_ev = build_cyber_evidence(sid, wstart, wend, mrow)

    # ── E. Expected vs actual ─────────────────────────────────────────────────
    ev_pkt = ev_packets.get(sid, {})
    e_vs_a = build_expected_vs_actual(sid, wstart, wend, mrow, phys_ev, cyber_ev, ev_pkt)

    # ── F. Timing alignment ───────────────────────────────────────────────────
    timing = build_timing(sid, wstart, wend, mrow, cyber_ev, ev_pkt)

    # ── G. Guardrails ─────────────────────────────────────────────────────────
    guardrails = {
        "no_packet_level_evidence":   True,
        "no_real_field_telemetry":    True,
        "no_external_attacker_identity": True,
        "evidence_only_instruction":  (
            "Base your explanation solely on the provided evidence fields. "
            "Do not claim packet-level protocol details, real field telemetry, "
            "or external attacker identity. Use explanation_type=insufficient_evidence "
            "if evidence is missing or contradictory."
        )
    }

    return {
        "detection":         detection,
        "scenario":          scenario,
        "physical_evidence": phys_ev,
        "cyber_evidence":    cyber_ev,
        "expected_vs_actual": e_vs_a,
        "timing_alignment":  timing,
        "guardrails":        guardrails,
    }

def build_physical_evidence(sid, wstart, wend, mrow):
    phys_ev = {
        "top_signals":                       [],
        "signal_direction_summary":          {},
        "affected_variables_from_scenario":  [],
        "physical_effect_active_fraction_in_window": None,
        "physical_evidence_available":       False,
    }
    if time_col_attacked is None:
        return phys_ev

    window_phys = attacked_df[
        (attacked_df[time_col_attacked] >= wstart) &
        (attacked_df[time_col_attacked] <= wend)
    ]
    if window_phys.empty:
        return phys_ev

    phys_ev["physical_evidence_available"] = True

    # Compute per-signal stats
    signal_stats = {}
    for col in phys_cols_available:
        vals = window_phys[col].dropna()
        if len(vals) < 2:
            continue
        signal_stats[col] = {
            "mean":  round(float(vals.mean()), 4),
            "std":   round(float(vals.std()), 4),
            "min":   round(float(vals.min()), 4),
            "max":   round(float(vals.max()), 4),
            "delta": round(float(vals.iloc[-1] - vals.iloc[0]), 4),
        }

    # Top signals by std deviation (most variable)
    sorted_by_std = sorted(signal_stats.items(), key=lambda x: abs(x[1]["std"]), reverse=True)
    phys_ev["top_signals"] = [
        {"signal": k, "mean": v["mean"], "std": v["std"],
         "min": v["min"], "max": v["max"], "delta": v["delta"]}
        for k, v in sorted_by_std[:6]
    ]

    # Direction summary
    for k, v in signal_stats.items():
        d = v["delta"]
        if abs(d) < 0.01:
            phys_ev["signal_direction_summary"][k] = "stable"
        elif d > 0:
            phys_ev["signal_direction_summary"][k] = "increase"
        else:
            phys_ev["signal_direction_summary"][k] = "decrease"

    # Affected variables from scenario manifest
    for col_name in ["affected_variable", "affected_variables", "effect_variable"]:
        if col_name in mrow and mrow[col_name]:
            v = mrow[col_name]
            if isinstance(v, str):
                phys_ev["affected_variables_from_scenario"] = [v]
            break

    # physical_effect_active fraction from aligned table
    if time_col_aligned and "physical_effect_active" in aligned_df.columns:
        window_aligned = aligned_df[
            (aligned_df[time_col_aligned] >= wstart) &
            (aligned_df[time_col_aligned] <= wend)
        ]
        if len(window_aligned) > 0:
            frac = float(window_aligned["physical_effect_active"].mean())
            phys_ev["physical_effect_active_fraction_in_window"] = round(frac, 4)

    return phys_ev

def build_cyber_evidence(sid, wstart, wend, mrow):
    cyber_ev = {
        "cyber_states_seen":        [],
        "lifecycle_stages_seen":    [],
        "command_flags_seen":       {},
        "blocked_flag_seen":        False,
        "replay_flag_seen":         False,
        "mismatch_flag_seen":       False,
        "stale_command_flag_seen":  False,
        "timeout_flag_seen":        False,
        "cyber_anomaly_active_fraction_in_window": None,
        "cyber_evidence_available": False,
        "event_level_only":         True,
        "packet_level_protocol_compliance_claimed": False,
    }
    if time_col_aligned is None:
        return cyber_ev

    window_aligned = aligned_df[
        (aligned_df[time_col_aligned] >= wstart) &
        (aligned_df[time_col_aligned] <= wend)
    ]
    if window_aligned.empty:
        return cyber_ev

    cyber_ev["cyber_evidence_available"] = True

    if cyber_state_col and cyber_state_col in window_aligned.columns:
        cyber_ev["cyber_states_seen"] = sorted(window_aligned[cyber_state_col].dropna().unique().tolist())

    if lifecycle_col and lifecycle_col in window_aligned.columns:
        cyber_ev["lifecycle_stages_seen"] = sorted(window_aligned[lifecycle_col].dropna().unique().tolist())

    for flag in cyber_flag_cols:
        vals = window_aligned[flag]
        fraction = float(vals.mean())
        cyber_ev["command_flags_seen"][flag] = {
            "fraction_active": round(fraction, 4),
            "any_active": bool((vals > 0).any())
        }

    for flag_name, ev_key in [
        ("blocked_flag",       "blocked_flag_seen"),
        ("replay_flag",        "replay_flag_seen"),
        ("mismatch_flag",      "mismatch_flag_seen"),
        ("stale_command_flag", "stale_command_flag_seen"),
        ("timeout_flag",       "timeout_flag_seen"),
    ]:
        if flag_name in window_aligned.columns:
            cyber_ev[ev_key] = bool((window_aligned[flag_name] > 0).any())

    if "cyber_anomaly_active" in window_aligned.columns:
        cyber_ev["cyber_anomaly_active_fraction_in_window"] = round(
            float(window_aligned["cyber_anomaly_active"].mean()), 4
        )

    # Event log rows for this scenario
    ev_rows = get_event_rows(sid)
    cyber_ev["scenario_event_log_rows"] = len(ev_rows)
    if ev_rows:
        cyber_ev["event_log_lifecycle_stages"] = list({r.get("lifecycle_stage") for r in ev_rows if r.get("lifecycle_stage")})

    return cyber_ev

def build_expected_vs_actual(sid, wstart, wend, mrow, phys_ev, cyber_ev, ev_pkt):
    e_vs_a = {
        "expected_behavior_summary":       ev_pkt.get("expected_behavior", ""),
        "observed_behavior_summary":       "",
        "expected_vs_observed_difference": "",
        "affected_variables_from_scenario": phys_ev.get("affected_variables_from_scenario", []),
        "top_physical_signals_from_window": [s["signal"] for s in phys_ev.get("top_signals", [])[:4]],
        "cyber_flags_observed":            [k for k, v in cyber_ev.get("command_flags_seen", {}).items() if v.get("any_active")],
        "timing_relationship_observed":    "",
    }

    scls = str(mrow.get("scenario_class", "unknown"))
    phy_frac = phys_ev.get("physical_effect_active_fraction_in_window") or 0.0
    cyb_frac = cyber_ev.get("cyber_anomaly_active_fraction_in_window") or 0.0

    obs_parts = []
    if phy_frac > 0.1:
        obs_parts.append(f"Physical effect active {phy_frac*100:.0f}% of window.")
    if cyb_frac > 0.1:
        obs_parts.append(f"Cyber anomaly active {cyb_frac*100:.0f}% of window.")
    if phys_ev.get("top_signals"):
        top = phys_ev["top_signals"][0]
        obs_parts.append(
            f"Most variable physical signal: {top['signal']} "
            f"(mean={top['mean']}, std={top['std']}, delta={top['delta']})."
        )
    e_vs_a["observed_behavior_summary"] = " ".join(obs_parts) if obs_parts else "No significant deviation observed."

    if scls == "normal":
        e_vs_a["expected_vs_observed_difference"] = "Expected normal — no significant deviation observed."
        e_vs_a["timing_relationship_observed"]    = "No anomaly. Normal monitoring."
    elif scls == "physical_only":
        e_vs_a["expected_vs_observed_difference"] = "Expected physical change with no cyber cause."
        e_vs_a["timing_relationship_observed"]    = "Physical effect active without preceding cyber event."
    elif scls == "cyber_only":
        e_vs_a["expected_vs_observed_difference"] = "Expected cyber anomaly with no physical signal change."
        e_vs_a["timing_relationship_observed"]    = "Cyber flags active but no physical effect."
    elif scls == "cyber_physical":
        e_vs_a["expected_vs_observed_difference"] = "Expected cyber event followed by physical signal change."
        e_vs_a["timing_relationship_observed"]    = "Cyber event preceded or coincided with physical effect."

    return e_vs_a

def build_timing(sid, wstart, wend, mrow, cyber_ev, ev_pkt):
    timing = {
        "detection_overlaps_scenario":    False,
        "detection_overlap_seconds":      0,
        "command_apply_time_s":           None,
        "physical_effect_start_time_s":   None,
        "cyber_before_physical":          None,
        "timing_alignment_status":        "unknown",
    }

    scls = str(mrow.get("scenario_class", "unknown"))

    # Scenario time range from manifest
    sc_start = coerce(mrow.get("start_time_s"))
    sc_end   = coerce(mrow.get("end_time_s"))
    if sc_start is not None and sc_end is not None:
        overlap_start = max(wstart, int(sc_start))
        overlap_end   = min(wend, int(sc_end))
        overlap = max(0, overlap_end - overlap_start + 1)
        timing["detection_overlaps_scenario"]  = overlap > 0
        timing["detection_overlap_seconds"]    = overlap

    # Pull timing from evidence packet
    time_window = ev_pkt.get("time_window", {})
    if time_window:
        timing["command_apply_time_s"]        = time_window.get("command_apply_time_s")
        timing["physical_effect_start_time_s"] = time_window.get("effect_start_time_s")

    # Cyber before physical?
    cap = timing["command_apply_time_s"]
    pep = timing["physical_effect_start_time_s"]
    if cap is not None and pep is not None:
        timing["cyber_before_physical"] = int(cap) <= int(pep)

    # Alignment status
    if scls == "normal":
        timing["timing_alignment_status"] = "no_anomaly"
    elif scls == "physical_only":
        timing["timing_alignment_status"] = "physical_only_no_cyber_event"
    elif scls == "cyber_only":
        timing["timing_alignment_status"] = "cyber_only_no_physical_effect"
    elif scls == "cyber_physical":
        if timing["cyber_before_physical"] is True:
            timing["timing_alignment_status"] = "cyber_before_physical_confirmed"
        elif timing["cyber_before_physical"] is False:
            timing["timing_alignment_status"] = "cyber_physical_simultaneous"
        else:
            timing["timing_alignment_status"] = "cyber_physical_timing_unknown"

    return timing

# ── Process all detections ────────────────────────────────────────────────────
print(f"[INFO] Building evidence packets for {len(detections_df)} detections ...")
packets = []
table_rows = []

for idx, row in detections_df.iterrows():
    try:
        pkt = build_evidence_packet(row)
        packets.append(pkt)
        # Flat table row
        table_rows.append({
            "detection_id":         pkt["detection"]["detection_id"],
            "model_name":           pkt["detection"]["model_name"],
            "window_id":            pkt["detection"]["window_id"],
            "window_start_s":       pkt["detection"]["window_start_s"],
            "window_end_s":         pkt["detection"]["window_end_s"],
            "scenario_id":          pkt["scenario"]["scenario_id"],
            "scenario_class":       pkt["scenario"]["scenario_class"],
            "scenario_family":      pkt["scenario"]["scenario_family"],
            "author_model":         pkt["scenario"]["author_model"],
            "target_asset_id":      pkt["scenario"]["target_asset_id"],
            "predicted_label":      pkt["detection"]["predicted_label"],
            "anomaly_score":        pkt["detection"]["anomaly_score"],
            "smoke_detection_only": pkt["detection"]["smoke_detection_only"],
            "physical_evidence_available":  pkt["physical_evidence"]["physical_evidence_available"],
            "cyber_evidence_available":     pkt["cyber_evidence"]["cyber_evidence_available"],
            "physical_effect_active_frac":  pkt["physical_evidence"]["physical_effect_active_fraction_in_window"],
            "cyber_anomaly_active_frac":    pkt["cyber_evidence"]["cyber_anomaly_active_fraction_in_window"],
            "detection_overlaps_scenario":  pkt["timing_alignment"]["detection_overlaps_scenario"],
            "overlap_seconds":              pkt["timing_alignment"]["detection_overlap_seconds"],
            "timing_alignment_status":      pkt["timing_alignment"]["timing_alignment_status"],
        })
    except Exception as e:
        print(f"  [WARN] Failed for detection {row.get('detection_id')}: {e}")

# Save JSONL
jsonl_path = OUT_DIR / "explanation_inputs.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for pkt in packets:
        f.write(json.dumps(pkt, default=str) + "\n")
print(f"[INFO] explanation_inputs.jsonl saved: {jsonl_path} ({len(packets)} packets)")

# Save table CSV
table_df = pd.DataFrame(table_rows)
table_path = OUT_DIR / "explanation_input_table.csv"
table_df.to_csv(table_path, index=False)
print(f"[INFO] explanation_input_table.csv saved: {table_path}")

# ── Report ─────────────────────────────────────────────────────────────────────
class_dist = table_df["scenario_class"].value_counts()
phys_ok    = table_df["physical_evidence_available"].sum()
cyber_ok   = table_df["cyber_evidence_available"].sum()
overlap_ok = table_df["detection_overlaps_scenario"].sum()

report_lines = [
    "# EXPLANATION INPUT BUILD REPORT",
    "",
    f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "## Summary",
    "",
    f"| Item | Value |",
    f"|---|---|",
    f"| Detection packets built | {len(packets)} |",
    f"| Physical evidence available | {phys_ok} |",
    f"| Cyber evidence available | {cyber_ok} |",
    f"| Detections overlapping scenario window | {overlap_ok} |",
    "",
    "## Class Distribution",
    "",
    "| scenario_class | count |",
    "|---|---|",
]
for cls, cnt in class_dist.items():
    report_lines.append(f"| {cls} | {cnt} |")

report_lines += [
    "",
    "## Evidence Sources",
    "",
    "All physical/cyber/context evidence was pulled from real Phase 2 files:",
    "",
    "- Physical signals: `zero_day_physical_attacked.csv`",
    "- Cyber flags: `zero_day_cyber_physical_aligned_1s.csv`",
    "- Event log: `zero_day_cyber_event_log.csv`",
    "- Scenario metadata: `zero_day_scenario_manifest.csv`",
    "- Evidence packets: `zero_day_phase3_evidence_packets.jsonl`",
    "",
    "Only `anomaly_score`, `threshold`, `predicted_label`, `model_name` are synthetic smoke values.",
    ""
]

rpt_path = RPT_DIR / "EXPLANATION_INPUT_BUILD_REPORT.md"
with open(rpt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"[INFO] Report saved: {rpt_path}")
print(f"[DONE] {len(packets)} explanation inputs built.")
