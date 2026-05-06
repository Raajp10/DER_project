"""
DER Demo UI - Data Preparation Script
Converts large Phase 2/3 outputs into lightweight browser JSON for the demo.
Outputs go to public/data/ for Vite to serve.
Also writes fallback copies so the demo never opens blank.
"""

import json
import math
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUB_DATA = ROOT / "public" / "data"
PUB_DATA.mkdir(parents=True, exist_ok=True)

PHASE2  = Path("D:/updated_dataset/phase2_zero_day_eval/outputs")
PHASE3  = Path("D:/updated_dataset/phase3_explanations")

PHYSICAL_CSV  = PHASE2 / "zero_day_physical_attacked.csv"
ALIGNED_CSV   = PHASE2 / "zero_day_cyber_physical_aligned_1s.csv"
MANIFEST_CSV  = PHASE2 / "zero_day_scenario_manifest.csv"
SMOKE_DET_CSV = PHASE3 / "inputs" / "smoke_model_detections.csv"
EXP_JSONL     = PHASE3 / "outputs" / "explanation_inputs.jsonl"
PARSED_CSV    = PHASE3 / "outputs" / "llm_explanations_parsed.csv"
MATRIX_CSV    = PHASE3 / "outputs" / "explanation_evidence_matrix.csv"
SUMMARY_JSON  = PHASE3 / "outputs" / "explanation_score_summary.json"

BUFFER_S = 120
PHYS_SIGNALS = [
    "pv_p_kw","pv_q_kvar","bess_p_kw","bess_q_kvar","bess_soc_percent",
    "pcc_v_a_pu","pcc_v_b_pu","pcc_v_c_pu","pcc_i_a_amp","pcc_p_kw",
    "pcc_q_kvar","irradiance_pu","temperature_c",
]
CYBER_FLAGS = [
    "cyber_anomaly_active","physical_effect_active","cyber_state",
    "cyber_lifecycle_stage","command_created_flag","command_sent_flag",
    "command_recv_flag","command_accept_flag","command_apply_flag",
    "command_response_flag","blocked_flag","replay_flag","mismatch_flag",
    "stale_command_flag","timeout_flag",
]

def c(v):
    if isinstance(v, (float, np.floating)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)
    if isinstance(v, (int, np.integer)): return int(v)
    if isinstance(v, (bool, np.bool_)):  return bool(v)
    try:
        if pd.isna(v): return None
    except Exception: pass
    return v

def slist(v):
    if isinstance(v, list): return v
    if isinstance(v, str):
        try: return json.loads(v)
        except: return [v] if v.strip() else []
    return []

def load_ev_packets():
    d = {}
    if EXP_JSONL.exists():
        with open(EXP_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    did = obj.get("detection", {}).get("detection_id")
                    if did: d[did] = obj
    return d

def build_cyber_timeline(scenario_class, ev_pkt):
    ce = ev_pkt.get("cyber_evidence", {})
    ta = ev_pkt.get("timing_alignment", {})
    sc_fam = ev_pkt.get("scenario", {}).get("scenario_family", "")

    if scenario_class == "cyber_physical":
        stages = [
            {"key": "command_created",           "label": "Created",           "num": 1, "anomaly": False},
            {"key": "command_sent",               "label": "Sent",              "num": 2, "anomaly": False},
            {"key": "command_received",           "label": "Received",          "num": 3, "anomaly": False},
            {"key": "command_accepted",           "label": "Accepted",          "num": 4, "anomaly": False},
            {"key": "command_applied",            "label": "Applied",           "num": 5, "anomaly": False},
            {"key": "physical_response_observed", "label": "Physical Response", "num": 6, "anomaly": True},
            {"key": "status_report",              "label": "Status",            "num": 7, "anomaly": False},
        ]
        if ce.get("timeout_flag_seen"):
            stages[4]["anomaly"] = True
            stages[4]["label"] = "Applied (delayed)"
    elif scenario_class == "cyber_only":
        if ce.get("blocked_flag_seen"):
            stages = [
                {"key": "command_created", "label": "Created",  "num": 1, "anomaly": False},
                {"key": "command_sent",    "label": "Sent",     "num": 2, "anomaly": False},
                {"key": "blocked",         "label": "Blocked",  "num": 3, "anomaly": True},
                {"key": "alert",           "label": "Security Alert", "num": 4, "anomaly": True},
                {"key": "status_report",   "label": "Status",   "num": 5, "anomaly": False},
            ]
        elif ce.get("timeout_flag_seen"):
            stages = [
                {"key": "command_created", "label": "Created",  "num": 1, "anomaly": False},
                {"key": "command_sent",    "label": "Sent",     "num": 2, "anomaly": False},
                {"key": "timeout",         "label": "Timeout",  "num": 3, "anomaly": True},
                {"key": "stale",           "label": "Stale/Delayed", "num": 4, "anomaly": True},
                {"key": "status_report",   "label": "Status",   "num": 5, "anomaly": False},
            ]
        elif ce.get("mismatch_flag_seen"):
            stages = [
                {"key": "command_created", "label": "Created",   "num": 1, "anomaly": False},
                {"key": "command_sent",    "label": "Sent",      "num": 2, "anomaly": False},
                {"key": "mismatch",        "label": "Mismatch",  "num": 3, "anomaly": True},
                {"key": "alert",           "label": "FDI Alert", "num": 4, "anomaly": True},
                {"key": "status_report",   "label": "Status",    "num": 5, "anomaly": False},
            ]
        else:
            stages = [
                {"key": "command_created", "label": "Created",       "num": 1, "anomaly": False},
                {"key": "command_sent",    "label": "Sent",          "num": 2, "anomaly": False},
                {"key": "anomaly",         "label": "Cyber Anomaly", "num": 3, "anomaly": True},
                {"key": "status_report",   "label": "Status",        "num": 4, "anomaly": False},
            ]
    elif scenario_class == "physical_only":
        stages = [
            {"key": "monitoring_pre",             "label": "Pre-event Monitoring", "num": 1, "anomaly": False},
            {"key": "physical_event_observed",    "label": "Physical Event",       "num": 2, "anomaly": True},
            {"key": "monitoring_post",            "label": "Post-event Monitoring","num": 3, "anomaly": False},
        ]
    else:  # normal
        stages = [
            {"key": "normal_monitoring",   "label": "Normal Monitoring", "num": 1, "anomaly": False},
            {"key": "monitoring_complete", "label": "Normal Status",     "num": 2, "anomaly": False},
        ]

    if "command_delay" in sc_fam or ce.get("timeout_flag_seen"):
        desc = "Delayed command response detected at the control/communications layer."
    elif ce.get("blocked_flag_seen"):
        desc = "Command suppression detected — command was blocked before execution."
    elif ce.get("mismatch_flag_seen"):
        desc = "Command/measurement mismatch detected — possible false data injection."
    elif ce.get("replay_flag_seen"):
        desc = "Stale or replay-like event detected."
    elif scenario_class == "physical_only":
        desc = "Physical signal deviated without any associated cyber event."
    elif scenario_class == "cyber_physical":
        desc = "Cyber event preceded and likely caused the physical signal change."
    elif scenario_class == "normal":
        desc = "Normal DER operation — no anomalous events detected."
    else:
        desc = ""

    return {
        "stages": stages,
        "anomaly_description": desc,
        "cyber_before_physical": ta.get("cyber_before_physical"),
        "timing_alignment_status": ta.get("timing_alignment_status", "unknown"),
    }

def extract_ts(phy_df, aln_df, event_start, event_end, time_col, phys_cols, cyber_flag_cols, aln_time_col):
    buf_start = max(0, int(event_start) - BUFFER_S)
    buf_end   = int(event_end) + BUFFER_S

    pw = phy_df[(phy_df[time_col] >= buf_start) & (phy_df[time_col] <= buf_end)].copy()

    # Downsample to max 600 points
    if len(pw) > 600:
        step = max(1, len(pw) // 600)
        pw = pw.iloc[::step]

    t_abs = [int(x) for x in pw[time_col].tolist()]
    t_rel = [int(x) - buf_start for x in t_abs]

    ts = {
        "time_absolute": t_abs,
        "time_relative": t_rel,
        "attack_start_relative": int(event_start) - buf_start,
        "attack_end_relative":   int(event_end) - buf_start,
        "window_start_relative": int(event_start) - buf_start,
        "window_end_relative":   int(event_end) - buf_start,
        "buffer_start_s": buf_start,
        "n_points": len(t_rel),
    }

    for col in phys_cols:
        if col in pw.columns:
            ts[col] = [c(v) for v in pw[col].tolist()]

    # Cyber alignment overlay (aggregated, every 5s)
    if aln_time_col and len(aln_df) > 0:
        aw = aln_df[(aln_df[aln_time_col] >= buf_start) & (aln_df[aln_time_col] <= buf_end)].copy()
        if len(aw) > 300:
            aw = aw.iloc[::max(1, len(aw)//300)]
        for col in ["cyber_anomaly_active", "physical_effect_active"]:
            if col in aw.columns:
                ts[f"overlay_{col}"] = [c(v) for v in aw[col].tolist()]
        ts["overlay_time_relative"] = [int(x) - buf_start for x in aw[aln_time_col].tolist()] if aln_time_col in aw.columns else []

    return ts

def build_record(det, ev_pkt, parsed, matrix, manifest):
    sc         = ev_pkt.get("scenario", {})
    pe         = ev_pkt.get("physical_evidence", {})
    ce         = ev_pkt.get("cyber_evidence", {})
    ta         = ev_pkt.get("timing_alignment", {})
    eva        = ev_pkt.get("expected_vs_actual", {})

    scls       = str(det.get("scenario_class", sc.get("scenario_class", "unknown")))
    w_start    = int(det.get("window_start_s", 0))
    w_end      = int(det.get("window_end_s", w_start + 59))

    ev_start = c(manifest.get("start_time_s")) if manifest else None
    ev_end   = c(manifest.get("end_time_s")) if manifest else None
    if ev_start is None: ev_start = w_start
    if ev_end is None:   ev_end   = w_end

    exp = {}
    if parsed:
        exp = {
            "explanation_type":    str(parsed.get("explanation_type", "")),
            "confidence":          str(parsed.get("confidence", "")),
            "primary_asset":       str(parsed.get("primary_asset", "")),
            "primary_physical_signals": slist(parsed.get("primary_physical_signals", "[]")),
            "primary_cyber_evidence":   slist(parsed.get("primary_cyber_evidence", "[]")),
            "expected_vs_observed_summary": str(parsed.get("expected_vs_observed_summary", "")),
            "timing_summary":      str(parsed.get("timing_summary", "")),
            "operator_summary":    str(parsed.get("operator_summary", "")),
            "recommended_operator_checks": slist(parsed.get("recommended_operator_checks", "[]")),
            "human_explanation":   str(parsed.get("human_explanation", "")),
            "evidence_used":       slist(parsed.get("evidence_used", "[]")),
        }

    ev_mat = {}
    if matrix:
        for col in ["scenario_class_match","asset_match","physical_signal_match",
                    "cyber_state_match","timing_match","expected_vs_observed_match",
                    "unsupported_claim_flag","packet_claim_flag","field_telemetry_claim_flag",
                    "external_attacker_claim_flag","uses_old_asset_name_flag",
                    "overall_evidence_score","parse_success"]:
            ev_mat[col] = c(matrix.get(col))

    author = sc.get("author_model") or (manifest.get("author_model") if manifest else None) or det.get("scenario_id","").split("_")[1] if "_" in det.get("scenario_id","") else "unknown"
    asset  = sc.get("target_asset_id") or (manifest.get("target_asset_id") if manifest else None) or "der_site_001"
    comp   = sc.get("target_component") or (manifest.get("target_component") if manifest else None) or "site"

    return {
        "detection_id": str(det.get("detection_id", "")),
        "scenario_id":  str(det.get("scenario_id", "")),
        "scenario_class": scls,
        "scenario_family": str(det.get("scenario_family", "")),
        "author_model": str(author),
        "target_asset_id": str(asset),
        "target_component": str(comp),
        "detection": {
            "model_name":    "smoke_detector",
            "display_name":  "Demo mode — placeholder detection scores",
            "anomaly_score": c(det.get("anomaly_score")),
            "threshold":     c(det.get("threshold", 0.5)),
            "predicted_label": int(det.get("predicted_label", 0)),
            "smoke_detection_only": True,
        },
        "labels": sc.get("labels", {}),
        "timing": {
            "window_start_s": w_start, "window_end_s": w_end,
            "event_start_s": int(ev_start), "event_end_s": int(ev_end),
            "command_apply_time_s":        c(ta.get("command_apply_time_s")),
            "physical_effect_start_time_s":c(ta.get("physical_effect_start_time_s")),
            "cyber_before_physical":       c(ta.get("cyber_before_physical")),
            "timing_alignment_status":     str(ta.get("timing_alignment_status","unknown")),
        },
        "timeseries": None,  # filled after
        "cyber_timeline": build_cyber_timeline(scls, ev_pkt),
        "explanation":     exp,
        "evidence":        ev_mat,
        "physical_evidence": {
            "top_signals":                      pe.get("top_signals", [])[:5],
            "affected_variables_from_scenario": pe.get("affected_variables_from_scenario", []),
            "physical_effect_active_fraction":  c(pe.get("physical_effect_active_fraction_in_window")),
            "expected_behavior_summary":        str(eva.get("expected_behavior_summary", "")),
            "observed_behavior_summary":        str(eva.get("observed_behavior_summary", "")),
            "expected_vs_observed_difference":  str(eva.get("expected_vs_observed_difference", "")),
        },
        "cyber_evidence": {
            "cyber_states_seen":      ce.get("cyber_states_seen", []),
            "lifecycle_stages_seen":  ce.get("lifecycle_stages_seen", []),
            "blocked_flag_seen":      bool(ce.get("blocked_flag_seen", False)),
            "replay_flag_seen":       bool(ce.get("replay_flag_seen", False)),
            "mismatch_flag_seen":     bool(ce.get("mismatch_flag_seen", False)),
            "stale_command_flag_seen":bool(ce.get("stale_command_flag_seen", False)),
            "timeout_flag_seen":      bool(ce.get("timeout_flag_seen", False)),
            "cyber_anomaly_active_fraction": c(ce.get("cyber_anomaly_active_fraction_in_window")),
        },
    }

def main():
    print("[INFO] Loading data files ...")

    # Load detections
    dets_df = pd.read_csv(SMOKE_DET_CSV)
    print(f"  Detections: {len(dets_df)}")

    # Load manifeset
    manifest_df = pd.read_csv(MANIFEST_CSV)
    manifest_by_sid = {r["scenario_id"]: r for _, r in manifest_df.iterrows()}

    # Load parsed explanations
    parsed_df = pd.read_csv(PARSED_CSV) if PARSED_CSV.exists() else pd.DataFrame()
    parsed_by_did = {}
    if len(parsed_df) > 0:
        for _, r in parsed_df.iterrows():
            parsed_by_did[str(r.get("detection_id", ""))] = r.to_dict()

    # Load evidence matrix
    matrix_df = pd.read_csv(MATRIX_CSV) if MATRIX_CSV.exists() else pd.DataFrame()
    matrix_by_did = {}
    if len(matrix_df) > 0:
        for _, r in matrix_df.iterrows():
            matrix_by_did[str(r.get("detection_id", ""))] = r.to_dict()

    # Load evidence packets
    ev_pkts = load_ev_packets()
    print(f"  Evidence packets: {len(ev_pkts)}")

    # Load score summary
    summary = {}
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON) as f:
            summary = json.load(f)

    # Load physical CSV (large - use chunked read)
    print("[INFO] Loading physical CSV (large) ...")
    phy_df = pd.read_csv(PHYSICAL_CSV, low_memory=False)
    print(f"  Physical CSV: {phy_df.shape}")

    # Detect time column
    time_col = None
    for c_name in ["time_s", "timestamp_s", "second", "t_s"]:
        if c_name in phy_df.columns:
            time_col = c_name
            break
    if time_col is None:
        time_col = phy_df.columns[0]
    print(f"  Time col: {time_col}")

    phys_cols_avail = [c for c in PHYS_SIGNALS if c in phy_df.columns]
    print(f"  Physical signal cols: {len(phys_cols_avail)}")

    # Load aligned CSV
    print("[INFO] Loading aligned 1s CSV ...")
    aln_df = pd.read_csv(ALIGNED_CSV, low_memory=False)
    print(f"  Aligned CSV: {aln_df.shape}")

    aln_time_col = None
    for c_name in ["time_s", "timestamp_s", "second"]:
        if c_name in aln_df.columns:
            aln_time_col = c_name
            break
    cyber_flag_cols_avail = [c for c in CYBER_FLAGS if c in aln_df.columns]
    print(f"  Cyber flag cols: {len(cyber_flag_cols_avail)}")

    # ── Build all records ───────────────────────────────────────────────────
    print("[INFO] Building scenario records ...")
    all_records = []
    for _, det in dets_df.iterrows():
        did = str(det.get("detection_id", ""))
        sid = str(det.get("scenario_id", ""))
        ev_pkt  = ev_pkts.get(did, {})
        parsed  = parsed_by_did.get(did)
        matrix  = matrix_by_did.get(did)
        manifest = manifest_by_sid.get(sid)

        if not ev_pkt:
            # Create minimal evidence packet from detection row
            ev_pkt = {
                "detection": det.to_dict(),
                "scenario": {"scenario_id": sid, "scenario_class": det.get("scenario_class", "unknown"),
                             "scenario_family": det.get("scenario_family", ""), "labels": {}},
                "physical_evidence": {}, "cyber_evidence": {}, "timing_alignment": {},
                "expected_vs_actual": {},
            }

        rec = build_record(det.to_dict(), ev_pkt, parsed, matrix, manifest.to_dict() if manifest is not None else None)

        # Add timeseries
        ev_start = rec["timing"]["event_start_s"]
        ev_end   = rec["timing"]["event_end_s"]
        ts = extract_ts(phy_df, aln_df, ev_start, ev_end, time_col,
                        phys_cols_avail, cyber_flag_cols_avail, aln_time_col)
        rec["timeseries"] = ts

        all_records.append(rec)
        print(f"  Built {did} | {sid} | {rec['scenario_class']} | {len(ts.get('time_relative',[]))} pts")

    # ── Select presenter scenarios (1 per class, best score) ─────────────────
    presenter_records = []
    CLASS_ORDER = ["cyber_physical", "physical_only", "cyber_only", "normal"]
    DEMO_LABELS = {
        "cyber_physical": "Demo A — Cyber-Physical",
        "physical_only":  "Demo B — Physical-Only",
        "cyber_only":     "Demo C — Cyber-Only",
        "normal":         "Demo D — Normal Behavior",
    }

    score_by_did = {
        str(r.get("detection_id", "")): float(r.get("overall_evidence_score", 0) or 0)
        for _, r in matrix_df.iterrows()
    } if len(matrix_df) > 0 else {}

    used_sids = set()
    for cls in CLASS_ORDER:
        candidates = [r for r in all_records if r["scenario_class"] == cls and r["scenario_id"] not in used_sids]
        if not candidates:
            continue
        # Sort by evidence score descending, then by physical signal delta
        def score_key(r):
            ev_score = score_by_did.get(r["detection_id"], 0.5)
            ts = r.get("timeseries", {})
            pv_delta = abs(ts.get("pv_p_kw", [0])[-1] - ts.get("pv_p_kw", [0])[0]) if ts.get("pv_p_kw") else 0
            bess_delta = abs(ts.get("bess_p_kw", [0])[-1] - ts.get("bess_p_kw", [0])[0]) if ts.get("bess_p_kw") else 0
            return ev_score + 0.01 * max(pv_delta, bess_delta)
        candidates.sort(key=score_key, reverse=True)
        best = candidates[0]
        best = dict(best)
        best["demo_label"] = DEMO_LABELS[cls]
        best["demo_class_key"] = cls
        used_sids.add(best["scenario_id"])
        presenter_records.append(best)

    print(f"\n  Presenter scenarios selected: {len(presenter_records)}")
    for r in presenter_records:
        print(f"    {r['demo_label']}: {r['scenario_id']}")

    # ── Write demo_summary.json ───────────────────────────────────────────────
    avg_score = summary.get("average_evidence_score", 0.0)
    dem_summary = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "total_demo_scenarios": len(all_records),
        "presenter_scenarios": len(presenter_records),
        "average_evidence_score": avg_score,
        "unsupported_claims_count": summary.get("unsupported_claims_total", 0),
        "packet_level_claims_count": summary.get("packet_claims_total", 0),
        "field_telemetry_claims_count": summary.get("field_telemetry_claims", 0),
        "detection_mode_display": "Demo mode — placeholder detection scores",
        "selected_llm_model": "qwen2.5:3b-instruct",
        "phase2_scenarios_total": 64,
        "phase3_detections_total": len(all_records),
        "data_source": "live",
        "class_counts": {r["scenario_class"]: sum(1 for x in all_records if x["scenario_class"] == r["scenario_class"])
                         for r in all_records},
    }

    # ── Write JSON files ───────────────────────────────────────────────────────
    def write_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Wrote {path.name} ({path.stat().st_size // 1024} KB)")

    write_json(PUB_DATA / "demo_presenter_scenarios.json", presenter_records)
    write_json(PUB_DATA / "demo_technical_scenarios.json", all_records)
    write_json(PUB_DATA / "demo_summary.json", dem_summary)

    # Legacy filenames for compatibility
    write_json(PUB_DATA / "demo_scenarios.json", presenter_records)

    # ── Write fallback copies ─────────────────────────────────────────────────
    fallback_summary = dict(dem_summary)
    fallback_summary["data_source"] = "fallback"

    write_json(PUB_DATA / "fallback_demo_presenter_scenarios.json", presenter_records)
    write_json(PUB_DATA / "fallback_demo_technical_scenarios.json", all_records)
    write_json(PUB_DATA / "fallback_demo_summary.json", fallback_summary)
    write_json(PUB_DATA / "fallback_demo_scenarios.json", presenter_records)

    print(f"\n[DONE] Demo data exported to {PUB_DATA}")
    print(f"  Presenter scenarios : {len(presenter_records)}")
    print(f"  Technical scenarios : {len(all_records)}")
    print(f"  Average evidence score: {avg_score:.3f}")
    return True

if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"[ERROR] prepare_demo_data.py failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
