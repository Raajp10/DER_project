"""
Phase 3 Step 4: Score LLM explanations against known scenario evidence.
Does not fake results. Reports actual scores.
"""

import json
import re
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

print("[INFO] Loading scoring inputs ...")
inputs_path  = OUT_DIR / "explanation_inputs.jsonl"
parsed_path  = OUT_DIR / "llm_explanations_parsed.csv"
manifest_path = Path(cfg["zero_day_scenario_manifest"])

parsed_df   = pd.read_csv(parsed_path)
manifest_df = pd.read_csv(manifest_path)
print(f"  Parsed rows: {len(parsed_df)}")
print(f"  Manifest  : {len(manifest_df)}")

# Load input packets for evidence cross-check
input_pkts = {}
with open(inputs_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            did = obj.get("detection", {}).get("detection_id")
            if did:
                input_pkts[did] = obj

# ── Helpers ───────────────────────────────────────────────────────────────────
OLD_ASSET_NAMES = {"pv35", "pv60", "pv83", "bess48", "bess108"}
VALID_ASSETS    = {"der_site_001", "pv_001", "bess_001", "pcc_001"}
GLOSSARY_FIELDS = {
    "pv_p_kw", "pv_q_kvar", "bess_p_kw", "bess_q_kvar", "bess_soc_percent",
    "pcc_v_a_pu", "pcc_v_b_pu", "pcc_v_c_pu", "pcc_i_a_amp", "pcc_i_b_amp",
    "pcc_i_c_amp", "pcc_p_kw", "pcc_q_kvar", "irradiance_pu", "temperature_c",
    "cyber_state", "cyber_lifecycle_stage", "cyber_anomaly_active",
    "physical_effect_active", "command_created_flag", "command_sent_flag",
    "command_recv_flag", "command_accept_flag", "command_apply_flag",
    "command_response_flag", "blocked_flag", "replay_flag", "mismatch_flag",
    "stale_command_flag", "timeout_flag",
    "command_apply_time_s", "physical_effect_start_time_s",
}
PACKET_CLAIM_PATTERNS = re.compile(
    r"packet|ieee\s*2030\.5|modbus|dnp3|mqtt|payload|frame|byte|register|protocol trace",
    re.IGNORECASE
)
FIELD_TELEMETRY_PATTERNS = re.compile(
    r"real\s+field|live\s+measurement|actual\s+telemetry|real\s+telemetry|real\s+sensor",
    re.IGNORECASE
)
ATTACKER_PATTERNS = re.compile(
    r"hacker|malware|exploit|attacker|threat\s+actor|nation[\s-]state|ransomware|apt\s*\d",
    re.IGNORECASE
)

CLASS_TO_EXPTYPE = {
    "normal":        "normal",
    "physical_only": "physical_only",
    "cyber_only":    "cyber_only",
    "cyber_physical":"cyber_physical",
}

TIMING_EXPECTED = {
    "normal":         ["no_anomaly"],
    "physical_only":  ["physical_only_no_cyber_event"],
    "cyber_only":     ["cyber_only_no_physical_effect"],
    "cyber_physical": ["cyber_before_physical_confirmed", "cyber_physical_simultaneous", "cyber_physical_timing_unknown"],
}

def safe_list_field(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return [v] if v else []
    return []

def text_fields(row):
    """Concatenate all text fields for scanning."""
    parts = [
        str(row.get("expected_vs_observed_summary", "")),
        str(row.get("timing_summary", "")),
        str(row.get("operator_summary", "")),
        str(row.get("human_explanation", "")),
        str(row.get("primary_cyber_evidence", "")),
        str(row.get("primary_physical_signals", "")),
    ]
    return " ".join(parts).lower()

# ── Score each row ────────────────────────────────────────────────────────────
def get_manifest_row(sid):
    sub = manifest_df[manifest_df["scenario_id"] == sid]
    if len(sub) == 0:
        return {}
    return sub.iloc[0].to_dict()

score_rows = []

for _, row in parsed_df.iterrows():
    det_id  = row.get("detection_id", "")
    sid     = row.get("scenario_id", "")
    exptype = str(row.get("explanation_type", ""))
    conf    = str(row.get("confidence", ""))
    p_asset = str(row.get("primary_asset", "")).strip().lower()
    parse_ok = row.get("parse_status", "SKIPPED") in ("OK", "OK_REPAIRED")

    mrow = get_manifest_row(sid)
    true_class       = str(mrow.get("scenario_class", "unknown"))
    true_asset_id    = str(mrow.get("target_asset_id", "")).strip().lower()
    true_component   = str(mrow.get("target_component", "")).strip().lower()
    true_family      = str(mrow.get("scenario_family", ""))

    pkt = input_pkts.get(det_id, {})
    pe  = pkt.get("physical_evidence", {})
    ce  = pkt.get("cyber_evidence", {})
    ta  = pkt.get("timing_alignment", {})

    affected_vars = pe.get("affected_variables_from_scenario", [])
    ev_signals    = [s["signal"] for s in pe.get("top_signals", [])]
    cyber_states  = ce.get("cyber_states_seen", [])
    timing_status = ta.get("timing_alignment_status", "")

    full_text = text_fields(row)
    ev_list   = safe_list_field(row.get("evidence_used", "[]"))
    phys_sigs = safe_list_field(row.get("primary_physical_signals", "[]"))
    cyber_ev  = safe_list_field(row.get("primary_cyber_evidence", "[]"))

    # ── Scoring ───────────────────────────────────────────────────────────────
    sc = {}

    # 1. scenario_class_match
    expected_exptype = CLASS_TO_EXPTYPE.get(true_class)
    sc["scenario_class_match"] = int(parse_ok and exptype == expected_exptype)

    # 2. asset_match
    matches_asset = (
        p_asset in (true_asset_id, true_component) or
        true_component in p_asset or
        true_asset_id in p_asset
    )
    sc["asset_match"] = int(parse_ok and (matches_asset or p_asset in VALID_ASSETS))

    # 3. physical_signal_match
    if true_class in ("physical_only", "cyber_physical"):
        phys_in_ev = any(s in (phys_sigs + ev_list) for s in (affected_vars + ev_signals))
        sc["physical_signal_match"] = int(parse_ok and phys_in_ev)
    else:
        sc["physical_signal_match"] = 1  # N/A for cyber_only/normal

    # 4. cyber_state_match
    if true_class in ("cyber_only", "cyber_physical"):
        cyber_mentioned = len(cyber_ev) > 0 or any(
            flag in full_text for flag in ["blocked_flag", "mismatch_flag", "replay_flag",
                                           "timeout_flag", "stale_command_flag",
                                           "command_apply_flag", "cyber_state"]
        )
        sc["cyber_state_match"] = int(parse_ok and cyber_mentioned)
    else:
        sc["cyber_state_match"] = 1  # N/A

    # 5. timing_match
    expected_timings = TIMING_EXPECTED.get(true_class, [])
    sc["timing_match"] = int(parse_ok and (
        timing_status in expected_timings or
        not expected_timings  # no expectation
    ))

    # 6. expected_vs_observed_match
    evos = str(row.get("expected_vs_observed_summary", "")).strip()
    sc["expected_vs_observed_match"] = int(parse_ok and len(evos) > 30)

    # 7. glossary_field_usage_correct
    used_fields = set(ev_list + phys_sigs + cyber_ev)
    gloss_used  = used_fields & GLOSSARY_FIELDS
    sc["glossary_field_usage_correct"] = int(parse_ok and len(gloss_used) > 0)

    # 8. explanation_mentions_evidence_fields
    gloss_in_text = sum(1 for f in GLOSSARY_FIELDS if f in full_text)
    sc["explanation_mentions_evidence_fields"] = int(parse_ok and gloss_in_text >= 1)

    # 9. timing_relationship_correct
    timing_text = str(row.get("timing_summary", "")).lower()
    timing_ok = False
    if true_class == "normal":
        timing_ok = any(w in timing_text for w in ["normal", "no anomaly", "no event"])
    elif true_class == "physical_only":
        timing_ok = any(w in timing_text for w in ["physical", "no cyber", "without cyber"])
    elif true_class == "cyber_only":
        timing_ok = any(w in timing_text for w in ["cyber", "no physical", "not physical"])
    elif true_class == "cyber_physical":
        timing_ok = any(w in timing_text for w in ["before", "coincide", "preceded", "followed", "cyber"])
    sc["timing_relationship_correct"] = int(parse_ok and timing_ok)

    # 10. operator_action_relevance
    rec_checks_raw = row.get("recommended_operator_checks", "[]")
    rec_checks = safe_list_field(rec_checks_raw)
    rec_text   = " ".join(rec_checks).lower()
    op_rel     = False
    if true_class == "normal":
        op_rel = len(rec_checks) == 0  # no action needed for normal
    elif true_component:
        op_rel = true_component in rec_text or true_asset_id in rec_text
    sc["operator_action_relevance"] = int(parse_ok and op_rel)

    # ── Penalty flags ─────────────────────────────────────────────────────────
    # Use both LLM-self-reported flags AND text scanning for robustness
    lm_pkt_claim   = bool(row.get("packet_level_claim_made", False))
    lm_ft_claim    = bool(row.get("field_telemetry_claim_made", False))
    lm_att_claim   = bool(row.get("external_attacker_claim_made", False))
    lm_old_asset   = bool(row.get("old_asset_name_used", False))
    lm_unsupported = bool(row.get("unsupported_claims_made", False))

    text_pkt  = bool(PACKET_CLAIM_PATTERNS.search(full_text))
    text_ft   = bool(FIELD_TELEMETRY_PATTERNS.search(full_text))
    text_att  = bool(ATTACKER_PATTERNS.search(full_text))
    text_old  = any(n in full_text for n in OLD_ASSET_NAMES)

    sc["unsupported_claim_flag"]       = int(lm_unsupported)
    sc["packet_claim_flag"]            = int(lm_pkt_claim or text_pkt)
    sc["field_telemetry_claim_flag"]   = int(lm_ft_claim or text_ft)
    sc["external_attacker_claim_flag"] = int(lm_att_claim or text_att)
    sc["uses_old_asset_name_flag"]     = int(lm_old_asset or text_old)

    sc["parse_success"] = int(parse_ok)

    # ── Overall evidence score ────────────────────────────────────────────────
    # Positive checks (10 items)
    positive_keys = [
        "scenario_class_match", "asset_match", "physical_signal_match",
        "cyber_state_match", "timing_match", "expected_vs_observed_match",
        "glossary_field_usage_correct", "explanation_mentions_evidence_fields",
        "timing_relationship_correct", "operator_action_relevance",
    ]
    # Penalty flags (5 items) — each deducts 0.15
    penalty_keys = [
        "unsupported_claim_flag", "packet_claim_flag",
        "field_telemetry_claim_flag", "external_attacker_claim_flag",
        "uses_old_asset_name_flag",
    ]
    if parse_ok:
        pos_score = sum(sc[k] for k in positive_keys) / len(positive_keys)
        penalties = sum(sc[k] for k in penalty_keys) * 0.15
        overall = max(0.0, min(1.0, pos_score - penalties))
    else:
        overall = 0.0

    sc["overall_evidence_score"] = round(overall, 4)

    score_rows.append({
        "detection_id":         det_id,
        "scenario_id":          sid,
        "true_scenario_class":  true_class,
        "predicted_explanation_type": exptype,
        **{k: sc[k] for k in positive_keys},
        **{k: sc[k] for k in penalty_keys},
        "parse_success":        sc["parse_success"],
        "overall_evidence_score": sc["overall_evidence_score"],
    })

# ── Save matrix ───────────────────────────────────────────────────────────────
matrix_df = pd.DataFrame(score_rows)
matrix_path = OUT_DIR / "explanation_evidence_matrix.csv"
matrix_df.to_csv(matrix_path, index=False)
print(f"[INFO] Evidence matrix saved: {matrix_path} ({len(matrix_df)} rows)")

# ── Summary JSON ──────────────────────────────────────────────────────────────
parsed_ok_mask = matrix_df["parse_success"] == 1
avg_score = float(matrix_df.loc[parsed_ok_mask, "overall_evidence_score"].mean()) if parsed_ok_mask.any() else 0.0
class_scores = {}
for cls in matrix_df["true_scenario_class"].unique():
    sub = matrix_df[matrix_df["true_scenario_class"] == cls]
    sub_ok = sub[sub["parse_success"] == 1]
    class_scores[cls] = round(float(sub_ok["overall_evidence_score"].mean()), 4) if len(sub_ok) > 0 else None

summary = {
    "timestamp":              datetime.utcnow().isoformat() + "Z",
    "total_scored":           int(len(matrix_df)),
    "parse_success_count":    int(parsed_ok_mask.sum()),
    "average_evidence_score": round(avg_score, 4),
    "class_scores":           class_scores,
    "scenario_class_match_rate": round(float(matrix_df.loc[parsed_ok_mask,"scenario_class_match"].mean()), 4) if parsed_ok_mask.any() else 0.0,
    "unsupported_claims_total": int(matrix_df["unsupported_claim_flag"].sum()),
    "packet_claims_total":      int(matrix_df["packet_claim_flag"].sum()),
    "field_telemetry_claims":   int(matrix_df["field_telemetry_claim_flag"].sum()),
    "external_attacker_claims": int(matrix_df["external_attacker_claim_flag"].sum()),
    "old_asset_name_uses":      int(matrix_df["uses_old_asset_name_flag"].sum()),
}

summary_path = OUT_DIR / "explanation_score_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[INFO] Score summary saved: {summary_path}")

# ── Report ─────────────────────────────────────────────────────────────────────
positive_cols = [
    "scenario_class_match", "asset_match", "physical_signal_match",
    "cyber_state_match", "timing_match", "expected_vs_observed_match",
    "glossary_field_usage_correct", "explanation_mentions_evidence_fields",
    "timing_relationship_correct", "operator_action_relevance",
]

report_lines = [
    "# EXPLANATION EVIDENCE MATRIX REPORT",
    "",
    f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "## Summary",
    "",
    f"| Item | Value |",
    f"|---|---|",
    f"| Total scored | {summary['total_scored']} |",
    f"| Parse success | {summary['parse_success_count']} |",
    f"| Average evidence score | {summary['average_evidence_score']:.3f} |",
    f"| Scenario class match rate | {summary['scenario_class_match_rate']:.3f} |",
    f"| Unsupported claims | {summary['unsupported_claims_total']} |",
    f"| Packet-level claims | {summary['packet_claims_total']} |",
    f"| Field telemetry claims | {summary['field_telemetry_claims']} |",
    f"| External attacker claims | {summary['external_attacker_claims']} |",
    f"| Old asset name uses | {summary['old_asset_name_uses']} |",
    "",
    "## Score by Scenario Class",
    "",
    "| class | avg_score |",
    "|---|---|",
]
for cls, score in class_scores.items():
    score_str = f"{score:.3f}" if score is not None else "N/A (no parsed results)"
    report_lines.append(f"| {cls} | {score_str} |")

report_lines += [
    "",
    "## Positive Check Rates (parsed only)",
    "",
    "| Check | Rate |",
    "|---|---|",
]
for col in positive_cols:
    rate = float(matrix_df.loc[parsed_ok_mask, col].mean()) if parsed_ok_mask.any() else 0.0
    report_lines.append(f"| {col} | {rate:.3f} |")

report_lines += [
    "",
    "## Scoring Logic",
    "",
    "- **Positive checks**: 10 items scored 0/1; averaged to base score (0-1).",
    "- **Penalties**: each flag deducts 0.15 from the base score.",
    "- **overall_evidence_score** = max(0, min(1, avg_positive - sum_penalties * 0.15)).",
    "- If parse_success = 0, overall_evidence_score = 0.",
    "",
    "## Disclaimer",
    "",
    "These are smoke-run scores. The LLM explanations were generated from",
    "synthetic smoke model detections. Replace smoke_model_detections.csv with",
    "real frozen-model outputs and re-run for authoritative results.",
    ""
]

rpt_path = RPT_DIR / "EXPLANATION_EVIDENCE_MATRIX_REPORT.md"
with open(rpt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"[INFO] Report saved: {rpt_path}")
print(f"[DONE] Average evidence score: {avg_score:.3f}")
