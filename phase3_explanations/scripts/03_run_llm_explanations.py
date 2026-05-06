"""
Phase 3 Step 3: Call Ollama/Qwen on each explanation input packet.
Uses local Ollama only. Falls back gracefully if LLM not available.
"""

import json
import re
import time
import urllib.request
import urllib.error
import pandas as pd
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "phase3_explanation_config.json"
RUNTIME_CFG = ROOT / "configs" / "runtime_selected_model.json"
OUT_DIR = ROOT / "outputs"
RPT_DIR = ROOT / "reports"
PROMPTS_DIR = ROOT / "prompts"
OUT_DIR.mkdir(exist_ok=True)
RPT_DIR.mkdir(exist_ok=True)

with open(CFG_PATH) as f:
    cfg = json.load(f)

# Load runtime model selection
llm_status = "NOT_READY"
selected_model = None
if RUNTIME_CFG.exists():
    with open(RUNTIME_CFG) as f:
        rt = json.load(f)
    selected_model = rt.get("selected_model")
    llm_status = rt.get("llm_status", "NOT_READY")

OLLAMA_HOST = cfg["ollama_host"]
TEMPERATURE = cfg.get("temperature", 0.1)
TOP_P = cfg.get("top_p", 0.9)
MAX_TOKENS_APPROX = cfg.get("max_prompt_tokens_approx", 3500)

print(f"[INFO] LLM status: {llm_status}")
print(f"[INFO] Selected model: {selected_model}")

# ── Load prompt template ───────────────────────────────────────────────────────
template_path = PROMPTS_DIR / "grounded_explanation_prompt_template.md"
prompt_template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""

# ── Load explanation inputs ───────────────────────────────────────────────────
inputs_path = OUT_DIR / "explanation_inputs.jsonl"
packets = []
with open(inputs_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            packets.append(json.loads(line))
print(f"[INFO] Loaded {len(packets)} explanation input packets.")

# ── LLM call helper ───────────────────────────────────────────────────────────
REQUIRED_KEYS = [
    "explanation_type", "confidence", "primary_asset",
    "primary_physical_signals", "primary_cyber_evidence",
    "expected_vs_observed_summary", "timing_summary", "operator_summary",
    "recommended_operator_checks", "evidence_used", "evidence_missing",
    "unsupported_claims_made", "packet_level_claim_made",
    "field_telemetry_claim_made", "external_attacker_claim_made",
    "old_asset_name_used", "human_explanation",
]

VALID_EXPLANATION_TYPES = {
    "normal", "physical_only", "cyber_only", "cyber_physical", "insufficient_evidence"
}

def build_compact_evidence(pkt):
    """Trim evidence packet to stay within token budget."""
    det = pkt.get("detection", {})
    sc  = pkt.get("scenario", {})
    pe  = pkt.get("physical_evidence", {})
    ce  = pkt.get("cyber_evidence", {})
    eva = pkt.get("expected_vs_actual", {})
    ta  = pkt.get("timing_alignment", {})

    # Keep top 4 physical signals only
    top_sigs = pe.get("top_signals", [])[:4]

    # Keep only flags that are active
    active_flags = {k: v for k, v in ce.get("command_flags_seen", {}).items()
                    if v.get("any_active")}

    compact = {
        "detection": {
            "detection_id":    det.get("detection_id"),
            "window_start_s":  det.get("window_start_s"),
            "window_end_s":    det.get("window_end_s"),
            "anomaly_score":   det.get("anomaly_score"),
            "threshold":       det.get("threshold"),
            "predicted_label": det.get("predicted_label"),
            "smoke_detection_only": det.get("smoke_detection_only"),
        },
        "scenario": {
            "scenario_id":      sc.get("scenario_id"),
            "scenario_class":   sc.get("scenario_class"),
            "scenario_family":  sc.get("scenario_family"),
            "target_asset_id":  sc.get("target_asset_id"),
            "target_component": sc.get("target_component"),
        },
        "physical_evidence": {
            "top_signals":                      top_sigs,
            "signal_direction_summary":         pe.get("signal_direction_summary", {}),
            "affected_variables_from_scenario": pe.get("affected_variables_from_scenario", []),
            "physical_effect_active_fraction":  pe.get("physical_effect_active_fraction_in_window"),
            "physical_evidence_available":      pe.get("physical_evidence_available"),
        },
        "cyber_evidence": {
            "cyber_states_seen":              ce.get("cyber_states_seen", []),
            "lifecycle_stages_seen":          ce.get("lifecycle_stages_seen", []),
            "active_flags":                   list(active_flags.keys()),
            "blocked_flag_seen":              ce.get("blocked_flag_seen"),
            "replay_flag_seen":               ce.get("replay_flag_seen"),
            "mismatch_flag_seen":             ce.get("mismatch_flag_seen"),
            "stale_command_flag_seen":        ce.get("stale_command_flag_seen"),
            "timeout_flag_seen":              ce.get("timeout_flag_seen"),
            "cyber_anomaly_active_fraction":  ce.get("cyber_anomaly_active_fraction_in_window"),
            "event_level_only":               True,
            "packet_level_protocol_compliance_claimed": False,
        },
        "expected_vs_actual": {
            "expected_behavior_summary":       eva.get("expected_behavior_summary", ""),
            "observed_behavior_summary":       eva.get("observed_behavior_summary", ""),
            "top_physical_signals_from_window": eva.get("top_physical_signals_from_window", []),
            "cyber_flags_observed":            eva.get("cyber_flags_observed", []),
        },
        "timing": {
            "detection_overlaps_scenario":     ta.get("detection_overlaps_scenario"),
            "overlap_seconds":                 ta.get("detection_overlap_seconds"),
            "command_apply_time_s":            ta.get("command_apply_time_s"),
            "physical_effect_start_time_s":    ta.get("physical_effect_start_time_s"),
            "cyber_before_physical":           ta.get("cyber_before_physical"),
            "timing_alignment_status":         ta.get("timing_alignment_status"),
        },
        "guardrails": pkt.get("guardrails", {}),
    }
    return compact

def build_prompt(pkt):
    compact = build_compact_evidence(pkt)
    evidence_json = json.dumps(compact, indent=2, default=str)
    prompt = prompt_template.replace("{{EVIDENCE_PACKET_JSON}}", evidence_json)
    return prompt

def call_ollama(model, prompt, attempt=1):
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p":       TOP_P,
        }
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", ""), None
    except urllib.error.URLError as e:
        return None, f"URLError: {e}"
    except Exception as e:
        return None, f"Error: {e}"

def extract_json(raw):
    """Try to extract JSON from LLM response."""
    if raw is None:
        return None, "null_response"
    text = raw.strip()
    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    # Find first { to last }
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return None, "no_json_braces"
    json_str = text[start:end+1]
    try:
        return json.loads(json_str), "ok"
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"

REPAIR_PROMPT = (
    "The previous response was not valid JSON. "
    "Return ONLY a valid JSON object matching this schema: "
    '{"explanation_type":"...","confidence":"...","primary_asset":"...",'
    '"primary_physical_signals":[],"primary_cyber_evidence":[],'
    '"expected_vs_observed_summary":"...","timing_summary":"...","operator_summary":"...",'
    '"recommended_operator_checks":[],"evidence_used":[],"evidence_missing":[],'
    '"unsupported_claims_made":false,"packet_level_claim_made":false,'
    '"field_telemetry_claim_made":false,"external_attacker_claim_made":false,'
    '"old_asset_name_used":false,"human_explanation":"..."}'
    " No markdown, no explanation outside JSON."
)

def safe_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [v]
    return []

def normalize_parsed(parsed):
    """Fill missing keys with safe defaults."""
    if not isinstance(parsed, dict):
        return None
    if "explanation_type" not in parsed or parsed["explanation_type"] not in VALID_EXPLANATION_TYPES:
        parsed["explanation_type"] = "insufficient_evidence"
    parsed.setdefault("confidence", "low")
    parsed.setdefault("primary_asset", "")
    parsed.setdefault("primary_physical_signals", [])
    parsed.setdefault("primary_cyber_evidence", [])
    parsed.setdefault("expected_vs_observed_summary", "")
    parsed.setdefault("timing_summary", "")
    parsed.setdefault("operator_summary", "")
    parsed.setdefault("recommended_operator_checks", [])
    parsed.setdefault("evidence_used", [])
    parsed.setdefault("evidence_missing", [])
    parsed.setdefault("unsupported_claims_made", False)
    parsed.setdefault("packet_level_claim_made", False)
    parsed.setdefault("field_telemetry_claim_made", False)
    parsed.setdefault("external_attacker_claim_made", False)
    parsed.setdefault("old_asset_name_used", False)
    parsed.setdefault("human_explanation", "")
    # Ensure lists
    for k in ["primary_physical_signals", "primary_cyber_evidence",
              "recommended_operator_checks", "evidence_used", "evidence_missing"]:
        parsed[k] = safe_list(parsed[k])
    return parsed

# ── Run explanations ──────────────────────────────────────────────────────────
raw_outputs   = []
parsed_rows   = []
run_start     = datetime.utcnow()
llm_available = llm_status in ("READY", "READY_BUT_TEST_UNEXPECTED", "READY_BUT_JSON_PARSE_FAILED")

skipped = 0

for i, pkt in enumerate(packets):
    det_id  = pkt.get("detection", {}).get("detection_id", f"det_{i:04d}")
    sid     = pkt.get("scenario", {}).get("scenario_id", "unknown")
    scls    = pkt.get("scenario", {}).get("scenario_class", "unknown")
    print(f"  [{i+1}/{len(packets)}] {det_id} | {sid} | {scls} ...", end=" ", flush=True)

    raw_record = {
        "detection_id":  det_id,
        "scenario_id":   sid,
        "scenario_class": scls,
        "model_used":    selected_model,
        "llm_available": llm_available,
        "raw_response":  None,
        "parse_status":  "SKIPPED",
        "parsed":        None,
        "error":         None,
        "timestamp":     datetime.utcnow().isoformat() + "Z",
    }

    if not llm_available:
        raw_record["parse_status"] = "SKIPPED_LLM_NOT_READY"
        raw_record["error"] = f"LLM not ready: {llm_status}"
        skipped += 1
        print("SKIPPED")
        raw_outputs.append(raw_record)
        continue

    try:
        prompt = build_prompt(pkt)
        raw_resp, err = call_ollama(selected_model, prompt)
        raw_record["raw_response"] = raw_resp[:2000] if raw_resp else None

        if err:
            raw_record["parse_status"] = "FAILED_CALL_ERROR"
            raw_record["error"] = err
            print(f"CALL_ERROR: {err}")
        else:
            parsed, status = extract_json(raw_resp)
            if parsed is not None:
                parsed = normalize_parsed(parsed)
                raw_record["parsed"] = parsed
                raw_record["parse_status"] = "OK"
                print("OK")
            else:
                # One repair attempt
                print(f"JSON_FAIL({status}) -> repair...", end=" ", flush=True)
                repair_resp, repair_err = call_ollama(selected_model, REPAIR_PROMPT)
                raw_record["raw_response"] = (raw_record["raw_response"] or "") + \
                    "\n---REPAIR---\n" + (repair_resp[:1000] if repair_resp else "")
                if repair_err:
                    raw_record["parse_status"] = "FAILED_REPAIR_CALL_ERROR"
                    raw_record["error"] = repair_err
                    print("REPAIR_CALL_ERROR")
                else:
                    parsed2, status2 = extract_json(repair_resp)
                    if parsed2 is not None:
                        parsed2 = normalize_parsed(parsed2)
                        raw_record["parsed"] = parsed2
                        raw_record["parse_status"] = "OK_REPAIRED"
                        print("REPAIRED")
                    else:
                        raw_record["parse_status"] = "FAILED"
                        raw_record["error"] = f"Parse failed after repair: {status2}"
                        print(f"FAILED: {status2}")
    except Exception as e:
        raw_record["parse_status"] = "EXCEPTION"
        raw_record["error"] = str(e)
        print(f"EXCEPTION: {e}")

    raw_outputs.append(raw_record)

# ── Save raw JSONL ─────────────────────────────────────────────────────────────
raw_path = OUT_DIR / "llm_explanations_raw.jsonl"
with open(raw_path, "w", encoding="utf-8") as f:
    for rec in raw_outputs:
        f.write(json.dumps(rec, default=str) + "\n")
print(f"[INFO] Raw outputs saved: {raw_path} ({len(raw_outputs)} records)")

# ── Build parsed CSV ──────────────────────────────────────────────────────────
for rec in raw_outputs:
    p = rec.get("parsed") or {}
    row = {
        "detection_id":              rec["detection_id"],
        "scenario_id":               rec["scenario_id"],
        "model_name":                rec.get("model_used") or "none",
        "explanation_type":          p.get("explanation_type", ""),
        "confidence":                p.get("confidence", ""),
        "primary_asset":             p.get("primary_asset", ""),
        "primary_physical_signals":  json.dumps(p.get("primary_physical_signals", [])),
        "primary_cyber_evidence":    json.dumps(p.get("primary_cyber_evidence", [])),
        "expected_vs_observed_summary": p.get("expected_vs_observed_summary", ""),
        "timing_summary":            p.get("timing_summary", ""),
        "operator_summary":          p.get("operator_summary", ""),
        "recommended_operator_checks": json.dumps(p.get("recommended_operator_checks", [])),
        "unsupported_claims_made":   p.get("unsupported_claims_made", False),
        "packet_level_claim_made":   p.get("packet_level_claim_made", False),
        "field_telemetry_claim_made": p.get("field_telemetry_claim_made", False),
        "external_attacker_claim_made": p.get("external_attacker_claim_made", False),
        "old_asset_name_used":       p.get("old_asset_name_used", False),
        "human_explanation":         p.get("human_explanation", ""),
        "parse_status":              rec.get("parse_status", "SKIPPED"),
        "evidence_used":             json.dumps(p.get("evidence_used", [])),
        "evidence_missing":          json.dumps(p.get("evidence_missing", [])),
    }
    parsed_rows.append(row)

parsed_df = pd.DataFrame(parsed_rows)
parsed_path = OUT_DIR / "llm_explanations_parsed.csv"
parsed_df.to_csv(parsed_path, index=False)
print(f"[INFO] Parsed CSV saved: {parsed_path} ({len(parsed_df)} rows)")

# ── Report ─────────────────────────────────────────────────────────────────────
run_end = datetime.utcnow()
elapsed = (run_end - run_start).total_seconds()

status_counts = parsed_df["parse_status"].value_counts().to_dict()
ok_count   = status_counts.get("OK", 0) + status_counts.get("OK_REPAIRED", 0)
fail_count = sum(v for k, v in status_counts.items() if "FAIL" in k)
skip_count = sum(v for k, v in status_counts.items() if "SKIP" in k)

exp_types = parsed_df[parsed_df["parse_status"].isin(["OK","OK_REPAIRED"])]["explanation_type"].value_counts().to_dict()

report_lines = [
    "# LLM EXPLANATION RUN REPORT",
    "",
    f"Generated: {run_end.strftime('%Y-%m-%d %H:%M UTC')}",
    f"Elapsed: {elapsed:.1f}s",
    "",
    "## LLM Configuration",
    "",
    f"| Item | Value |",
    f"|---|---|",
    f"| LLM status | {llm_status} |",
    f"| Selected model | {selected_model or 'None'} |",
    f"| Ollama host | {OLLAMA_HOST} |",
    f"| Temperature | {TEMPERATURE} |",
    "",
    "## Run Summary",
    "",
    f"| Item | Count |",
    f"|---|---|",
    f"| Total packets | {len(packets)} |",
    f"| Successful (OK + repaired) | {ok_count} |",
    f"| Failed | {fail_count} |",
    f"| Skipped (LLM not ready) | {skip_count} |",
    "",
    "## Parse Status Breakdown",
    "",
    "| Status | Count |",
    "|---|---|",
]
for k, v in status_counts.items():
    report_lines.append(f"| {k} | {v} |")

report_lines += [
    "",
    "## Explanation Type Distribution (successful only)",
    "",
    "| explanation_type | count |",
    "|---|---|",
]
for k, v in exp_types.items():
    report_lines.append(f"| {k} | {v} |")

report_lines += [
    "",
    "## Notes",
    "",
    "- LLM is used for explanation only, not anomaly detection.",
    "- All physical/cyber/context evidence is from real Phase 2 files.",
    "- Smoke detections replace only anomaly_score/threshold/predicted_label.",
    "- When real frozen-model outputs are available, replace smoke_model_detections.csv",
    "  and re-run this pipeline unchanged.",
    ""
]

rpt_path = RPT_DIR / "LLM_EXPLANATION_RUN_REPORT.md"
with open(rpt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"[INFO] Report saved: {rpt_path}")
print(f"[DONE] {ok_count} explanations OK, {fail_count} failed, {skip_count} skipped.")
