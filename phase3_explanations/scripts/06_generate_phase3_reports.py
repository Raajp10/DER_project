"""
Phase 3 Step 6: Generate Phase 3 final reports.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "phase3_explanation_config.json"
OUT_DIR = ROOT / "outputs"
RPT_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"
RPT_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

with open(CFG_PATH) as f:
    cfg = json.load(f)

now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# ── Load available data ────────────────────────────────────────────────────────
def try_load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

def try_load_csv(path, default_cols=None):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=default_cols or [])

summary      = try_load_json(OUT_DIR / "explanation_score_summary.json")
runtime_cfg  = try_load_json(ROOT / "configs" / "runtime_selected_model.json")
ollama_check = try_load_json(OUT_DIR / "ollama_qwen_check.json")
parsed_df    = try_load_csv(OUT_DIR / "llm_explanations_parsed.csv")
matrix_df    = try_load_csv(OUT_DIR / "explanation_evidence_matrix.csv")

selected_model = runtime_cfg.get("selected_model", "None")
llm_status     = runtime_cfg.get("llm_status", "NOT_READY")

avg_score      = summary.get("average_evidence_score", 0.0)
class_scores   = summary.get("class_scores", {})
class_match    = summary.get("scenario_class_match_rate", 0.0)
unsupported    = summary.get("unsupported_claims_total", 0)
pkt_claims     = summary.get("packet_claims_total", 0)
ft_claims      = summary.get("field_telemetry_claims", 0)
att_claims     = summary.get("external_attacker_claims", 0)
old_asset      = summary.get("old_asset_name_uses", 0)
parse_ok_count = summary.get("parse_success_count", 0)
total_scored   = summary.get("total_scored", 0)

exp_types = {}
if len(parsed_df) > 0 and "explanation_type" in parsed_df.columns:
    exp_types = parsed_df["explanation_type"].value_counts().to_dict()

# ── PHASE3_SMOKE_EXPLANATION_REPORT.md ────────────────────────────────────────
smoke_report = [
    "# PHASE 3 SMOKE EXPLANATION REPORT",
    "",
    f"Generated: {now_str}",
    "",
    "## Important: This Is a Smoke Run",
    "",
    "This report covers the **Phase 3 smoke-test explanation pipeline**.",
    "The pipeline uses synthetic smoke model detection scores to exercise the full",
    "grounded explanation framework before real frozen-model evaluation results",
    "become available.",
    "",
    "| Component | Status |",
    "|---|---|",
    "| Model detection scores | **SMOKE** (synthetic; see note below) |",
    "| Physical evidence | **REAL** (from zero_day_physical_attacked.csv) |",
    "| Cyber evidence | **REAL** (from zero_day_cyber_physical_aligned_1s.csv) |",
    "| Context evidence | **REAL** (from zero_day_cyber_event_log.csv) |",
    "| Scenario metadata | **REAL** (from zero_day_scenario_manifest.csv) |",
    "| LLM used for | **Explanation only** (not detection) |",
    "",
    "**Smoke detection note:** Only `model_name`, `anomaly_score`, `threshold`, and",
    "`predicted_label` are synthetic. All window IDs, scenario IDs, and scenario",
    "metadata come from real Phase 2 zero-day window data.",
    "",
    "## Replacing Smoke with Real Detections",
    "",
    "When Phase 1 frozen-model evaluation is complete:",
    "",
    "1. Set `detection_input_mode: real` in `phase3_explanation_config.json`.",
    "2. Ensure `zero_day_model_scores.csv` exists at `future_real_detection_file` path.",
    "3. Re-run `run_phase3_smoke_explanations.py` — the evidence pipeline is unchanged.",
    "",
    "## Ollama / Qwen Configuration",
    "",
    f"| Item | Value |",
    f"|---|---|",
    f"| LLM status | {llm_status} |",
    f"| Selected model | {selected_model} |",
    f"| Ollama host | {cfg.get('ollama_host', 'http://localhost:11434')} |",
    f"| Temperature | {cfg.get('temperature', 0.1)} |",
    "",
    "## Evidence Matrix Score Summary",
    "",
    f"| Metric | Value |",
    f"|---|---|",
    f"| Total scored | {total_scored} |",
    f"| Parse success | {parse_ok_count} |",
    f"| Average evidence score | {avg_score:.3f} |",
    f"| Scenario class match rate | {class_match:.3f} |",
    f"| Unsupported claims | {unsupported} |",
    f"| Packet-level claims | {pkt_claims} |",
    f"| Field telemetry claims | {ft_claims} |",
    f"| External attacker claims | {att_claims} |",
    "",
    "## Score by Scenario Class",
    "",
    "| Class | Avg Score |",
    "|---|---|",
]
for cls, score in class_scores.items():
    score_str = f"{score:.3f}" if score is not None else "N/A"
    smoke_report.append(f"| {cls} | {score_str} |")

# Example good/bad explanations from parsed_df
smoke_report += ["", "## Explanation Examples", ""]
if len(parsed_df) > 0 and "parse_status" in parsed_df.columns:
    good = parsed_df[parsed_df["parse_status"].isin(["OK","OK_REPAIRED"])].head(2)
    bad  = parsed_df[~parsed_df["parse_status"].isin(["OK","OK_REPAIRED"])].head(1)
    if len(good) > 0:
        smoke_report.append("### Good Explanation Examples")
        smoke_report.append("")
        for _, r in good.iterrows():
            smoke_report.append(f"**{r.get('detection_id','?')}** ({r.get('scenario_id','?')}, class={r.get('explanation_type','?')}):")
            smoke_report.append(f"> {str(r.get('human_explanation',''))[:300]}")
            smoke_report.append("")
    if len(bad) > 0:
        smoke_report.append("### Failed / Skipped Examples")
        smoke_report.append("")
        for _, r in bad.iterrows():
            smoke_report.append(f"**{r.get('detection_id','?')}**: parse_status={r.get('parse_status','?')}")
            smoke_report.append("")
else:
    smoke_report.append("No LLM explanations generated (LLM not ready or not run yet).")

smoke_report_path = RPT_DIR / "PHASE3_SMOKE_EXPLANATION_REPORT.md"
with open(smoke_report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(smoke_report))
print(f"[INFO] Wrote {smoke_report_path}")

# ── PHASE3_GROUNDEDNESS_REPORT.md ─────────────────────────────────────────────
ground_report = [
    "# PHASE 3 GROUNDEDNESS REPORT",
    "",
    f"Generated: {now_str}",
    "",
    "## What Makes This Pipeline Grounded",
    "",
    "The Phase 3 explanation pipeline is designed to produce evidence-grounded explanations,",
    "not free-form LLM guessing. The following mechanisms enforce groundedness:",
    "",
    "### 1. Field Glossary",
    "",
    "A dedicated glossary (`prompts/field_glossary.md`) defines every field the LLM may",
    "reference. The LLM is instructed to cite actual field names from the glossary.",
    "Field usage is audited post-hoc: `glossary_field_usage_correct` and",
    "`explanation_mentions_evidence_fields` are scored per explanation.",
    "",
    "### 2. Few-Shot Examples",
    "",
    "Five grounded few-shot examples (`prompts/few_shot_grounded_examples.md`) demonstrate",
    "correct output format for each explanation type:",
    "- `physical_only`: physical deviation with no cyber flags",
    "- `cyber_only`: cyber flags with no physical change",
    "- `cyber_physical`: cyber event timing aligned with physical response",
    "- `normal`: no anomaly flags active",
    "- `insufficient_evidence`: contradictory or missing evidence",
    "",
    "### 3. Expected-vs-Actual Evidence",
    "",
    "Each evidence packet includes a pre-computed `expected_vs_actual` block that",
    "summarises what the scenario schema says should happen (expected_behavior_summary)",
    "versus what the physical/cyber data shows in the detection window.",
    "This guides the LLM toward a structured comparison rather than free association.",
    "",
    "### 4. Hallucination Guardrails",
    "",
    "The prompt explicitly forbids:",
    "- Packet-level protocol claims (IEEE 2030.5 frame contents, MQTT payloads, etc.)",
    "- Real field telemetry claims (dataset is synthetic simulation data)",
    "- External attacker identity (no hacker/malware/exploit attribution without evidence)",
    "- Old asset names (PV35, PV60, PV83, BESS48, BESS108 do not exist in this dataset)",
    "",
    "### 5. Unsupported Claim Penalties",
    "",
    "The scoring script (`04_score_explanations.py`) independently scans explanation text",
    "using regex patterns to detect packet-level, telemetry, and attacker claims, even if",
    "the LLM self-reported them as false. Each violation deducts 0.15 from the overall",
    "evidence score. Violations are reported separately in the evidence matrix.",
    "",
    "### 6. Packet / Telemetry Claim Checks",
    "",
    "| Check | Pattern | Penalty |",
    "|---|---|---|",
    "| Packet-level claim | 'packet', 'IEEE 2030.5', 'DNP3', 'Modbus', 'payload', 'frame' | -0.15 |",
    "| Field telemetry claim | 'real field', 'live measurement', 'actual telemetry' | -0.15 |",
    "| External attacker claim | 'hacker', 'malware', 'exploit', 'threat actor', 'APT' | -0.15 |",
    "| Old asset name | 'PV35', 'PV60', 'PV83', 'BESS48', 'BESS108' | -0.15 |",
    "| Unsupported claim (self-reported) | LLM sets unsupported_claims_made=true | -0.15 |",
    "",
    "### 7. Insufficient Evidence Path",
    "",
    "`explanation_type = insufficient_evidence` is a first-class output type.",
    "The LLM is explicitly trained (via few-shot examples) to use it when evidence",
    "is missing, contradictory, or ambiguous. This prevents forced classification.",
    "",
    "## Claim Boundaries",
    "",
    "This dataset is synthetic simulation data. All cyber context is event-level only.",
    "No packet captures, no byte-level protocol traces, no real field telemetry exist.",
    "These boundaries are stated in the prompt, the field glossary, and the guardrails block.",
]

ground_path = RPT_DIR / "PHASE3_GROUNDEDNESS_REPORT.md"
with open(ground_path, "w", encoding="utf-8") as f:
    f.write("\n".join(ground_report))
print(f"[INFO] Wrote {ground_path}")

# ── PHASE3_COMPLETION_VERDICT.md ──────────────────────────────────────────────
# Determine verdict
pipeline_ran     = (OUT_DIR / "explanation_inputs.jsonl").exists()
smoke_det_exists = (ROOT / "inputs" / "smoke_model_detections.csv").exists()
llm_ran          = (OUT_DIR / "llm_explanations_raw.jsonl").exists()
scoring_ran      = (OUT_DIR / "explanation_evidence_matrix.csv").exists()

if not pipeline_ran or not smoke_det_exists:
    verdict = "PHASE3_FAILED_NOT_READY"
    verdict_reason = "Smoke detections or explanation inputs not generated."
elif llm_status not in ("READY", "READY_BUT_TEST_UNEXPECTED", "READY_BUT_JSON_PARSE_FAILED"):
    verdict = "PHASE3_SMOKE_READY_WAITING_FOR_REAL_MODEL_RESULTS"
    verdict_reason = (
        "Smoke detection pipeline and evidence packet pipeline completed successfully. "
        "LLM (Ollama/Qwen) was not available; explanations could not be generated. "
        "Waiting for: (1) Ollama/Qwen setup, (2) real frozen-model detection results."
    )
else:
    verdict = "PHASE3_SMOKE_READY_WAITING_FOR_REAL_MODEL_RESULTS"
    verdict_reason = (
        "Phase 3 smoke-test pipeline completed. Evidence packets, LLM explanations, "
        "and scoring all ran successfully. "
        "Waiting for: real frozen-model evaluation results from Phase 1/2."
    )

verdict_lines = [
    "# PHASE 3 COMPLETION VERDICT",
    "",
    f"Generated: {now_str}",
    "",
    f"## Verdict: {verdict}",
    "",
    f"{verdict_reason}",
    "",
    "## Pipeline Component Status",
    "",
    "| Component | Status |",
    "|---|---|",
    f"| Folder structure | {'OK' if (ROOT / 'scripts').exists() else 'MISSING'} |",
    f"| Config | {'OK' if CFG_PATH.exists() else 'MISSING'} |",
    f"| Ollama/Qwen check | {'OK' if (OUT_DIR / 'ollama_qwen_check.json').exists() else 'NOT_RUN'} |",
    f"| LLM status | {llm_status} |",
    f"| Field glossary | {'OK' if (ROOT / 'prompts' / 'field_glossary.md').exists() else 'MISSING'} |",
    f"| Few-shot examples | {'OK' if (ROOT / 'prompts' / 'few_shot_grounded_examples.md').exists() else 'MISSING'} |",
    f"| Smoke detections | {'OK' if smoke_det_exists else 'MISSING'} |",
    f"| Explanation inputs | {'OK' if pipeline_ran else 'MISSING'} |",
    f"| LLM explanations | {'OK' if llm_ran else 'NOT_RUN'} |",
    f"| Scoring | {'OK' if scoring_ran else 'NOT_RUN'} |",
    "",
    "## Next Steps",
    "",
    "1. If LLM not ready: install Ollama, run `ollama serve`, pull `qwen2.5:3b-instruct`.",
    "2. Run Phase 1 model training/evaluation to get `zero_day_model_scores.csv`.",
    "3. Set `detection_input_mode: real` in config and re-run.",
    "4. Final verdict will update to `PHASE3_COMPLETE_VERIFIED` when both steps complete.",
    ""
]

verdict_path = RPT_DIR / "PHASE3_COMPLETION_VERDICT.md"
with open(verdict_path, "w", encoding="utf-8") as f:
    f.write("\n".join(verdict_lines))
print(f"[INFO] Wrote {verdict_path}")

# ── README_PHASE3_EXPLANATIONS.md ──────────────────────────────────────────────
readme_lines = [
    "# Phase 3 Grounded Explanation Pipeline",
    "",
    f"Last updated: {now_str}",
    "",
    "## Overview",
    "",
    "This folder contains the Phase 3 grounded explanation pipeline for the DER",
    "cyber-physical anomaly detection benchmark. The pipeline takes anomaly detection",
    "results and generates grounded, evidence-based explanations using a local LLM",
    "(Ollama/Qwen).",
    "",
    "## Current Status",
    "",
    f"**Verdict: {verdict}**",
    "",
    "## Folder Structure",
    "",
    "```",
    "phase3_explanations/",
    "  configs/            # Configuration files",
    "  inputs/             # Smoke detection inputs",
    "  prompts/            # Prompt templates, glossary, few-shot examples",
    "  scripts/            # Processing scripts (run in order 00-06)",
    "  outputs/            # Generated evidence packets, LLM outputs, scores",
    "  reports/            # Markdown reports",
    "  figures/            # Matplotlib figures (300 dpi)",
    "  docs/               # Documentation",
    "  logs/               # Run logs",
    "```",
    "",
    "## Script Execution Order",
    "",
    "| Script | Purpose |",
    "|---|---|",
    "| `00_check_ollama_qwen.py` | Check Ollama/Qwen availability, pull if needed |",
    "| `01_create_smoke_detections.py` | Create synthetic model detection inputs |",
    "| `02_build_explanation_inputs.py` | Build grounded evidence packets |",
    "| `03_run_llm_explanations.py` | Call Ollama/Qwen for explanations |",
    "| `04_score_explanations.py` | Score explanations against evidence |",
    "| `05_generate_phase3_figures.py` | Generate matplotlib figures |",
    "| `06_generate_phase3_reports.py` | Generate final reports |",
    "",
    "Or simply run: `python run_phase3_smoke_explanations.py`",
    "",
    "## Requirements",
    "",
    "- Python 3.10+",
    "- `pandas`, `numpy`, `matplotlib`, `pyarrow`",
    "- Ollama server running locally with `qwen2.5:3b-instruct` (or fallback 1.5b)",
    "",
    "## Grounding Principles",
    "",
    "All physical/cyber/context evidence comes from real Phase 2 generated files.",
    "Only anomaly_score, threshold, predicted_label, model_name are synthetic smoke.",
    "See `reports/PHASE3_GROUNDEDNESS_REPORT.md` for full details.",
    "",
    "## Claim Boundaries",
    "",
    "- Event-level cyber context only",
    "- No packet-level protocol traces",
    "- No real field telemetry claims",
    "- No external attacker attribution",
    "- No old asset names (PV35/BESS48/etc.)",
    ""
]

readme_path = DOCS_DIR / "README_PHASE3_EXPLANATIONS.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))
print(f"[INFO] Wrote {readme_path}")

print(f"[DONE] All Phase 3 reports generated. Verdict: {verdict}")
