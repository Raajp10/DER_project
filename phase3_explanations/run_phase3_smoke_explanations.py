"""
Phase 3 Master Runner: Smoke Explanation Pipeline
Runs all Phase 3 steps in order and prints final status.
"""

import json
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
OUT_DIR     = ROOT / "outputs"
RPT_DIR     = ROOT / "reports"
LOG_DIR     = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_PATH = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_step(name, script_path):
    log(f"\n{'='*60}")
    log(f"STEP: {name}")
    log(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        log(f"[WARN] {name} exited with code {result.returncode}")
        return False
    return True

start_time = datetime.now()
log(f"Phase 3 Smoke Explanation Pipeline")
log(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Root: {ROOT}")

# ── Step 1: Ollama/Qwen check ─────────────────────────────────────────────────
run_step("Check Ollama/Qwen", SCRIPTS_DIR / "00_check_ollama_qwen.py")

# ── Step 2: Create smoke detections ──────────────────────────────────────────
ok2 = run_step("Create Smoke Detections", SCRIPTS_DIR / "01_create_smoke_detections.py")

# ── Step 3: Build explanation inputs ─────────────────────────────────────────
ok3 = run_step("Build Explanation Inputs", SCRIPTS_DIR / "02_build_explanation_inputs.py")

# ── Step 4: Run LLM explanations ─────────────────────────────────────────────
# Always attempt — script handles LLM-not-ready gracefully
run_step("Run LLM Explanations", SCRIPTS_DIR / "03_run_llm_explanations.py")

# ── Step 5: Score explanations ────────────────────────────────────────────────
run_step("Score Explanations", SCRIPTS_DIR / "04_score_explanations.py")

# ── Step 6: Generate figures ──────────────────────────────────────────────────
run_step("Generate Figures", SCRIPTS_DIR / "05_generate_phase3_figures.py")

# ── Step 7: Generate reports ──────────────────────────────────────────────────
run_step("Generate Reports", SCRIPTS_DIR / "06_generate_phase3_reports.py")

# ── Step 8: Build ZIP package ─────────────────────────────────────────────────
log("\n" + "="*60)
log("STEP: Build ZIP Package")
log("="*60)
zip_path = ROOT / "phase3_smoke_explanation_package.zip"
INCLUDE_DIRS = ["configs", "inputs", "prompts", "scripts", "outputs",
                "reports", "figures", "docs", "logs"]
INCLUDE_FILES = ["run_phase3_smoke_explanations.py", "run_phase3_smoke_explanations.bat"]

try:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in INCLUDE_DIRS:
            dir_path = ROOT / d
            if dir_path.exists():
                for fpath in dir_path.rglob("*"):
                    if fpath.is_file():
                        arcname = fpath.relative_to(ROOT)
                        zf.write(fpath, arcname)
        for fname in INCLUDE_FILES:
            fpath = ROOT / fname
            if fpath.exists():
                zf.write(fpath, fname)
    zip_size_kb = zip_path.stat().st_size // 1024
    log(f"[OK] Package: {zip_path} ({zip_size_kb} KB)")
except Exception as e:
    log(f"[WARN] ZIP failed: {e}")
    zip_path = None

# ── Load final stats ──────────────────────────────────────────────────────────
def try_load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

runtime_cfg  = try_load_json(ROOT / "configs" / "runtime_selected_model.json")
ollama_check = try_load_json(OUT_DIR / "ollama_qwen_check.json")
summary      = try_load_json(OUT_DIR / "explanation_score_summary.json")

ollama_ready   = ollama_check.get("ollama_server_running", False)
llm_status     = runtime_cfg.get("llm_status", "NOT_READY")
selected_model = runtime_cfg.get("selected_model", "None")

smoke_det_path = ROOT / "inputs" / "smoke_model_detections.csv"
exp_input_path = OUT_DIR / "explanation_inputs.jsonl"
raw_exp_path   = OUT_DIR / "llm_explanations_raw.jsonl"
parsed_exp_path = OUT_DIR / "llm_explanations_parsed.csv"
matrix_path    = OUT_DIR / "explanation_evidence_matrix.csv"
verdict_path   = RPT_DIR / "PHASE3_COMPLETION_VERDICT.md"

# Count rows
def count_file_rows(path, is_jsonl=False):
    p = Path(path)
    if not p.exists():
        return 0
    try:
        if is_jsonl:
            return sum(1 for l in open(p, encoding="utf-8") if l.strip())
        import pandas as pd
        return len(pd.read_csv(p))
    except Exception:
        return -1

smoke_count   = count_file_rows(smoke_det_path)
inputs_count  = count_file_rows(exp_input_path, is_jsonl=True)
raw_count     = count_file_rows(raw_exp_path, is_jsonl=True)
parsed_count  = count_file_rows(parsed_exp_path)
matrix_count  = count_file_rows(matrix_path)

avg_score     = summary.get("average_evidence_score", 0.0)
class_match   = summary.get("scenario_class_match_rate", 0.0)
unsupported   = summary.get("unsupported_claims_total", 0)
pkt_claims    = summary.get("packet_claims_total", 0)
ft_claims     = summary.get("field_telemetry_claims", 0)
att_claims    = summary.get("external_attacker_claims", 0)
old_asset     = summary.get("old_asset_name_uses", 0)

# Read verdict from file
phase3_status = "UNKNOWN"
if verdict_path.exists():
    for line in verdict_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Verdict:"):
            phase3_status = line.replace("## Verdict:", "").strip()
            break

end_time = datetime.now()
elapsed  = (end_time - start_time).total_seconds()

# ── Final print ───────────────────────────────────────────────────────────────
DIVIDER = "=" * 60
log(f"\n{DIVIDER}")
log("PHASE 3 SMOKE EXPLANATION PIPELINE - FINAL STATUS")
log(DIVIDER)
log(f"Ollama ready             : {'yes' if ollama_ready else 'no'}")
log(f"Qwen model selected      : {selected_model}")
log(f"smoke detections created : {smoke_count}")
log(f"explanation inputs built : {inputs_count}")
log(f"LLM explanations generated: {raw_count}")
log(f"parsed explanations      : {parsed_count}")
log(f"evidence matrix rows     : {matrix_count}")
log(f"average evidence score   : {avg_score:.3f}")
log(f"unsupported claims       : {unsupported}")
log(f"packet-level claims      : {pkt_claims}")
log(f"field telemetry claims   : {ft_claims}")
log(f"external attacker claims : {att_claims}")
log(f"old asset name claims    : {old_asset}")
log(f"Phase 3 status           : {phase3_status}")
log(f"final report path        : {RPT_DIR}")
log(f"package path             : {zip_path or 'NOT_CREATED'}")
log(f"elapsed                  : {elapsed:.1f}s")
log(DIVIDER)
