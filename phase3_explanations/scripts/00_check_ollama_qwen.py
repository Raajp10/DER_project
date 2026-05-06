"""
Phase 3 Step 0: Check Ollama installation and pull/verify Qwen model.
Saves selected model to configs/runtime_selected_model.json.
Writes ollama_qwen_check.json and OLLAMA_QWEN_SETUP_REPORT.md.
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "phase3_explanation_config.json"
OUT_DIR = ROOT / "outputs"
RPT_DIR = ROOT / "reports"
CFG_DIR = ROOT / "configs"
OUT_DIR.mkdir(exist_ok=True)
RPT_DIR.mkdir(exist_ok=True)
CFG_DIR.mkdir(exist_ok=True)

with open(CFG_PATH) as f:
    cfg = json.load(f)

OLLAMA_HOST   = cfg["ollama_host"]
MODEL_PRIMARY  = cfg["ollama_model_primary"]
MODEL_FALLBACK = cfg["ollama_model_fallback"]

result = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "ollama_installed": False,
    "ollama_server_running": False,
    "primary_model_available": False,
    "fallback_model_available": False,
    "selected_model": None,
    "model_test_passed": False,
    "llm_status": "NOT_READY",
    "model_test_response": None,
    "error_messages": [],
    "instructions_if_not_ready": []
}

# ── 1. Check Ollama binary ──────────────────────────────────────────────────
try:
    r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        result["ollama_installed"] = True
        print(f"[OK] Ollama installed: {r.stdout.strip()}")
    else:
        result["error_messages"].append("ollama --version returned non-zero")
except FileNotFoundError:
    result["error_messages"].append("ollama binary not found in PATH")
    print("[WARN] Ollama not found in PATH.")
except Exception as e:
    result["error_messages"].append(f"ollama check error: {e}")

if not result["ollama_installed"]:
    result["instructions_if_not_ready"] = [
        "1. Install Ollama from https://ollama.com/download",
        "2. Start the Ollama server: ollama serve",
        "3. Pull the model: ollama pull qwen2.5:3b-instruct",
        "4. Re-run this script."
    ]

# ── 2. Check server running ─────────────────────────────────────────────────
if result["ollama_installed"]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as resp:
            tags_data = json.loads(resp.read().decode())
        result["ollama_server_running"] = True
        available_models = [m["name"] for m in tags_data.get("models", [])]
        print(f"[OK] Ollama server running. Models: {available_models}")
    except Exception as e:
        result["error_messages"].append(f"Server not reachable: {e}")
        available_models = []
        print(f"[WARN] Ollama server not reachable: {e}")
        result["instructions_if_not_ready"] = [
            "1. Start the Ollama server: ollama serve",
            "2. Pull the model: ollama pull qwen2.5:3b-instruct",
            "3. Re-run this script."
        ]
else:
    available_models = []

# ── 3. Check / pull models ───────────────────────────────────────────────────
def model_exists(name, avail):
    return any(name in m for m in avail)

def pull_model(name):
    print(f"[INFO] Pulling {name} ...")
    r = subprocess.run(["ollama", "pull", name], timeout=600)
    return r.returncode == 0

if result["ollama_server_running"]:
    # Primary
    if model_exists(MODEL_PRIMARY, available_models):
        result["primary_model_available"] = True
        print(f"[OK] Primary model {MODEL_PRIMARY} found.")
    else:
        print(f"[INFO] {MODEL_PRIMARY} not found locally. Attempting pull...")
        try:
            ok = pull_model(MODEL_PRIMARY)
            result["primary_model_available"] = ok
            if ok:
                print(f"[OK] {MODEL_PRIMARY} pulled successfully.")
            else:
                result["error_messages"].append(f"Failed to pull {MODEL_PRIMARY}")
                print(f"[WARN] Failed to pull {MODEL_PRIMARY}.")
        except Exception as e:
            result["error_messages"].append(f"Pull error {MODEL_PRIMARY}: {e}")
            print(f"[WARN] Pull error: {e}")

    # Fallback
    if not result["primary_model_available"]:
        if model_exists(MODEL_FALLBACK, available_models):
            result["fallback_model_available"] = True
            print(f"[OK] Fallback model {MODEL_FALLBACK} found.")
        else:
            print(f"[INFO] {MODEL_FALLBACK} not found. Attempting pull...")
            try:
                ok = pull_model(MODEL_FALLBACK)
                result["fallback_model_available"] = ok
                if ok:
                    print(f"[OK] {MODEL_FALLBACK} pulled.")
                else:
                    result["error_messages"].append(f"Failed to pull {MODEL_FALLBACK}")
            except Exception as e:
                result["error_messages"].append(f"Pull error {MODEL_FALLBACK}: {e}")

# ── 4. Select model ──────────────────────────────────────────────────────────
if result["primary_model_available"]:
    result["selected_model"] = MODEL_PRIMARY
elif result["fallback_model_available"]:
    result["selected_model"] = MODEL_FALLBACK
else:
    result["selected_model"] = None

# ── 5. Test selected model ───────────────────────────────────────────────────
TEST_PROMPT = (
    'Return only valid JSON with one key "status" and value "ok". '
    'No explanation, no markdown. Just the JSON object.'
)

def call_ollama(model, prompt):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

if result["selected_model"] and result["ollama_server_running"]:
    try:
        print(f"[INFO] Testing model {result['selected_model']} ...")
        resp = call_ollama(result["selected_model"], TEST_PROMPT)
        raw = resp.get("response", "")
        result["model_test_response"] = raw[:500]
        # Try parsing
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw_clean)
        if parsed.get("status") == "ok":
            result["model_test_passed"] = True
            result["llm_status"] = "READY"
            print(f"[OK] Model test passed. Response: {parsed}")
        else:
            result["model_test_passed"] = False
            result["llm_status"] = "READY_BUT_TEST_UNEXPECTED"
            print(f"[WARN] Model responded but test output unexpected: {parsed}")
    except json.JSONDecodeError:
        result["model_test_passed"] = False
        result["llm_status"] = "READY_BUT_JSON_PARSE_FAILED"
        result["error_messages"].append("Model test JSON parse failed.")
        print(f"[WARN] Model test response not clean JSON: {result['model_test_response'][:200]}")
    except Exception as e:
        result["error_messages"].append(f"Model test error: {e}")
        result["llm_status"] = "READY_BUT_TEST_FAILED"
        print(f"[WARN] Model test error: {e}")
else:
    print("[WARN] LLM not ready. Explanations will be skipped.")
    if not result["instructions_if_not_ready"]:
        result["instructions_if_not_ready"] = [
            "1. Install Ollama: https://ollama.com/download",
            "2. Run: ollama serve",
            "3. Run: ollama pull qwen2.5:3b-instruct",
            "4. Re-run this script."
        ]

# ── 6. Save runtime selected model ──────────────────────────────────────────
runtime = {
    "selected_model": result["selected_model"],
    "llm_status": result["llm_status"],
    "timestamp": result["timestamp"]
}
runtime_path = CFG_DIR / "runtime_selected_model.json"
with open(runtime_path, "w") as f:
    json.dump(runtime, f, indent=2)
print(f"[INFO] Runtime model saved: {runtime_path}")

# ── 7. Save check JSON ───────────────────────────────────────────────────────
check_path = OUT_DIR / "ollama_qwen_check.json"
with open(check_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"[INFO] Check JSON saved: {check_path}")

# ── 8. Write markdown report ─────────────────────────────────────────────────
status_emoji = "PASS" if result["llm_status"] == "READY" else "WARN"
report_lines = [
    "# OLLAMA QWEN SETUP REPORT",
    "",
    f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "## Status Summary",
    "",
    f"| Check | Result |",
    f"|---|---|",
    f"| Ollama installed | {'Yes' if result['ollama_installed'] else 'No'} |",
    f"| Ollama server running | {'Yes' if result['ollama_server_running'] else 'No'} |",
    f"| Primary model ({MODEL_PRIMARY}) available | {'Yes' if result['primary_model_available'] else 'No'} |",
    f"| Fallback model ({MODEL_FALLBACK}) available | {'Yes' if result['fallback_model_available'] else 'No'} |",
    f"| Selected model | {result['selected_model'] or 'None'} |",
    f"| Model test passed | {'Yes' if result['model_test_passed'] else 'No'} |",
    f"| LLM status | **{result['llm_status']}** |",
    "",
]

if result["error_messages"]:
    report_lines += ["## Errors", ""]
    for e in result["error_messages"]:
        report_lines.append(f"- {e}")
    report_lines.append("")

if result["instructions_if_not_ready"]:
    report_lines += ["## Setup Instructions", ""]
    for i in result["instructions_if_not_ready"]:
        report_lines.append(i)
    report_lines.append("")

if result["model_test_response"]:
    report_lines += [
        "## Model Test Response",
        "",
        "```",
        result["model_test_response"][:300],
        "```",
        ""
    ]

report_lines += [
    "## Notes",
    "",
    "- All Phase 3 physical/cyber/context evidence comes from real Phase 2 generated files.",
    "- Only model detection scores (anomaly_score, threshold, predicted_label) are synthetic smoke values.",
    "- Ollama/Qwen is used for explanation generation only, not for anomaly detection.",
    ""
]

rpt_path = RPT_DIR / "OLLAMA_QWEN_SETUP_REPORT.md"
with open(rpt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"[INFO] Report saved: {rpt_path}")

print(f"\n[RESULT] LLM status: {result['llm_status']}")
print(f"[RESULT] Selected model: {result['selected_model']}")
