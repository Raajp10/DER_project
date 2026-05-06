"""
Phase 3 Step 1: Create smoke model detections from real zero-day windows.
Only the model_name, anomaly_score, threshold, predicted_label are synthetic.
All window/scenario metadata comes from real Phase 2 parquet file.
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "phase3_explanation_config.json"
INPUTS_DIR = ROOT / "inputs"
RPT_DIR = ROOT / "reports"
INPUTS_DIR.mkdir(exist_ok=True)
RPT_DIR.mkdir(exist_ok=True)

with open(CFG_PATH) as f:
    cfg = json.load(f)

WINDOWS_PATH = Path(cfg["zero_day_windows"])
MAX_DETECTIONS = cfg.get("smoke_detection_count", 40)
SMOKE_THRESHOLD = 0.5

print("[INFO] Loading zero_day_windows.parquet ...")
df = pd.read_parquet(WINDOWS_PATH)
print(f"  Loaded {len(df)} windows, columns: {list(df.columns[:10])} ...")

# ── Identify label columns ────────────────────────────────────────────────────
class_col = None
for c in ["zero_day_scenario_class", "scenario_class"]:
    if c in df.columns:
        class_col = c
        break

family_col = None
for c in ["zero_day_scenario_family", "scenario_family"]:
    if c in df.columns:
        family_col = c
        break

sid_col = None
for c in ["zero_day_scenario_id", "scenario_id"]:
    if c in df.columns:
        sid_col = c
        break

label_col = None
for c in ["zero_day_label_anomaly", "label_anomaly"]:
    if c in df.columns:
        label_col = c
        break

# Window time columns
wstart_col = None
for c in ["window_start_s", "start_s"]:
    if c in df.columns:
        wstart_col = c
        break

wend_col = None
for c in ["window_end_s", "end_s"]:
    if c in df.columns:
        wend_col = c
        break

wid_col = None
for c in ["window_id"]:
    if c in df.columns:
        wid_col = c
        break

print(f"  Class col: {class_col}, Family col: {family_col}, SID col: {sid_col}")
print(f"  Label col: {label_col}, Window start: {wstart_col}, Window ID: {wid_col}")

# ── Classify windows ──────────────────────────────────────────────────────────
def get_class(row):
    if class_col and pd.notna(row.get(class_col, None)):
        return str(row[class_col])
    if label_col:
        if row.get(label_col, 0) == 0:
            return "normal"
    return "unknown"

df["_class"] = df.apply(get_class, axis=1)
df["_family"] = df[family_col].fillna("unknown") if family_col else "unknown"
df["_sid"] = df[sid_col].fillna("none") if sid_col else "none"

class_counts = df["_class"].value_counts()
print(f"  Window classes: {class_counts.to_dict()}")

# ── Per-class budget ──────────────────────────────────────────────────────────
# Target: ~40 total, balanced across classes
# Layout: 12 cyber_physical, 10 physical_only, 10 cyber_only, 6 normal, 2 false_alert
BUDGET = {
    "cyber_physical": 5,
    "physical_only":  4,
    "cyber_only":     4,
    "normal":         2,
}
FALSE_ALERT_COUNT = 0  # skip false alerts to keep total at 15

def sample_class(cls, n):
    sub = df[df["_class"] == cls]
    if len(sub) == 0:
        print(f"  [WARN] No windows found for class {cls}")
        return pd.DataFrame()
    n = min(n, len(sub))
    return sub.sample(n, random_state=42)

sampled_parts = []
for cls, n in BUDGET.items():
    part = sample_class(cls, n)
    if len(part) > 0:
        part = part.copy()
        part["_budget_class"] = cls
        sampled_parts.append(part)

# False alert examples: normal windows predicted as anomaly
normal_windows = df[df["_class"] == "normal"]
if len(normal_windows) >= FALSE_ALERT_COUNT:
    # Pick ones not already sampled
    already_sampled_idx = set()
    for p in sampled_parts:
        already_sampled_idx.update(p.index.tolist())
    remaining_normal = normal_windows[~normal_windows.index.isin(already_sampled_idx)]
    if len(remaining_normal) >= FALSE_ALERT_COUNT:
        fa_part = remaining_normal.sample(FALSE_ALERT_COUNT, random_state=43).copy()
        fa_part["_budget_class"] = "normal_false_alert"
        sampled_parts.append(fa_part)

all_sampled = pd.concat(sampled_parts, ignore_index=True)
print(f"  Total sampled windows: {len(all_sampled)}")

# ── Build smoke detections ────────────────────────────────────────────────────
records = []
det_id_counter = 1

for _, row in all_sampled.iterrows():
    det_class = row["_budget_class"]
    is_false_alert = det_class == "normal_false_alert"
    true_class = row["_class"]

    # Smoke anomaly score
    if true_class in ("cyber_physical", "physical_only", "cyber_only") or is_false_alert:
        # Anomaly: score > threshold
        score = round(random.uniform(0.55, 0.95), 4)
        pred_label = 1
    else:
        # Normal: score < threshold
        score = round(random.uniform(0.05, 0.45), 4)
        pred_label = 0

    wid = row[wid_col] if wid_col else f"w_{det_id_counter:05d}"
    wstart = int(row[wstart_col]) if wstart_col else 0
    wend = int(row[wend_col]) if wend_col else wstart + 59
    sid = row["_sid"]
    family = row["_family"]

    rec = {
        "detection_id":       f"det_{det_id_counter:04d}",
        "model_name":         "smoke_detector",
        "window_id":          wid,
        "window_start_s":     wstart,
        "window_end_s":       wend,
        "scenario_id":        sid,
        "scenario_class":     true_class,
        "scenario_family":    family,
        "anomaly_score":      score,
        "threshold":          SMOKE_THRESHOLD,
        "predicted_label":    pred_label,
        "smoke_detection_only": True,
        "detection_input_mode": "smoke",
        "false_alert_smoke_example": is_false_alert,
    }
    records.append(rec)
    det_id_counter += 1

out_df = pd.DataFrame(records)
out_path = Path(cfg["smoke_detection_file"])
out_path.parent.mkdir(exist_ok=True, parents=True)
out_df.to_csv(out_path, index=False)
print(f"[INFO] Smoke detections saved: {out_path} ({len(out_df)} rows)")

# ── Report ────────────────────────────────────────────────────────────────────
class_summary = out_df["scenario_class"].value_counts()
pred_summary  = out_df["predicted_label"].value_counts()
fa_count = out_df["false_alert_smoke_example"].sum()

report = [
    "# SMOKE DETECTION INPUT REPORT",
    "",
    f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "## Purpose",
    "",
    "Smoke model detections are used to drive the Phase 3 grounded explanation pipeline",
    "before real frozen-model evaluation results are available.",
    "",
    "**Only the following fields are synthetic smoke values:**",
    "- `model_name` = `smoke_detector`",
    "- `anomaly_score` — randomly sampled above/below threshold",
    "- `threshold` = 0.5",
    "- `predicted_label` — derived from anomaly_score vs threshold",
    "",
    "**All other fields come from real Phase 2 zero_day_windows.parquet:**",
    "- `window_id`, `window_start_s`, `window_end_s`",
    "- `scenario_id`, `scenario_class`, `scenario_family`",
    "",
    "## Detection Summary",
    "",
    f"| Field | Value |",
    f"|---|---|",
    f"| Total detections | {len(out_df)} |",
    f"| Source | zero_day_windows.parquet |",
    f"| False alert smoke examples | {fa_count} |",
    f"| detection_input_mode | smoke |",
    "",
    "## Class Distribution",
    "",
    "| scenario_class | count |",
    "|---|---|",
]
for cls, cnt in class_summary.items():
    report.append(f"| {cls} | {cnt} |")
report += [
    "",
    "## Predicted Label Distribution",
    "",
    "| predicted_label | count |",
    "|---|---|",
]
for lbl, cnt in pred_summary.items():
    report.append(f"| {lbl} | {cnt} |")
report += [
    "",
    "## Notes",
    "",
    "- When real frozen-model results become available, replace `detection_input_mode: smoke`",
    "  inputs with outputs from `zero_day_model_scores.csv`.",
    "- The physical/cyber/context evidence pipeline is identical for both smoke and real detections.",
    ""
]

rpt_path = RPT_DIR / "SMOKE_DETECTION_INPUT_REPORT.md"
with open(rpt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"[INFO] Report saved: {rpt_path}")
print(f"[DONE] {len(out_df)} smoke detections written.")
