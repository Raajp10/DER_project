# Phase 3 Grounded Explanation Pipeline

Last updated: 2026-05-06 06:11 UTC

## Overview

This folder contains the Phase 3 grounded explanation pipeline for the DER
cyber-physical anomaly detection benchmark. The pipeline takes anomaly detection
results and generates grounded, evidence-based explanations using a local LLM
(Ollama/Qwen).

## Current Status

**Verdict: PHASE3_SMOKE_READY_WAITING_FOR_REAL_MODEL_RESULTS**

## Folder Structure

```
phase3_explanations/
  configs/            # Configuration files
  inputs/             # Smoke detection inputs
  prompts/            # Prompt templates, glossary, few-shot examples
  scripts/            # Processing scripts (run in order 00-06)
  outputs/            # Generated evidence packets, LLM outputs, scores
  reports/            # Markdown reports
  figures/            # Matplotlib figures (300 dpi)
  docs/               # Documentation
  logs/               # Run logs
```

## Script Execution Order

| Script | Purpose |
|---|---|
| `00_check_ollama_qwen.py` | Check Ollama/Qwen availability, pull if needed |
| `01_create_smoke_detections.py` | Create synthetic model detection inputs |
| `02_build_explanation_inputs.py` | Build grounded evidence packets |
| `03_run_llm_explanations.py` | Call Ollama/Qwen for explanations |
| `04_score_explanations.py` | Score explanations against evidence |
| `05_generate_phase3_figures.py` | Generate matplotlib figures |
| `06_generate_phase3_reports.py` | Generate final reports |

Or simply run: `python run_phase3_smoke_explanations.py`

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `pyarrow`
- Ollama server running locally with `qwen2.5:3b-instruct` (or fallback 1.5b)

## Grounding Principles

All physical/cyber/context evidence comes from real Phase 2 generated files.
Only anomaly_score, threshold, predicted_label, model_name are synthetic smoke.
See `reports/PHASE3_GROUNDEDNESS_REPORT.md` for full details.

## Claim Boundaries

- Event-level cyber context only
- No packet-level protocol traces
- No real field telemetry claims
- No external attacker attribution
- No old asset names (PV35/BESS48/etc.)
