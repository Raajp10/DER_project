# DER Cyber-Physical Anomaly Detection Dataset — Updated Version

## Dataset Purpose

This dataset supports research in **cyber-physical anomaly detection for distributed energy resources (DERs)** connected to the IEEE 123-bus distribution feeder. It provides:

- 7-day, 1-second resolution physical DER timeseries (clean and attacked)
- IEEE 2030.5-style semantic cyber event logs (normal and anomalous)
- Event-specific context windows linking cyber commands to physical responses
- Labeled scenarios covering 11 anomaly classes

## Exact Final Claim

> "The updated dataset combines OpenDSS-based IEEE 123-bus clean DER simulation outputs with OpenDSS event-window-resolved or physics-constrained anomalous DER responses, IEEE 1547-informed DER operating constraints, and an IEEE 2030.5-style semantic cyber event lifecycle representing DER control, status, metering, and response behavior. Cyber and physical layers are aligned through event-specific command-to-response context windows for cyber-physical anomaly detection."

## How to Run the Pipeline

### Prerequisites

```bash
pip install pandas numpy matplotlib scipy pyyaml
# Optional for OpenDSS simulation:
pip install opendssdirect.py
```

### Run

```bat
cd D:\DER_Project_update
python scripts_updated\run_updated_dataset_pipeline.py
```

Or double-click `run_pipeline.bat`.

## Folder Structure

```
D:\DER_Project_update\
├── ieee123_base\               ← OpenDSS source files (IEEE 123-bus feeder + DER)
├── original_input_backup\      ← Backup of source files (do not modify)
├── scripts_updated\            ← All pipeline scripts
│   ├── 00_common\              ← Shared utilities (paths, config, time, validation)
│   ├── 01_setup\               ← Environment discovery and check
│   ├── 02_scenario_generation\ ← Scenario manifest generator
│   ├── 03_physical_data_generation\ ← Physical timeseries generation
│   ├── 04_cyber_log_generation\ ← IEEE 2030.5-style cyber event log
│   ├── 05_physical_cyber_mapping\ ← Context window and evidence packet builder
│   ├── 06_visualization\       ← Figure generators
│   ├── 07_validation_gate\     ← Validation scripts
│   └── run_updated_dataset_pipeline.py ← Master pipeline
├── data_updated\               ← All generated outputs
│   ├── raw\                    ← Physical timeseries and cyber logs
│   ├── processed\              ← Lifecycle maps, context windows, evidence packets
│   ├── scenarios\              ← Scenario manifest CSV and JSON
│   ├── metadata\               ← DER ratings, discovery report, schema
│   └── validation\             ← All validation reports and summaries
├── reports\                    ← Narrative reports (00–09)
├── figures\                    ← Generated PNG figures (≥300 DPI)
├── logs\                       ← Pipeline run log and error log
├── final_package\              ← ZIP archive of full dataset
├── run_pipeline.bat            ← Windows batch runner
└── README_UPDATED_DATASET.md   ← This file
```

## Generated Files

### Physical Data

| File | Description | Rows |
|------|-------------|------|
| `data_updated/raw/physical_timeseries_clean_improved_7d.csv` | Clean 7-day DER timeseries | 604,800 |
| `data_updated/raw/physical_timeseries_attacked_improved_7d.csv` | Attacked timeseries with scenario effects | 604,800 |
| `data_updated/raw/physical_residuals_improved_7d.csv` | Residuals (attacked − clean) | 604,800 |

**Key physical columns:** `timestamp_utc`, `time_s`, `pv_actual_p_kw`, `pv_available_kw`, `bess_actual_p_kw`, `bess_soc_percent`, `pcc_v_a_pu`, `pcc_v_b_pu`, `pcc_v_c_pu`, `pcc_voltage_unbalance_pu`, `physical_effect_active_flag`, `generation_method`

### Cyber Log

| File | Description |
|------|-------------|
| `data_updated/raw/cyber_event_log_normal_ieee2030_5_semantic_7d.csv` | Routine DER control, metering, status events |
| `data_updated/raw/cyber_event_log_anomalous_ieee2030_5_semantic_7d.csv` | Anomalous events for all 180 scenarios |

**Key cyber columns:** `lifecycle_stage`, `protocol_claim_level`, `label_anomaly`, `label_cyber_anomaly`, `blocked_flag`, `replay_flag`, `mismatch_flag`, `cia_dimension`, `ieee2030_5_resource_type`

### Context Mapping

| File | Description |
|------|-------------|
| `data_updated/processed/cyber_physical_lifecycle_map_7d.csv` | Per-scenario timing map |
| `data_updated/processed/event_specific_context_windows_7d.csv` | Physical+cyber rows per scenario window |
| `data_updated/processed/cyber_physical_evidence_packets_7d.jsonl` | JSON evidence packets per scenario |

## Physical Data Explanation

The physical layer models a 100 kW PV + 50 kW / 200 kWh BESS site at Bus 65 of the IEEE 123-bus feeder. Clean data uses OpenDSS QSTS simulation when available, or a physics-constrained surrogate model. Attacked data applies scenario-specific physical effects (irradiance drops, load steps, wrong setpoints, etc.) in event windows and recomputes BESS SOC from scratch.

**DER Ratings (from DSS files):**
- PV: 100 kW, 111.11 kVA
- BESS: 50 kW, 200 kWh, SOC 10–90%, initial SOC 50%

## Cyber Data Explanation

The cyber log models the IEEE 2030.5 DER management protocol lifecycle at the **semantic level**. Each command follows: CREATED → SENT → RECEIVED → ACCEPTED → APPLIED → RESPONSE. Attack scenarios inject appropriate anomaly flags. 

**Critical:** `protocol_claim_level = semantic_ieee2030_5_style`. No official IEEE 2030.5 compliance is claimed. No EXI encoding or network packet captures are present.

## Context Mapping Explanation

Context windows (120s pre-event, 300s post-event) are built for each scenario. They include:
- Cyber timing (command_sent → command_apply)
- Physical timing (physical_effect_start → peak → end)
- Relative time offsets
- Physical residuals (delta from clean baseline)
- Constraint violation flags
- Evidence strength scores

## Validation Summary

Five validation gates run automatically:
1. Physical constraints (SOC, apparent power, voltage unbalance, timestamps)
2. Clean/attacked alignment (time_s exact match)
3. IEEE 2030.5 semantic layer (lifecycle stages, protocol claim level, flag logic)
4. Context causality (flag integrity, delayed-scenario timing)
5. Final gate (all outputs exist, all figures generated)

See `data_updated/validation/final_updated_dataset_validation_report.md`.

## Important Limitations

- **No real field telemetry** — all data is simulation/model-based
- **No official IEEE 2030.5 compliance** — semantic model only
- **No packet captures** — `protocol_claim_level` is never `packet_capture`
- **Surrogate fallback** — if OpenDSS is not installed, physics-constrained surrogate is used
- See `reports/09_remaining_limitations.md` for full details

## Citation / Use

If using this dataset, please cite the generation method and acknowledge:
- IEEE 123-Bus Test Feeder (Kersting, 1991)
- OpenDSS (EPRI)
- IEEE 1547-2018 (DER interconnection standard)
- IEEE 2030.5 (Smart Energy Profile — semantic model only)
