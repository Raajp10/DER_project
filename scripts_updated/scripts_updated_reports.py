"""
Generate all final reports (06–09) for the DER dataset.
Called from the master pipeline.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import (
    REPORTS, DER_METADATA_JSON, SCENARIO_MANIFEST_CSV, FINAL_VALIDATION_JSON,
    CYBER_ANOMALOUS_CSV, ATTACKED_PHYSICAL_CSV, ENV_CHECK_JSON, METADATA,
)
from validation_utils import write_report


def load_json(p: Path) -> dict:
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def generate_report_06_claim_support():
    meta = load_json(DER_METADATA_JSON)
    val = load_json(FINAL_VALIDATION_JSON)
    gen_counts = val.get("generation_method_counts", {})
    proto_counts = val.get("protocol_claim_level_counts", {})

    final_claim = val.get("final_claim", "See final validation report.")

    lines = [
        "# Report 06 — Final Claim Support",
        "",
        "## 1. What the Dataset CAN Claim",
        "",
        "- OpenDSS-based IEEE 123-bus feeder as simulation foundation",
        "- IEEE 1547-informed DER operating constraints (SOC, apparent power limits, ramp rates)",
        "- IEEE 2030.5-style semantic DER control lifecycle (NOT packet-level compliance)",
        "- Physics-consistent 7-day, 1-second resolution timeseries",
        "- Cyber-physical alignment via event-specific command-to-response context windows",
        "- Multiple labeled anomaly types (11 classes)",
        "- Honest generation_method labeling on every row",
        "",
        "## 2. What the Dataset CANNOT Claim",
        "",
        "- Real utility field telemetry",
        "- Official IEEE 2030.5 compliance (no certification)",
        "- Packet-level IEEE 2030.5 protocol compliance (no EXI, no packet capture)",
        "- 100% OpenDSS event-window resolution (if surrogate was used)",
        "- Real network traffic or endpoint data",
        "",
        "## 3. Exact Final Claim Sentence",
        "",
        f'> "{final_claim}"',
        "",
        "## 4. Supporting Files",
        "",
        "- `data_updated/raw/physical_timeseries_clean_improved_7d.csv`",
        "- `data_updated/raw/physical_timeseries_attacked_improved_7d.csv`",
        "- `data_updated/raw/physical_residuals_improved_7d.csv`",
        "- `data_updated/raw/cyber_event_log_normal_ieee2030_5_semantic_7d.csv`",
        "- `data_updated/raw/cyber_event_log_anomalous_ieee2030_5_semantic_7d.csv`",
        "- `data_updated/processed/cyber_physical_lifecycle_map_7d.csv`",
        "- `data_updated/processed/event_specific_context_windows_7d.csv`",
        "- `data_updated/processed/cyber_physical_evidence_packets_7d.jsonl`",
        "- `data_updated/metadata/der_physical_metadata.json`",
        "",
        "## 5. Validation Reports Supporting the Claim",
        "",
        "- `data_updated/validation/physical_constraints_report.md`",
        "- `data_updated/validation/ieee2030_5_semantic_validation_report.md`",
        "- `data_updated/validation/context_causality_validation_report.md`",
        "- `data_updated/validation/final_updated_dataset_validation_report.md`",
        "",
        "## 6. Generation Method Counts",
        "",
    ]
    for k, v in gen_counts.items():
        lines.append(f"- `{k}`: {v:,} rows")
    lines += [
        "",
        "## 7. Protocol Claim Level Counts",
        "",
    ]
    for k, v in proto_counts.items():
        lines.append(f"- `{k}`: {v:,} events")
    lines += [
        "",
        "## 8. Remaining Limitations",
        "",
        "See Report 09.",
    ]
    write_report(lines, REPORTS / "06_final_claim_support_report.md")


def generate_report_07_professor_explanation():
    lines = [
        "# Report 07 — Plain-Language Explanation for Review",
        "",
        "## How Physical Data Was Generated",
        "",
        "The physical data represents what would be measured at a DER installation site",
        "(Bus 65 of the IEEE 123-bus test feeder). The feeder is a well-known benchmark",
        "used in power systems research.",
        "",
        "**Step 1 — Clean baseline:** We generate 7 days of 1-second physical measurements",
        "representing normal DER operation. A solar irradiance model drives PV output.",
        "The BESS charges during mid-day solar hours and discharges in the evening.",
        "PCC voltages, currents, and powers are computed from physics equations.",
        "",
        "If OpenDSS was available, it would run the actual power flow simulation.",
        "If not, a physics-constrained surrogate model generates equivalent data.",
        "Every row is labeled with `generation_method` so readers know exactly",
        "which approach produced each data point.",
        "",
        "**Step 2 — Attacked dataset:** Starting from the clean baseline, scenario-specific",
        "physical effects are applied in event windows. For example, an irradiance drop",
        "scenario reduces solar output for a few minutes; a voltage sag scenario lowers",
        "the PCC voltage. The BESS SOC is recomputed from scratch after all modifications.",
        "",
        "## How Cyber Data Was Generated",
        "",
        "The cyber log models the IEEE 2030.5 DER management protocol lifecycle at a",
        "semantic level. IEEE 2030.5 defines how a utility DERMS (DER Management System)",
        "sends commands to DER devices using HTTP over TLS.",
        "",
        "For each scenario, we generate the sequence of events that would occur:",
        "- Command created at the DERMS",
        "- Command sent over the network",
        "- Command received by the DER",
        "- Command accepted and applied",
        "- DER sends back a response",
        "",
        "For attack scenarios, anomaly flags are set:",
        "- Unauthorized command: authz_status=failed, blocked_flag=1",
        "- Replay attack: replay_flag=1, previous_message_mrid points to stale message",
        "- Wrong setpoint: mismatch_flag=1, integrity_status=compromised",
        "- Delayed command: availability_status=degraded, total_delay_s much higher",
        "",
        "**IMPORTANT:** This is a SEMANTIC model only. No actual network packets were",
        "captured or generated. The `protocol_claim_level` column is always",
        "`semantic_ieee2030_5_style` to reflect this.",
        "",
        "## How Context Data Was Generated",
        "",
        "Context windows link cyber events to their physical effects. For each scenario,",
        "we extract a window of physical data (120s before, 300s after the event),",
        "compute residuals (attacked - clean), and record relative timing between:",
        "- When the cyber command was sent",
        "- When it was applied",
        "- When the physical effect started",
        "- When the physical effect peaked and ended",
        "",
        "This allows ML models to learn the causal relationship between DER commands",
        "and physical responses.",
        "",
        "## What `generation_method` Means",
        "",
        "| Value | Meaning |",
        "| --- | --- |",
        "| `opendss_clean_baseline` | OpenDSS QSTS simulation ran successfully |",
        "| `opendss_event_window_resolved` | OpenDSS solved local event window |",
        "| `physics_constrained_surrogate` | Python physics model (OpenDSS unavailable) |",
        "| `csv_rule_legacy` | Copied from prior dataset (reported as limitation) |",
        "",
        "## What `protocol_claim_level` Means",
        "",
        "| Value | Meaning |",
        "| --- | --- |",
        "| `semantic_ieee2030_5_style` | Semantic model only, no actual protocol bytes |",
        "| `serialized_ieee2030_5_xml` | Actual XML messages generated (research use) |",
        "| `packet_capture` | Real network packets (NOT present in this dataset) |",
        "",
        "## Why This Version Is Better Than the Original",
        "",
        "1. **Honest labeling:** Every row carries `generation_method` and `protocol_claim_level`",
        "   so readers know exactly what was simulated vs. modeled.",
        "2. **IEEE 1547-informed constraints:** PV and BESS operating limits are checked",
        "   and validated (apparent power, SOC, ramp rates).",
        "3. **SOC recomputed from scratch:** BESS SOC is never inherited from inconsistent",
        "   prior data — it is always recomputed from actual power and capacity.",
        "4. **No inflated ratings:** PV and BESS ratings come from the actual DSS files",
        "   (100 kW PV, 50 kW BESS), not generic placeholders.",
        "5. **Voltage unbalance tracked:** Three-phase voltage unbalance is computed",
        "   and validated at every timestep.",
        "6. **Cyber-physical alignment:** Context windows explicitly link command timing",
        "   to physical effect timing, including delay modeling.",
    ]
    write_report(lines, REPORTS / "07_professor_explanation_simple.md")


def generate_report_08_paper_method():
    lines = [
        "# Report 08 — IEEE Paper-Ready Method Text",
        "",
        "## Physical Simulation Layer",
        "",
        "The physical simulation layer is grounded in the IEEE 123-bus test feeder,",
        "a widely used benchmark for distribution system studies. A distributed energy",
        "resource (DER) site comprising a 100 kW photovoltaic (PV) system and a",
        "50 kW / 200 kWh battery energy storage system (BESS) is placed at Bus 65.",
        "The point of common coupling (PCC) is Line L64, terminal 2.",
        "",
        "A seven-day, one-second resolution timeseries (604,800 time steps) is generated",
        "as the clean baseline. When the OpenDSS simulation engine (opendssdirect.py) is",
        "available, the DER_QSTS_7day.dss script is compiled and solved in quasi-static",
        "time series (QSTS) mode. When OpenDSS is not available, a physics-constrained",
        "surrogate model generates equivalent data using a first-principles solar irradiance",
        "model, diurnal load profile, and simplified Thevenin voltage model. All rows are",
        "tagged with `generation_method` to distinguish simulation origin.",
        "",
        "## IEEE 1547-Informed DER Constraint Layer",
        "",
        "DER operating constraints follow IEEE 1547-2018. Validated constraints include:",
        "- PV apparent power limit: S_pv ≤ S_rated (111.11 kVA)",
        "- BESS apparent power limit: S_bess ≤ S_rated (55.56 kVA)",
        "- BESS state of charge: 10% ≤ SOC ≤ 90%",
        "- BESS SOC energy balance computed from actual power, capacity (200 kWh), and efficiency (95%)",
        "- Three-phase voltage unbalance: normal < 0.012 pu, warning < 0.020 pu, fail ≥ 0.030 pu",
        "- PV and BESS ramp rates computed and logged",
        "",
        "## IEEE 2030.5-Style Cyber Event Layer",
        "",
        "The cyber event log models the IEEE 2030.5 DER management protocol lifecycle",
        "at the semantic level. IEEE 2030.5 defines a RESTful HTTP/TLS protocol for DERMS",
        "command and control of distributed energy resources. For each command, the following",
        "lifecycle stages are modeled: DER_CONTROL_CREATED → DER_CONTROL_SENT →",
        "DER_CONTROL_RECEIVED → DER_CONTROL_ACCEPTED → DER_CONTROL_APPLIED →",
        "DER_CONTROL_RESPONSE. Security events (AUTH_FAILURE, REPLAY_DETECTED,",
        "INTEGRITY_MISMATCH, BLOCKED_COMMAND) are modeled for attack scenarios.",
        "",
        "**Claim boundary:** All cyber events carry `protocol_claim_level =",
        "semantic_ieee2030_5_style`. No actual serialized IEEE 2030.5 XML/EXI messages",
        "or network packet captures are present. Research-use XML artifacts are optionally",
        "generated to illustrate message structure.",
        "",
        "## Scenario Generation",
        "",
        "A total of 170 scenarios spanning 11 anomaly classes are distributed across the",
        "seven-day window with non-overlapping time windows. Scenario classes include:",
        "normal (routine DER control), cyber_only (unauthorized commands, replay attacks),",
        "physical_only (irradiance drops, load steps, voltage sags), and cyber_physical",
        "(delayed commands, wrong setpoints, BESS wrong direction, SOC violations,",
        "high-rate command bursts). Physical effects are applied in event windows",
        "(pre-event buffer: 120 s, post-event buffer: 300 s).",
        "",
        "**Generation method note:** Anomalous physical effects are generated using a",
        "physics-constrained surrogate model or OpenDSS event-window simulation, depending",
        "on stage configuration. The clean baseline uses the OpenDSS IEEE 123-bus feeder",
        "as its structural foundation. Every row is tagged with `generation_method` to",
        "clearly distinguish `physics_constrained_surrogate` from",
        "`opendss_event_window_resolved`. OpenDSS event-window resolution is tracked in",
        "Report 12.",
        "",
        "## Cyber-Physical Context Mapping",
        "",
        "Each scenario is associated with a context window that aligns cyber command",
        "lifecycle timing to physical effect timing. For delayed command scenarios,",
        "the physical effect is explicitly modeled as starting after command_apply_time,",
        "not at command_sent_time. Residuals (attacked − clean) are computed for key",
        "physical signals. Per-scenario evidence packets (JSONL) aggregate cyber evidence,",
        "physical residuals, constraint check results, and timing evidence for use in",
        "downstream anomaly detection and explainability tasks.",
        "",
        "## Validation and Quality Gates",
        "",
        "Five validation gates are applied: (1) physical constraints, (2) clean/attacked",
        "alignment, (3) IEEE 2030.5-style semantic layer, (4) context causality, and",
        "(5) final gate. Each gate writes a JSON summary and Markdown report. The pipeline",
        "exits with a nonzero code if any critical gate fails.",
        "",
        "## Dataset Limitations",
        "",
        "See Report 09.",
    ]
    write_report(lines, REPORTS / "08_paper_ready_method_text.md")


def generate_report_09_limitations():
    env = load_json(ENV_CHECK_JSON)
    val = load_json(FINAL_VALIDATION_JSON)
    gen_counts = val.get("generation_method_counts", {})
    opendss_avail = env.get("opendss_available", False)
    surrogate_n = int(gen_counts.get("physics_constrained_surrogate", 0))
    ew_n = int(gen_counts.get("opendss_event_window_resolved", 0))

    ods_results_path = METADATA / "opendss_event_window_results.json"
    ods_stats = load_json(ods_results_path)
    ods_successes = int(ods_stats.get("successes", 0))
    ods_failures = int(ods_stats.get("failures", 0))
    ods_stage = int(ods_stats.get("stage", 0))
    ods_scenarios = ods_stats.get("scenarios", [])

    xml_dir = ROOT / "data_updated" / "raw" / "ieee2030_5_xml_research_artifacts"
    xml_count = len(list(xml_dir.glob("*.xml"))) if xml_dir.exists() else 0

    lines = [
        "# Report 09 — Remaining Limitations",
        "",
        "This report honestly states all known limitations of this dataset version.",
        "",
        "## OpenDSS Execution Status",
        "",
        f"- OpenDSS available in environment: **{'YES' if opendss_avail else 'NO'}** (opendssdirect.py)",
        f"- OpenDSS used for clean baseline structure: **YES** (IEEE 123-bus DSS files parsed)",
        f"- OpenDSS event-window simulation executed: **{'YES' if ods_successes > 0 else 'NO'}**",
        "",
        "**OpenDSS != Surrogate:** The clean baseline derives its feeder topology,",
        "bus/line parameters, and rated equipment values from OpenDSS DSS files",
        "(DER_site_001.dss). Row-by-row timeseries values for the surrogate are computed",
        "by the physics-constrained surrogate, not by running OpenDSS power flow each step.",
        "",
    ]

    lines += [
        "## Scenario Resolution Method Counts",
        "",
        f"- OpenDSS event-window resolved: **{ods_successes} scenarios** ({ew_n:,} rows)",
        f"- OpenDSS event-window failures: **{ods_failures} scenarios**",
        f"- Physics-constrained surrogate: **{surrogate_n:,} rows**",
        f"- Stage executed: **{ods_stage if ods_stage else 'not yet run'}**",
        "",
    ]

    if ods_failures > 0:
        lines += ["### OpenDSS Failure Details", ""]
        for s in ods_scenarios:
            if s.get("status") == "FAIL":
                lines.append(f"- {s.get('scenario_id')}: {s.get('error', 'unknown error')[:120]}")
        lines.append("")

    lines += [
        "## XML Artifacts",
        "",
        f"- Research-use XML artifacts generated: **{xml_count} files**",
        "- These are semantic/syntactic illustrations only.",
        "- NOT official IEEE 2030.5 compliance.",
        "- NOT EXI-encoded.",
        "- NOT packet captures.",
        "",
        "## Packet Captures",
        "",
        "- Real network packet captures: **NONE**",
        "- No protocol_claim_level = packet_capture is used in this dataset.",
        "",
        "## IEEE 2030.5 Compliance",
        "",
        "- Official IEEE 2030.5 compliance: **NOT CLAIMED**",
        "- No certification or official conformance testing has been performed.",
        "- The cyber log is a semantic model of the IEEE 2030.5 lifecycle.",
        "",
        "## Real Field Telemetry",
        "",
        "- Real utility field telemetry: **NOT USED**",
        "- All data is simulation-based or model-based.",
        "",
        "## Physical Model Fidelity",
        "",
        "- The surrogate voltage model uses a simplified Thevenin approximation.",
        "  Full OpenDSS power flow would yield more accurate three-phase voltages.",
        "- The load model at Bus 65 is synthetic (diurnal profile).",
        "  Actual load data from IEEE 123-bus spot loads is available in the DSS files",
        "  but requires OpenDSS to extract at 1-second resolution.",
        "- Frequency is modeled at nominal 60 Hz with small noise.",
        "  Dynamic frequency deviation is not modeled in this version.",
        "",
        "## Summary",
        "",
        "| Item | Status |",
        "| --- | --- |",
        "| OpenDSS available | " + ("YES" if opendss_avail else "NO") + " |",
        "| OpenDSS event-window ran | " + (f"YES ({ods_successes} scenarios)" if ods_successes > 0 else "NO (surrogate used)") + " |",
        "| All scenarios OpenDSS event-window resolved | " + ("YES" if ew_n > 0 and surrogate_n == 0 else "NO") + " |",
        "| XML artifacts generated | " + ("YES (research only)" if xml_count > 0 else "NO") + " |",
        "| Packet captures | NO |",
        "| Official IEEE 2030.5 compliance | NOT CLAIMED |",
        "| Real field telemetry | NOT USED |",
    ]
    write_report(lines, REPORTS / "09_remaining_limitations.md")


def generate_report_12_opendss_event_window():
    """Report 12 summarizes OpenDSS event-window stage config and results."""
    ods_results_path = METADATA / "opendss_event_window_results.json"
    ods_cfg_path = METADATA / "opendss_event_window_config.json"
    ods_stats = load_json(ods_results_path)
    ods_cfg = load_json(ods_cfg_path)

    stage = int(ods_stats.get("stage", ods_cfg.get("stage", 1)))
    successes = int(ods_stats.get("successes", 0))
    failures = int(ods_stats.get("failures", 0))
    resolved_rows = int(ods_stats.get("resolved_rows", 0))
    scenarios = ods_stats.get("scenarios", [])
    gm_counts = ods_stats.get("generation_method_counts", {})
    timestamp = ods_stats.get("timestamp", "not yet run")

    lines = [
        "# Report 12 — OpenDSS Event-Window Layer",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Stage:** {stage}",
        f"**Scenarios succeeded:** {successes}",
        f"**Scenarios failed:** {failures}",
        f"**Rows resolved to opendss_event_window_resolved:** {resolved_rows:,}",
        "",
        "## Stage Configuration",
        "",
        "| Stage | Max Scenarios | Trigger |",
        "| --- | --- | --- |",
        "| 1 | 10 (1 per anomaly type) | Manual pipeline run |",
        "| 2 | 30 (3 per anomaly type) | Auto if Stage 1 >= 3 successes |",
        "| 3 | All (170) | Manual enable only |",
        "",
        f"**Current stage:** {stage}",
        f"**Stage 2 minimum successes required:** {ods_cfg.get('stage2_min_successes', 3)}",
        f"**Stage 3 manual only:** {ods_cfg.get('stage3_manual_only', True)}",
        "",
        "## Per-Scenario Results",
        "",
        "| Scenario | Effect Type | Status | Rows Resolved | Error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in scenarios:
        err_col = s.get("error", "")[:80] if s.get("status") == "FAIL" else ""
        lines.append(f"| {s.get('scenario_id', '')} | {s.get('effect_type', '')} | "
                     f"{s.get('status', '')} | {s.get('rows_resolved', 0)} | {err_col} |")

    lines += [
        "",
        "## Generation Method Counts After Stage",
        "",
        "| generation_method | Row Count |",
        "| --- | --- |",
    ]
    for gm, cnt in gm_counts.items():
        lines.append(f"| {gm} | {int(cnt):,} |")

    lines += [
        "",
        "## Honesty Guarantees",
        "",
        "- All OpenDSS results are from real power flow solves (not faked).",
        "- Attack deltas applied on top of OpenDSS values use physics-consistent models.",
        "- On any OpenDSS error, surrogate rows are kept unchanged and error is reported above.",
        "- `generation_method` column correctly reflects actual data origin for every row.",
        "- No result is ever fabricated or interpolated from non-OpenDSS sources.",
    ]

    if not scenarios:
        lines += [
            "",
            "> Stage 1 has not been run yet. Run the pipeline with 04e enabled to",
            "> attempt OpenDSS event-window resolution.",
        ]

    write_report(lines, REPORTS / "12_opendss_event_window_report.md")


def generate_all_reports():
    REPORTS.mkdir(parents=True, exist_ok=True)
    generate_report_06_claim_support()
    generate_report_07_professor_explanation()
    generate_report_08_paper_method()
    generate_report_09_limitations()
    generate_report_12_opendss_event_window()
    print("  All final reports generated.")


if __name__ == "__main__":
    generate_all_reports()
