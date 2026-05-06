"""Absolute path constants for the DER cyber-physical dataset pipeline."""
from pathlib import Path

ROOT = Path(r"D:\updated_dataset")

# Source
IEEE123_BASE = ROOT / "ieee123_base"
MASTER_DSS = IEEE123_BASE / "IEEE123Master.dss"
DER_SITE_DSS = IEEE123_BASE / "DER_site_001.dss"
QSTS_DSS = IEEE123_BASE / "DER_QSTS_7day.dss"
PROJECT_CONFIG_YAML = IEEE123_BASE / "configs" / "project_config.yaml"

# Backup
BACKUP = ROOT / "original_input_backup"

# Scripts
SCRIPTS = ROOT / "scripts_updated"

# Data
DATA = ROOT / "data_updated"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SCENARIOS = DATA / "scenarios"
METADATA = DATA / "metadata"
VALIDATION = DATA / "validation"
VIZ = DATA / "visualization"

# Outputs
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
FINAL_PACKAGE = ROOT / "final_package"

# Raw physical
CLEAN_PHYSICAL_CSV = RAW / "physical_timeseries_clean_improved_7d.csv"
ATTACKED_PHYSICAL_CSV = RAW / "physical_timeseries_attacked_improved_7d.csv"
RESIDUALS_CSV = RAW / "physical_residuals_improved_7d.csv"

# Cyber
CYBER_NORMAL_CSV = RAW / "cyber_event_log_normal_ieee2030_5_semantic_7d.csv"
CYBER_ANOMALOUS_CSV = RAW / "cyber_event_log_anomalous_ieee2030_5_semantic_7d.csv"
IEEE2030_5_SCHEMA_JSON = METADATA / "ieee2030_5_semantic_schema.json"
IEEE2030_5_XML_DIR = RAW / "ieee2030_5_xml_research_artifacts"

# Processed
LIFECYCLE_MAP_CSV = PROCESSED / "cyber_physical_lifecycle_map_7d.csv"
CONTEXT_WINDOWS_CSV = PROCESSED / "event_specific_context_windows_7d.csv"
EVIDENCE_PACKETS_JSONL = PROCESSED / "cyber_physical_evidence_packets_7d.jsonl"

# Scenarios
SCENARIO_MANIFEST_CSV = SCENARIOS / "scenario_manifest_improved_7d.csv"
SCENARIO_MANIFEST_JSON = SCENARIOS / "scenario_manifest_improved_7d.json"

# Metadata
DER_METADATA_JSON = METADATA / "der_physical_metadata.json"
INPUT_DISCOVERY_JSON = METADATA / "input_discovery_report.json"
ENV_CHECK_JSON = VALIDATION / "environment_check.json"

# Validation
PHYSICAL_CONSTRAINTS_REPORT = VALIDATION / "physical_constraints_report.md"
PHYSICAL_CONSTRAINTS_JSON = VALIDATION / "physical_constraints_summary.json"
ALIGNMENT_REPORT = VALIDATION / "clean_attacked_alignment_report.md"
CYBER_VALIDATION_REPORT = VALIDATION / "ieee2030_5_semantic_validation_report.md"
CYBER_VALIDATION_JSON = VALIDATION / "ieee2030_5_semantic_validation_summary.json"
CONTEXT_VALIDATION_REPORT = VALIDATION / "context_causality_validation_report.md"
CONTEXT_VALIDATION_JSON = VALIDATION / "context_causality_validation_summary.json"
FINAL_VALIDATION_REPORT = VALIDATION / "final_updated_dataset_validation_report.md"
FINAL_VALIDATION_JSON = VALIDATION / "final_updated_dataset_validation_summary.json"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [RAW, PROCESSED, SCENARIOS, METADATA, VALIDATION,
              VIZ, VIZ/"physical", VIZ/"cyber", VIZ/"context", VIZ/"summary",
              REPORTS, FIGURES, LOGS, FINAL_PACKAGE, IEEE2030_5_XML_DIR]:
        d.mkdir(parents=True, exist_ok=True)
