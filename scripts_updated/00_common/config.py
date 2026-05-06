"""Project-wide configuration constants for the DER cyber-physical dataset pipeline."""
from pathlib import Path

# ── Simulation time ──────────────────────────────────────────────────────────
START_TIME_UTC = "2026-01-01T00:00:00Z"
DURATION_DAYS = 7
TIMESTEP_S = 1
TOTAL_ROWS = 604_800        # 7 * 24 * 3600

# ── Site identifiers ─────────────────────────────────────────────────────────
DER_SITE_ID = "der_site_001"
PV_ASSET_ID = "pv_001"
BESS_ASSET_ID = "bess_001"
PCC_ASSET_ID = "pcc_001"
PCC_ID = "pcc_001"
BUS_ID = "65"

# ── Grid parameters ───────────────────────────────────────────────────────────
NOMINAL_FREQUENCY_HZ = 60.0
NOMINAL_VOLTAGE_KV = 4.16
NOMINAL_VOLTAGE_PU = 1.0

# ── DER ratings — sourced from DER_site_001.dss and project_config.yaml ──────
# Priority: DSS file > config YAML > these confirmed manual values
PV_P_RATED_KW = 100.0
PV_S_RATED_KVA = 111.11       # kVA=111.11 in DSS
BESS_P_RATED_KW = 50.0
BESS_S_RATED_KVA = 55.56      # headroom above 50 kW for reactive support
BESS_CAPACITY_KWH = 200.0
BESS_SOC_MIN_PERCENT = 10.0   # %reserve=10.0 in DSS
BESS_SOC_MAX_PERCENT = 90.0
BESS_INITIAL_SOC_PERCENT = 50.0
BESS_EFF_CHARGE_PERCENT = 95.0
BESS_EFF_DISCHARGE_PERCENT = 95.0
BESS_IDLING_KW_PERCENT = 0.1   # %idlingkW in DSS

# ── Load model at PCC bus ─────────────────────────────────────────────────────
# Local load at Bus 65 — approximate based on feeder context
LOAD_BASE_KW = 85.0           # average local load
LOAD_PEAK_KW = 150.0
LOAD_MIN_KW = 25.0

# ── Physical tolerances ───────────────────────────────────────────────────────
PV_APPARENT_POWER_TOLERANCE = 0.5    # kVA
BESS_APPARENT_POWER_TOLERANCE = 0.5  # kVA
VOLTAGE_UNBALANCE_WARN_PU = 0.012
VOLTAGE_UNBALANCE_HARD_PU = 0.020
VOLTAGE_UNBALANCE_FAIL_PU = 0.030
VOLTAGE_MIN_PLAUSIBLE_PU = 0.85
VOLTAGE_MAX_PLAUSIBLE_PU = 1.15

# ── Ramp rates ────────────────────────────────────────────────────────────────
PV_RAMP_RATE_MAX_KW_PER_S = 5.0
BESS_RAMP_RATE_MAX_KW_PER_S = 10.0

# ── Scenario generation ───────────────────────────────────────────────────────
RANDOM_SEED = 42
TARGET_SCENARIO_COUNT = 180     # default manageable benchmark
PRE_EVENT_BUFFER_S = 120
POST_EVENT_BUFFER_S = 300

# ── Cyber / IEEE 2030.5 ───────────────────────────────────────────────────────
NORMAL_LATENCY_MS_MEAN = 50
NORMAL_LATENCY_MS_STD = 15
NORMAL_PROCESSING_MS_MEAN = 20
NORMAL_PROCESSING_MS_STD = 8
DELAYED_LATENCY_MS_MEAN = 8000
PROTOCOL_CLAIM_LEVEL = "semantic_ieee2030_5_style"

# ── Rating source (set after discovery) ──────────────────────────────────────
RATING_SOURCE_FILE = "ieee123_base/DER_site_001.dss"
RATING_SOURCE_OBJECT = "PVSystem.pv_001 / Storage.bess_001"


def load_project_config():
    """Load and return the project YAML config dict, or empty dict if missing."""
    yaml_path = Path(r"D:\updated_dataset\ieee123_base\configs\project_config.yaml")
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            pass
    return {}
