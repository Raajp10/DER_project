"""Feature configuration for Phase 1 model benchmark.

Defines safe raw physical features and leakage exclusions.
No cyber log columns, no command columns, no label/flag columns.
"""

# Safe raw physical sensor features — all from physical timeseries CSV
# These are instrument readings, not derived from attack labels or command metadata
RAW_FEATURES = [
    "pv_actual_p_kw",          # PV active power output (sensor)
    "pv_actual_q_kvar",        # PV reactive power output (sensor)
    "bess_actual_p_kw",        # BESS active power (sensor, + = discharge)
    "bess_actual_q_kvar",      # BESS reactive power (sensor)
    "bess_soc_percent",        # BESS state of charge (sensor)
    "pcc_v_a_pu",              # PCC phase A voltage (sensor)
    "pcc_v_b_pu",              # PCC phase B voltage (sensor)
    "pcc_v_c_pu",              # PCC phase C voltage (sensor)
    "pcc_i_a_amp",             # PCC phase A current (sensor)
    "pcc_i_b_amp",             # PCC phase B current (sensor)
    "pcc_i_c_amp",             # PCC phase C current (sensor)
    "pcc_p_kw",                # PCC net active power (sensor)
    "pcc_q_kvar",              # PCC net reactive power (sensor)
    "irradiance_pu",           # Solar irradiance (sensor/model)
    "temperature_c",           # Ambient temperature (sensor/model)
    "pcc_voltage_mean_pu",     # Mean of three-phase voltage (derived sensor)
    "pcc_voltage_unbalance_pu", # Voltage unbalance (derived sensor)
    "pv_ramp_rate_kw_per_s",   # PV ramp rate (derived from sensor)
    "bess_ramp_rate_kw_per_s", # BESS ramp rate (derived from sensor)
    "voltage_min_pu",          # Min phase voltage (derived sensor)
    "voltage_max_pu",          # Max phase voltage (derived sensor)
    "line_loading_max_percent", # Line loading (network sensor)
]

N_RAW_FEATURES = len(RAW_FEATURES)

# Statistics computed per window per feature
WINDOW_STATS = ["mean", "std", "min", "max", "median", "slope", "first", "last", "delta", "rms"]
N_STATS = len(WINDOW_STATS)
N_FLAT_FEATURES = N_RAW_FEATURES * N_STATS  # 22 × 10 = 220

# Flat feature names: feature_stat (e.g. pv_actual_p_kw_mean)
FLAT_FEATURE_NAMES = [f"{feat}_{stat}" for feat in RAW_FEATURES for stat in WINDOW_STATS]

# Leakage columns — forbidden from model features
# These contain label info, command info, scenario info, or protocol metadata
LEAKAGE_COLUMNS = [
    # Temporal identifiers
    "timestamp_utc", "time_s",
    # Site identifiers (not physical signals)
    "der_site_id", "pcc_id",
    # Commanded values (not sensor readings — command leakage)
    "pv_commanded_p_kw", "pv_commanded_q_kvar",
    "bess_commanded_p_kw", "bess_commanded_q_kvar",
    # Derived command/setpoint columns
    "pv_curtailment_kw",
    # Constants (no signal, no variation)
    "pv_s_rated_kva", "bess_s_rated_kva",
    "bess_capacity_kwh", "bess_soc_min_percent", "bess_soc_max_percent",
    # Categorical operational state (not physical sensor)
    "pv_inverter_mode", "voltage_unbalance_status",
    "regulator_tap_position", "capacitor_status",
    # Attack/scenario labels — the target variable or derived from it
    "physical_scenario_id", "physical_effect_active_flag",
    "physical_effect_type", "physical_constraint_status",
    # Dataset metadata
    "generation_method",
    # Redundant aggregates already captured by individual phase columns
    "pv_p_kw", "pv_q_kvar",   # use pv_actual_p_kw instead
    "bess_p_kw", "bess_q_kvar",  # use bess_actual_p_kw instead
    # Constraint violation flags (derived from labels)
    "pv_constraint_violation_flag", "bess_constraint_violation_flag",
    # Availability (derived from scenario)
    "pv_available_kw",
]

# Window metadata columns (kept for reporting, not in feature tensor)
WINDOW_META_COLS = [
    "window_id", "source_dataset", "window_start_utc", "window_end_utc",
    "window_start_s", "window_end_s", "split",
    "y_anomaly", "y_cyber_anomaly", "y_physical_anomaly",
    "scenario_id", "scenario_name", "scenario_class",
    "anomaly_type", "generation_method_summary", "protocol_claim_level_summary",
]
