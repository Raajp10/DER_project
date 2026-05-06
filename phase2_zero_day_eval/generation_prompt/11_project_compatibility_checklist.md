# Project Compatibility Checklist — Must Pass Before Scenario Generation

Use this checklist before sharing the prompt with any LLM and again before compiling the generated scenarios.

## Canonical project root

The implementation must run inside the current final project release folder. Do not hardcode one timestamp permanently.

Preferred behavior:

```text
Use current working directory as project_root unless --project-root is provided.
```

If a script uses an old hardcoded path, it must fail and ask for `--project-root`.

## Required final project files

The project root must contain:

```text
data/raw/physical_timeseries_normal_7d.csv
data/raw/physical_timeseries_anomalous_7d.csv
data/processed/mapped_anomaly_event_windows_7d.csv
data/scenarios/scenario_manifest_7d.csv
configs/week13_14_config.yaml
week13_model_pipeline/
week14_local_llm_explanations/
```

## Required approved dataset checks

Before compiling any LLM scenario:

```text
normal physical rows = 604800
anomalous physical rows = 604800
normal time_s min/max = 0 / 604799
anomalous time_s min/max = 0 / 604799
missing physical values = 0
duplicate timestamps = 0
mapped/manifest label mismatch = 0
outside-window residual rows = 0
normal-window residual rows = 0
cyber-only physical residual rows = 0
```

## Canonical asset IDs

Allowed only:

```text
der_site_001
pv_001
bess_001
pcc_001
```

Reject any old project assets such as:

```text
pv35
pv60
pv83
pv76
pv49
pv104
pv114
bess48
bess76
bess108
bess114
IEEE123 multi-site DER list
```

## Canonical physical features

Allowed model/effect variables only:

```text
pv_p_kw
pv_q_kvar
bess_p_kw
bess_q_kvar
bess_soc_percent
pcc_v_a_pu
pcc_v_b_pu
pcc_v_c_pu
pcc_i_a_amp
pcc_i_b_amp
pcc_i_c_amp
pcc_p_kw
pcc_q_kvar
irradiance_pu
temperature_c
```

Important:

- `irradiance_pu` and `temperature_c` are contextual/environment variables.
- By default, LLM physical effects should not modify `temperature_c`.
- `irradiance_pu` may be modified only for `physical_irradiance_like_disturbance` if explicitly allowed by validator.
- There is no `pcc_freq_hz` in the approved dataset.

## Cyber layer scope

Allowed claim:

```text
Event-level protocol-inspired cyber context layer.
```

Forbidden claims:

```text
packet-level IEEE 2030.5 compliant
DNP3 compliant
IEC 61850 compliant
Modbus compliant
real cyber packet trace
real field telemetry
```

## Frozen model rule

Cross-LLM test sets are evaluation-only.

```text
Train Week 13 once on approved final dataset.
Freeze model artifacts, thresholds, scaler, feature list, window size, and stride.
Evaluate LLM-authored test sets without retraining or recalibrating.
```

## PASS condition

The package is compatible only if:

```text
asset IDs match
feature columns match
labels are consistent
no old project assets appear
project root is config-driven
cyber is event-level only
scenario families are intentionally held-out
validator rejects incompatible output
```
