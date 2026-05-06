# Physical/Cyber Label Context — Final Dataset Exact Match v2

## Detector input

The Week 13 detector uses physical-layer windows only.

Allowed physical feature columns:

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

## Leakage columns

Do not use the following as model features or generated physical effects:

```text
timestamp_utc
time_s
event_id
scenario_id
active_event_id
active_asset_id
scenario_name
scenario_class
scenario_family
anomaly_type
label_anomaly
label_cyber_anomaly
label_physical_anomaly
cyber_anomaly_flag
physical_anomaly_flag
final_event_class
time_since_command_s
time_until_apply_s
expected_p_kw
actual_p_kw
power_error_kw
expected_q_kvar
actual_q_kvar
reactive_power_error_kvar
actual_soc_percent
response_late_flag
physical_response_changed_flag
```

## Label truth

| scenario_class | label_anomaly | label_cyber_anomaly | label_physical_anomaly | physical_effects |
|---|---:|---:|---:|---|
| normal | 0 | 0 | 0 | empty |
| cyber_only | 1 | 1 | 0 | empty |
| physical_only | 1 | 0 | 1 | non-empty |
| cyber_physical | 1 | 1 | 1 | non-empty |

## Cyber layer role

Cyber logs are event-level context for Week 14 explanations and scenario metadata.

They are not primary detection features and are not packet-level protocol data.

## Zero-day-like rule

The new cross-LLM scenario families are intentionally held out. They do not need to match the original training scenario family names, but they must be listed in `10_attack_family_truth_table.csv`.
