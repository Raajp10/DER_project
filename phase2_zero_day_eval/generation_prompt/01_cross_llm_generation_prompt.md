# Cross-LLM Zero-Day-Like Scenario Generation Prompt — Final Dataset Exact Match v2

You are generating synthetic, physically bounded DER anomaly scenarios for defensive anomaly-detection research.

You must use only the provided shared files:

- `02_system_card_final_dataset.json`
- `03_zero_day_scenario_schema.json`
- `04_scenario_rules.md`
- `05_validator_rules.json`
- `06_llm_assignment_plan.json`
- `09_physical_cyber_label_context.md`
- `10_attack_family_truth_table.csv`
- `11_project_compatibility_checklist.md`

## Absolute compatibility rules

You must generate scenarios for this exact final dataset:

```text
der_site_001
pv_001
bess_001
pcc_001
7 days
604,800 rows
1-second resolution
15 physical features
```

Do not use old project assets:

```text
pv35, pv60, pv83, pv76, pv49, pv104, pv114
bess48, bess76, bess108, bess114
IEEE123 multi-site DER inventory
```

Do not use nonexistent features:

```text
pcc_freq_hz
frequency_hz
grid_frequency
bus_id
line_current_123
```

## Output requirements

Return JSON only.
Do not include markdown.
Do not include prose outside JSON.
Follow `03_zero_day_scenario_schema.json` exactly.
Use only existing assets and variables from `02_system_card_final_dataset.json`.
Use only scenario families from `10_attack_family_truth_table.csv`.
Do not invent new physical variables.
Do not invent packet-level cyber fields.
Do not provide real exploit steps, packet payloads, credentials, network intrusion procedures, malware, or offensive instructions.

## Required top-level statement

Your JSON must include exactly this field/value:

```json
"project_compatibility_statement": "I used only the final single-site dataset assets der_site_001, pv_001, bess_001, pcc_001; I did not use old multi-DER asset IDs or packet-level cyber fields."
```

## Required scenario count

Generate exactly 16 scenarios:

- 4 `physical_only`
- 4 `cyber_only`
- 6 `cyber_physical`
- 2 `normal`

## Timeline requirements

Use different start times across the full 7-day timeline.
Do not cluster all scenarios on one day.
Include at least one scenario on Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, and Day 7.

Day ranges:

```text
Day 1: 0–86399
Day 2: 86400–172799
Day 3: 172800–259199
Day 4: 259200–345599
Day 5: 345600–431999
Day 6: 432000–518399
Day 7: 518400–604799
```

Each scenario must satisfy:

```text
start_time_s + duration_s - 1 <= 604799
```

## Physical effect rules

Physical effects are allowed only for `physical_only` and `cyber_physical` scenarios.

`cyber_only` scenarios must have:

```json
"physical_effects": []
"labels": {"label_anomaly": 1, "label_cyber_anomaly": 1, "label_physical_anomaly": 0}
```

`normal` scenarios must have:

```json
"physical_effects": []
"labels": {"label_anomaly": 0, "label_cyber_anomaly": 0, "label_physical_anomaly": 0}
```

Allowed default physical effect variables:

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
```

`irradiance_pu` may be used only for `physical_irradiance_like_disturbance`.
Do not modify `temperature_c`.

## Scenario objective

Produce novel **zero-day-like synthetic scenarios** that are physically plausible and useful for frozen-model generalization testing.

The goal is not severity. The goal is bounded, plausible, sometimes subtle DER physical behavior changes.

## Safe claim boundary

Use safety notes like:

```text
Synthetic/simulation-only scenario for defensive frozen-model generalization testing; not real-world exploitation or packet-level protocol data.
```
