# Cross-LLM Zero-Day-Like Scenario Rules — Final Exact Match v2

## Scope

1. Generate only synthetic defensive research scenarios.
2. Use only the final dataset assets: `der_site_001`, `pv_001`, `bess_001`, `pcc_001`.
3. Use only final dataset physical variables listed in `02_system_card_final_dataset.json`.
4. Do not create packet-level cyber instructions.
5. Do not create exploit steps, credential steps, payloads, malware, or network intrusion instructions.
6. Do not claim field telemetry or real protocol compliance.
7. Keep all physical effects bounded and plausible.

## Asset rules

Allowed:

```text
der_site_001
pv_001
bess_001
pcc_001
```

Forbidden old assets:

```text
pv35, pv60, pv83, pv76, pv49, pv104, pv114
bess48, bess76, bess108, bess114
```

## Label rules

`normal`:

```text
label_anomaly=0
label_cyber_anomaly=0
label_physical_anomaly=0
physical_effects=[]
```

`cyber_only`:

```text
label_anomaly=1
label_cyber_anomaly=1
label_physical_anomaly=0
physical_effects=[]
```

`physical_only`:

```text
label_anomaly=1
label_cyber_anomaly=0
label_physical_anomaly=1
physical_effects must contain at least one effect
```

`cyber_physical`:

```text
label_anomaly=1
label_cyber_anomaly=1
label_physical_anomaly=1
physical_effects must contain at least one effect
```

## Physical residual rules

1. Physical residuals may exist only inside `[start_time_s, start_time_s + duration_s - 1]`.
2. No persistent effects are allowed unless the compiler explicitly supports persistent-effect metadata.
3. Cyber-only scenarios must not change physical values.
4. Normal scenarios must not change physical values.
5. BESS SOC must remain within 5% to 95%.
6. PV active power must remain nonnegative.
7. PCC voltage must remain within 0.90 to 1.10 pu.
8. Do not modify `temperature_c`.
9. Modify `irradiance_pu` only for `physical_irradiance_like_disturbance`.

## Timeline rules

1. Generate exactly 16 scenarios.
2. Cover all 7 days at least once.
3. Do not put more than 4 scenarios on any one day.
4. Duration must be 60 to 1800 seconds.
5. `start_time_s + duration_s - 1` must be <= 604799.
6. Prefer varied time-of-day: morning, midday, evening, and night.

## Scenario coverage rules

Each bundle must include:

1. at least 2 BESS/SOC scenarios
2. at least 2 PV scenarios
3. at least 2 PCC voltage/power/current scenarios
4. at least 2 cyber-only context scenarios
5. at least 1 coordinated PV+BESS scenario
6. exactly 2 normal control variation scenarios

## Cyber rules

1. Cyber context is event-level only.
2. `packet_level_protocol_compliance_claimed` must be false.
3. Do not include IP addresses, ports, payloads, exploit details, certificates, malware, protocol frames, or packet captures.
4. Use only these protocol profiles:
   - `ieee2030_5_inspired`
   - `ieee1547_interoperability_inspired`
   - `nistir7628_cybersecurity_inspired`
   - `synthetic_der_event_log`

## Big-LLM mistake prevention

Reject any generated scenario if it contains:

- old asset IDs
- new feature columns
- packet-level cyber fields
- real exploit steps
- field-telemetry claims
- IEEE/DNP3/IEC/Modbus compliance claims
- `pcc_freq_hz`
- `temperature_c` as a modified physical effect
