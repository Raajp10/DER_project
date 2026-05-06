# Phase 3 Field Glossary

This glossary defines every field that may appear in a Phase 3 evidence packet.
Use exact field names when citing evidence. Do not invent field names.

---

## Physical Signal Fields

| Field | Unit | Meaning |
|---|---|---|
| `pv_p_kw` | kW | PV active power output. Higher values mean more solar generation. A drop below expected irradiance-based output is a curtailment or fault signal. |
| `pv_q_kvar` | kvar | PV reactive power. Normally small. Deviations may indicate inverter control changes. |
| `bess_p_kw` | kW | BESS active power. Positive = discharging (exporting to grid/load). Negative = charging (importing from grid). Sign convention is asset-specific; check metadata if uncertain. |
| `bess_q_kvar` | kvar | BESS reactive power. Normally small. Deviations may indicate inverter control changes. |
| `bess_soc_percent` | % | BESS state of charge. Range 0–100. Unexpected drift (too fast or stale/flat) may indicate measurement error, stale data, or control anomaly. |
| `pcc_v_a_pu` | per-unit | PCC phase A voltage magnitude in per-unit. Nominal ~1.0 pu. Deviations above ~1.05 or below ~0.95 are significant. |
| `pcc_v_b_pu` | per-unit | PCC phase B voltage. Same interpretation as `pcc_v_a_pu`. |
| `pcc_v_c_pu` | per-unit | PCC phase C voltage. Same interpretation as `pcc_v_a_pu`. |
| `pcc_i_a_amp` | A | PCC phase A current in amperes. Unexpected spikes or flatlines indicate load/generation anomalies. |
| `pcc_i_b_amp` | A | PCC phase B current. Same interpretation as `pcc_i_a_amp`. |
| `pcc_i_c_amp` | A | PCC phase C current. Same interpretation as `pcc_i_a_amp`. |
| `pcc_p_kw` | kW | Active power measured at point of common coupling. Net generation minus net load. |
| `pcc_q_kvar` | kvar | Reactive power at PCC. May indicate reactive compensation anomalies. |
| `irradiance_pu` | per-unit | Solar irradiance context signal. Nominal range 0–1. Used to compute expected PV output. If `pv_p_kw` is low when `irradiance_pu` is high, curtailment or fault is likely. |
| `temperature_c` | °C | Ambient temperature. Context only. **Not modified by any attack scenario in this dataset.** Do not attribute anomalies to temperature unless corroborated. |

---

## Cyber / Context Fields

| Field | Type | Meaning |
|---|---|---|
| `cyber_state` | string | Current event-level cyber/control state. Values include: `normal_monitoring`, `command_active`, `command_blocked`, `stale_or_delayed`, `replay_detected`, `mismatch_detected`, `anomaly_active`, `physical_only_no_cyber_anomaly`. |
| `cyber_lifecycle_stage` | string | Stage of command or status lifecycle. Values: `command_created`, `command_sent`, `command_received`, `command_accepted`, `command_applied`, `physical_response_observed`, `status_report`, `monitoring_complete`. |
| `cyber_anomaly_active` | 0/1 | 1 if an event-level cyber or control anomaly is active at this second. 0 otherwise. |
| `physical_effect_active` | 0/1 | 1 if a physical signal effect is active at this second. 0 otherwise. |
| `command_created_flag` | 0/1 | A command or control event was created at this second. |
| `command_sent_flag` | 0/1 | A command or control event was sent toward the asset. |
| `command_recv_flag` | 0/1 | A command or control event was received by the asset controller. |
| `command_accept_flag` | 0/1 | The command was accepted by the asset. |
| `command_apply_flag` | 0/1 | The command became active / was applied to asset operation. |
| `command_response_flag` | 0/1 | A status report or response was observed after the command. |
| `blocked_flag` | 0/1 | A command or event was blocked or suppressed. Associated with `command_suppression` scenario family. |
| `replay_flag` | 0/1 | A stale or replay-like event was detected. Associated with `stale_or_delayed_measurement` or related families. |
| `mismatch_flag` | 0/1 | A command/status/measurement mismatch was detected. May indicate false data injection or command delay. |
| `stale_command_flag` | 0/1 | Stale measurement or delayed/stale command context detected. |
| `timeout_flag` | 0/1 | Delayed or unavailable cyber response observed. Associated with `command_delay` family. |
| `command_apply_time_s` | int | Second at which a command became active/applied. Used for timing alignment. |
| `physical_effect_start_time_s` | int | Second at which the physical signal effect begins. Compare with `command_apply_time_s` for timing analysis. |

---

## Claim Boundaries

These rules apply to ALL explanations. Violations are scored as penalties.

- **Event-level cyber context only.** This dataset contains synthetic event-level metadata only. There are no packet captures, no raw protocol traces, no byte-level payloads.
- **No packet-level protocol traces.** Do not claim IEEE 2030.5 packet contents, MQTT payloads, DNP3 frames, Modbus registers, or any protocol-level detail.
- **No real field telemetry.** This is a synthetic simulation dataset. Do not claim these are real measurements from a live DER site.
- **No real exploit evidence.** Do not claim evidence of actual malware, specific CVE exploitation, or known threat actors unless the evidence packet explicitly supports such a claim (it will not in this dataset).
- **No external attacker attribution.** Do not name or imply specific attackers, nation-states, or threat groups.
- **Cite actual field names.** When describing evidence, use field names exactly as they appear in this glossary. Do not invent field names.
- **Use `insufficient_evidence` when needed.** If the evidence is contradictory, missing, or does not clearly support a classification, set `explanation_type = insufficient_evidence`.

---

## Asset Names (this dataset)

| Asset ID | Component | Description |
|---|---|---|
| `der_site_001` | site | Whole DER site |
| `pv_001` | pv | PV inverter/array |
| `bess_001` | bess | Battery energy storage system |
| `pcc_001` | pcc | Point of common coupling |

**Do NOT use old multi-DER asset names:** PV35, PV60, PV83, BESS48, BESS108.
These names do not exist in this dataset and their use indicates hallucination.
