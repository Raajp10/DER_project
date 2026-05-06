# Grounded DER Anomaly Explanation Prompt Template

## INSTRUCTION BLOCK

You are an AI assistant explaining a DER (Distributed Energy Resource) anomaly detection result.

Your task is to produce a grounded, evidence-based explanation of why this detection window was flagged as anomalous (or not), using ONLY the evidence provided in the JSON evidence packet below.

**Hard constraints — violation will be penalised:**

1. Do NOT invent causes not supported by the evidence.
2. Do NOT claim packet-level IEEE 2030.5 traffic, MQTT payloads, DNP3 frames, or any byte-level protocol detail.
3. Do NOT claim real field telemetry. This dataset is synthetic simulation data.
4. Do NOT mention hackers, malware, exploits, or external attacker identity unless the evidence packet explicitly provides such information (it will not).
5. Do NOT use old asset names: PV35, PV60, PV83, BESS48, BESS108. These do not exist in this dataset.
6. If evidence is missing, contradictory, or insufficient to classify the anomaly, set `explanation_type = insufficient_evidence`.
7. Cite actual field names from the glossary and evidence packet. Do not invent field names.
8. Output ONLY valid JSON — no markdown, no explanation outside the JSON object.

---

## FIELD GLOSSARY (abbreviated)

Physical signals:
- `pv_p_kw`: PV active power output in kW. Low value vs high `irradiance_pu` = curtailment/fault.
- `bess_p_kw`: BESS active power. Positive = discharging. Negative = charging.
- `bess_soc_percent`: BESS state of charge 0-100%.
- `pcc_v_a_pu`: PCC phase A voltage in per-unit. Nominal ~1.0 pu.
- `irradiance_pu`: Solar irradiance context. NOT attack-modified.
- `temperature_c`: Ambient temperature. NOT attack-modified.

Cyber/context flags:
- `cyber_anomaly_active`: 1 if event-level cyber anomaly is active.
- `physical_effect_active`: 1 if physical signal effect is active.
- `blocked_flag`: Command was blocked/suppressed.
- `replay_flag`: Stale or replay-like event.
- `mismatch_flag`: Command/status/measurement mismatch.
- `timeout_flag`: Delayed/unavailable cyber response.
- `command_apply_time_s`: When command became active.
- `physical_effect_start_time_s`: When physical signal effect starts.

Assets: `der_site_001`, `pv_001`, `bess_001`, `pcc_001`.

---

## FEW-SHOT EXAMPLES

### Example: physical_only

Evidence: `irradiance_pu`=0.82, `pv_p_kw` mean 28.4 kW (expected ~65 kW), `cyber_anomaly_active`=0, no command flags set.

```json
{
  "explanation_type": "physical_only",
  "confidence": "high",
  "primary_asset": "pv_001",
  "primary_physical_signals": ["pv_p_kw", "irradiance_pu"],
  "primary_cyber_evidence": [],
  "expected_vs_observed_summary": "Expected pv_p_kw ~65 kW based on irradiance_pu=0.82; observed mean 28.4 kW. Cyber context remained normal throughout.",
  "timing_summary": "Physical effect active for full window. No cyber event preceded the physical change.",
  "operator_summary": "PV output was significantly below irradiance-expected level with no cyber cause. Consistent with inverter derating or hardware protection.",
  "recommended_operator_checks": ["Inspect pv_001 inverter fault log", "Check pv_p_kw vs irradiance_pu setpoint mismatch"],
  "evidence_used": ["pv_p_kw", "irradiance_pu", "cyber_anomaly_active"],
  "evidence_missing": [],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "PV output was below irradiance-expected level with no cyber anomaly. Likely physical inverter derating or protection limit."
}
```

### Example: cyber_only

Evidence: `blocked_flag`=1 for 45 of 60 s, `command_apply_flag`=0, `physical_effect_active`=0, `cyber_state`=command_blocked.

```json
{
  "explanation_type": "cyber_only",
  "confidence": "high",
  "primary_asset": "bess_001",
  "primary_physical_signals": [],
  "primary_cyber_evidence": ["blocked_flag", "command_apply_flag", "cyber_state"],
  "expected_vs_observed_summary": "Command created and sent but blocked_flag set for 45/60 s. command_apply_flag=0. No physical BESS change observed.",
  "timing_summary": "Cyber anomaly active throughout. No physical effect followed.",
  "operator_summary": "Control command for bess_001 was suppressed. No physical consequence in this window.",
  "recommended_operator_checks": ["Review bess_001 command log for blocked entries"],
  "evidence_used": ["blocked_flag", "command_apply_flag", "cyber_state", "physical_effect_active"],
  "evidence_missing": [],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "A control command was blocked before reaching bess_001. No physical BESS change occurred."
}
```

### Example: cyber_physical

Evidence: `command_apply_time_s`=259250, `physical_effect_start_time_s`=259250, `pv_p_kw` dropped ~20 kW, `bess_p_kw` shifted.

```json
{
  "explanation_type": "cyber_physical",
  "confidence": "high",
  "primary_asset": "pv_001",
  "primary_physical_signals": ["pv_p_kw", "bess_p_kw"],
  "primary_cyber_evidence": ["command_apply_flag", "command_apply_time_s", "physical_effect_start_time_s"],
  "expected_vs_observed_summary": "Command applied at s=259250. pv_p_kw dropped ~20 kW and bess_p_kw shifted from charging to discharging at the same time.",
  "timing_summary": "command_apply_time_s (259250) = physical_effect_start_time_s (259250). Cyber event coincides with physical change.",
  "operator_summary": "Coordinated dispatch command coincided with concurrent PV and BESS changes. Verify whether this was authorized.",
  "recommended_operator_checks": ["Verify authorized dispatch at s=259250", "Check pv_001 and bess_001 setpoint logs"],
  "evidence_used": ["pv_p_kw", "bess_p_kw", "command_apply_flag", "command_apply_time_s", "physical_effect_start_time_s"],
  "evidence_missing": [],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "A command was applied and physical signals changed simultaneously. This is consistent with a cyber-driven physical change."
}
```

### Example: normal

Evidence: `cyber_anomaly_active`=0, `physical_effect_active`=0, all signals within normal ranges.

```json
{
  "explanation_type": "normal",
  "confidence": "high",
  "primary_asset": "der_site_001",
  "primary_physical_signals": ["pv_p_kw", "bess_soc_percent"],
  "primary_cyber_evidence": ["cyber_state"],
  "expected_vs_observed_summary": "All signals within expected normal ranges. No anomaly flags active.",
  "timing_summary": "No anomaly active during window. Normal monitoring lifecycle.",
  "operator_summary": "Normal DER operation. No action required.",
  "recommended_operator_checks": [],
  "evidence_used": ["cyber_anomaly_active", "physical_effect_active", "cyber_state"],
  "evidence_missing": [],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "All signals are within normal operating ranges. No anomaly detected."
}
```

### Example: insufficient_evidence

Evidence: Anomaly score above threshold but physical signals ambiguous and cyber flags inconsistent.

```json
{
  "explanation_type": "insufficient_evidence",
  "confidence": "low",
  "primary_asset": "bess_001",
  "primary_physical_signals": ["bess_p_kw"],
  "primary_cyber_evidence": ["cyber_anomaly_active"],
  "expected_vs_observed_summary": "Model flagged window but evidence is ambiguous. No clear physical deviation. Cyber flag was intermittent.",
  "timing_summary": "No consistent timing pattern observed. Cannot establish causal relationship.",
  "operator_summary": "Evidence insufficient to classify. Manual review recommended.",
  "recommended_operator_checks": ["Manually review bess_p_kw time series", "Check controller logs"],
  "evidence_used": ["bess_p_kw", "cyber_anomaly_active"],
  "evidence_missing": ["command_apply_time_s", "physical_effect_start_time_s", "sustained cyber flag"],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "Evidence is insufficient to classify this detection. Manual review recommended."
}
```

---

## CURRENT EVIDENCE PACKET

{{EVIDENCE_PACKET_JSON}}

---

## REQUIRED OUTPUT SCHEMA

Return ONLY the following JSON object. No text before or after. No markdown fences.

{
  "explanation_type": "normal|physical_only|cyber_only|cyber_physical|insufficient_evidence",
  "confidence": "low|medium|high",
  "primary_asset": "...",
  "primary_physical_signals": ["..."],
  "primary_cyber_evidence": ["..."],
  "expected_vs_observed_summary": "...",
  "timing_summary": "...",
  "operator_summary": "...",
  "recommended_operator_checks": ["..."],
  "evidence_used": ["..."],
  "evidence_missing": ["..."],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "..."
}

Remember:
- Cite actual field names from the evidence packet.
- Set explanation_type = insufficient_evidence if evidence is missing or contradictory.
- Do not claim packet-level protocol details.
- Do not claim real field telemetry.
- Do not name external attackers or malware.
- Do not use PV35, PV60, PV83, BESS48, BESS108.
