# Phase 3 Few-Shot Grounded Examples

These examples demonstrate correct JSON output format for each explanation type.
All field names match the project field glossary. No packet-level claims are made.
No old asset names (PV35, PV60, BESS48, etc.) are used.

---

## Example 1 — physical_only

**Scenario context:** PV active power (`pv_p_kw`) dropped below expected level during high irradiance.
No cyber flags active. All cyber context shows normal monitoring.

**Evidence summary:**
- `irradiance_pu` = 0.82 (high — sufficient for full output)
- `pv_p_kw` mean in window = 28.4 kW (expected ~65 kW based on irradiance)
- `cyber_anomaly_active` = 0 throughout window
- `physical_effect_active` = 1 throughout window
- `cyber_state` = `physical_only_no_cyber_anomaly`
- No command flags set

**Output:**

```json
{
  "explanation_type": "physical_only",
  "confidence": "high",
  "primary_asset": "pv_001",
  "primary_physical_signals": ["pv_p_kw", "irradiance_pu"],
  "primary_cyber_evidence": [],
  "expected_vs_observed_summary": "Expected pv_p_kw ~65 kW based on irradiance_pu=0.82; observed mean 28.4 kW in window — approximately 56% below expected. Cyber context remained in normal monitoring state throughout.",
  "timing_summary": "Physical effect active for full window duration. No cyber event preceded or coincided with the physical change.",
  "operator_summary": "PV output was significantly below irradiance-expected level with no associated cyber anomaly. This is consistent with inverter derating, hardware protection limit, or physical curtailment. No evidence of cyber cause.",
  "recommended_operator_checks": [
    "Inspect pv_001 inverter fault log for protection triggers",
    "Check pv_p_kw setpoint vs irradiance_pu mismatch",
    "Verify inverter thermal derating status"
  ],
  "evidence_used": ["pv_p_kw", "irradiance_pu", "cyber_anomaly_active", "physical_effect_active", "cyber_state"],
  "evidence_missing": [],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "The PV inverter (pv_001) produced significantly less power than expected given the available irradiance. No cyber or control anomaly was detected. This suggests a physical inverter protection limit, derating, or hardware fault. Operators should check pv_001 inverter logs."
}
```

---

## Example 2 — cyber_only

**Scenario context:** Command suppression on BESS. A command was created and sent but blocked before reaching the asset. Physical BESS output did not change.

**Evidence summary:**
- `bess_p_kw` = stable, within normal range throughout window
- `blocked_flag` = 1 for 45 of 60 seconds in window
- `command_created_flag` = 1, `command_sent_flag` = 1
- `command_apply_flag` = 0 (command never applied)
- `physical_effect_active` = 0
- `cyber_anomaly_active` = 1
- `cyber_state` = `command_blocked`

**Output:**

```json
{
  "explanation_type": "cyber_only",
  "confidence": "high",
  "primary_asset": "bess_001",
  "primary_physical_signals": [],
  "primary_cyber_evidence": ["blocked_flag", "command_created_flag", "command_sent_flag", "command_apply_flag", "cyber_state"],
  "expected_vs_observed_summary": "A command was created and sent toward bess_001 but the blocked_flag was set for 45 of 60 seconds in the window. The command_apply_flag remained 0, meaning the command never reached execution. bess_p_kw remained stable — no physical deviation observed.",
  "timing_summary": "Cyber anomaly (blocked_flag) was active throughout most of the window. No physical effect followed, consistent with the command being suppressed before asset execution.",
  "operator_summary": "A control command targeting bess_001 was suppressed or blocked before application. The physical BESS output was unaffected. This is consistent with a command_suppression event at the control/communications layer.",
  "recommended_operator_checks": [
    "Review bess_001 control command log for suppressed or rejected commands",
    "Check BESS controller access control or firewall rules",
    "Confirm whether SCADA/EMS shows the command as accepted or rejected"
  ],
  "evidence_used": ["blocked_flag", "command_created_flag", "command_sent_flag", "command_apply_flag", "physical_effect_active", "cyber_state", "bess_p_kw"],
  "evidence_missing": [],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "A control command for bess_001 was created and sent but was blocked before it could be applied. No physical BESS power change occurred. This is an event-level cyber anomaly (command suppression) with no physical consequence in this window. Operators should review the BESS command log for blocked or rejected entries."
}
```

---

## Example 3 — cyber_physical

**Scenario context:** Coordinated PV and BESS response. A command was applied and both pv_p_kw and bess_p_kw changed shortly after. Cyber event preceded physical effect.

**Evidence summary:**
- `command_apply_time_s` = 259250
- `physical_effect_start_time_s` = 259250
- `pv_p_kw` dropped from ~70 kW to ~50 kW starting at physical_effect_start_time_s
- `bess_p_kw` shifted from -5 kW (charging) to +18 kW (discharging) at same time
- `cyber_anomaly_active` = 1 in window
- `physical_effect_active` = 1 in window
- `command_apply_flag` = 1
- `cyber_state` = `command_active`
- Timing: `cyber_before_physical` = true (command_apply_time_s <= physical_effect_start_time_s)

**Output:**

```json
{
  "explanation_type": "cyber_physical",
  "confidence": "high",
  "primary_asset": "pv_001",
  "primary_physical_signals": ["pv_p_kw", "bess_p_kw"],
  "primary_cyber_evidence": ["command_apply_flag", "cyber_state", "command_apply_time_s", "physical_effect_start_time_s"],
  "expected_vs_observed_summary": "A coordinated command was applied at second 259250. pv_p_kw dropped ~20 kW and bess_p_kw shifted from charging to discharging at the same time. Both cyber_anomaly_active and physical_effect_active were 1 throughout the detection window.",
  "timing_summary": "command_apply_time_s (259250) coincides with physical_effect_start_time_s (259250), indicating cyber-before-or-coincident-with-physical. Timing alignment supports a cyber-driven physical change.",
  "operator_summary": "A cyber/control command was applied to the DER site and both PV output and BESS dispatch changed concurrently. This is consistent with an unauthorized or anomalous coordinated dispatch command affecting pv_001 and bess_001 simultaneously.",
  "recommended_operator_checks": [
    "Verify whether a coordinated dispatch command was authorized at second 259250",
    "Check pv_001 and bess_001 setpoint logs at that timestamp",
    "Review EMS/SCADA command history for concurrent PV curtailment and BESS dispatch events"
  ],
  "evidence_used": ["pv_p_kw", "bess_p_kw", "command_apply_flag", "cyber_state", "command_apply_time_s", "physical_effect_start_time_s", "cyber_anomaly_active", "physical_effect_active"],
  "evidence_missing": [],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "A control event was applied at second 259250 and both PV power output and BESS dispatch changed at the same time. The timing alignment between the cyber event and physical change strongly suggests the physical deviation was cyber-driven. Operators should verify whether this dispatch was authorized."
}
```

---

## Example 4 — normal

**Scenario context:** Normal operating window. BESS and PV behaving as expected. No anomaly flags active.

**Evidence summary:**
- `cyber_anomaly_active` = 0 throughout
- `physical_effect_active` = 0 throughout
- `bess_soc_percent` = 52–58% (normal drift)
- `pv_p_kw` consistent with `irradiance_pu`
- `pcc_v_a_pu` = 0.998–1.002 (within nominal range)
- `cyber_state` = `normal_monitoring`
- No flags set

**Output:**

```json
{
  "explanation_type": "normal",
  "confidence": "high",
  "primary_asset": "der_site_001",
  "primary_physical_signals": ["bess_soc_percent", "pv_p_kw", "pcc_v_a_pu"],
  "primary_cyber_evidence": ["cyber_state"],
  "expected_vs_observed_summary": "All physical signals within expected ranges: pv_p_kw consistent with irradiance_pu, bess_soc_percent drifting normally (52–58%), pcc_v_a_pu near nominal 1.0 pu. cyber_anomaly_active = 0 and physical_effect_active = 0 throughout.",
  "timing_summary": "No anomaly active during window. Normal monitoring lifecycle observed.",
  "operator_summary": "Normal DER site operation. No physical or cyber anomaly detected. No operator action required.",
  "recommended_operator_checks": [],
  "evidence_used": ["bess_soc_percent", "pv_p_kw", "irradiance_pu", "pcc_v_a_pu", "cyber_anomaly_active", "physical_effect_active", "cyber_state"],
  "evidence_missing": [],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "All monitored signals are within expected normal operating ranges. No cyber event or physical anomaly is active. This detection window shows normal DER site behaviour."
}
```

---

## Example 5 — insufficient_evidence

**Scenario context:** The detection window shows an anomaly score above threshold, but the physical signals do not clearly deviate, the cyber context is partially contradictory, and key timing fields are absent.

**Evidence summary:**
- `predicted_label` = 1 (anomaly detected)
- `cyber_anomaly_active` = 1 for 10 of 60 seconds only
- `physical_effect_active` = 0
- `bess_p_kw` = noisy but within bounds — no clear step or clip
- `command_apply_time_s` = not available
- `physical_effect_start_time_s` = not available
- `cyber_state` = `anomaly_active` for some seconds, `normal_monitoring` for others
- No command flags consistently set

**Output:**

```json
{
  "explanation_type": "insufficient_evidence",
  "confidence": "low",
  "primary_asset": "bess_001",
  "primary_physical_signals": ["bess_p_kw"],
  "primary_cyber_evidence": ["cyber_anomaly_active", "cyber_state"],
  "expected_vs_observed_summary": "The model flagged this window as anomalous but the physical evidence is ambiguous: bess_p_kw shows noise but no clear directional change. cyber_anomaly_active was 1 for only 10 of 60 seconds. Key timing fields (command_apply_time_s, physical_effect_start_time_s) are not available.",
  "timing_summary": "No consistent timing pattern. Cyber flag was intermittent and did not precede a clear physical event. Cannot establish cyber-before-physical or physical-only timing.",
  "operator_summary": "Evidence is insufficient to classify this detection. The anomaly score exceeded threshold but neither a clear physical change nor a sustained cyber event is observable. Manual review recommended.",
  "recommended_operator_checks": [
    "Manually review bess_p_kw time series around this window",
    "Check whether bess_001 controller logged any command or fault at this time",
    "Determine whether the cyber_state flip to anomaly_active was sustained or transient"
  ],
  "evidence_used": ["bess_p_kw", "cyber_anomaly_active", "cyber_state", "physical_effect_active"],
  "evidence_missing": ["command_apply_time_s", "physical_effect_start_time_s", "clear physical deviation", "sustained cyber flag"],
  "unsupported_claims_made": false,
  "packet_level_claim_made": false,
  "field_telemetry_claim_made": false,
  "external_attacker_claim_made": false,
  "old_asset_name_used": false,
  "human_explanation": "Evidence is contradictory or insufficient to classify this detection. The model flagged the window but neither physical nor cyber evidence is clearly conclusive. Manual operator review of the bess_001 controller log is recommended before drawing further conclusions."
}
```
