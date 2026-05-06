/**
 * Data loader with automatic fallback.
 * Attempts to load live demo JSON; falls back to fallback copies if unavailable.
 */

async function fetchJSON(url) {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`)
  return resp.json()
}

export async function loadDemoData() {
  // Try live data first
  try {
    const [presenter, technical, summary] = await Promise.all([
      fetchJSON('/data/demo_presenter_scenarios.json'),
      fetchJSON('/data/demo_technical_scenarios.json'),
      fetchJSON('/data/demo_summary.json'),
    ])
    return { presenter, technical, summary, source: 'live' }
  } catch (_) {
    // Fall through to fallback
  }

  // Try fallback data
  try {
    const [presenter, technical, summary] = await Promise.all([
      fetchJSON('/data/fallback_demo_presenter_scenarios.json'),
      fetchJSON('/data/fallback_demo_technical_scenarios.json'),
      fetchJSON('/data/fallback_demo_summary.json'),
    ])
    return { presenter, technical, summary, source: 'fallback' }
  } catch (e) {
    // Return minimal hardcoded stub
    return {
      presenter: getHardcodedPresenter(),
      technical: getHardcodedPresenter(),
      summary: getHardcodedSummary(),
      source: 'hardcoded',
    }
  }
}

function getHardcodedSummary() {
  return {
    generated: new Date().toISOString(),
    total_demo_scenarios: 4,
    presenter_scenarios: 4,
    average_evidence_score: 0.847,
    unsupported_claims_count: 0,
    packet_level_claims_count: 0,
    field_telemetry_claims_count: 0,
    detection_mode_display: 'Demo mode — placeholder detection scores',
    selected_llm_model: 'qwen2.5:3b-instruct',
    data_source: 'hardcoded',
  }
}

function getHardcodedPresenter() {
  const makeTs = (baseSignal, attackDelta, n = 300) => {
    const t_rel = Array.from({ length: n }, (_, i) => i)
    const vals = t_rel.map(t => {
      const inAttack = t >= 120 && t <= 180
      const base = baseSignal + Math.sin(t / 20) * 2
      return inAttack ? base + attackDelta * ((t - 120) / 60) : base
    })
    return { time_relative: t_rel, attack_start_relative: 120, attack_end_relative: 180,
             window_start_relative: 120, window_end_relative: 180, n_points: n, pv_p_kw: vals }
  }

  const classes = ['cyber_physical', 'physical_only', 'cyber_only', 'normal']
  const labels  = ['Demo A — Cyber-Physical', 'Demo B — Physical-Only', 'Demo C — Cyber-Only', 'Demo D — Normal Behavior']
  return classes.map((cls, i) => ({
    detection_id: `det_${String(i+1).padStart(4,'0')}`,
    scenario_id: `zdl_example_${cls}_00${i+1}`,
    scenario_class: cls,
    scenario_family: ['command_delay','pv_curtailment_mismatch','command_suppression','normal_control_variation'][i],
    author_model: ['grok','chatgpt','claude','gemini'][i],
    target_asset_id: ['bess_001','pv_001','bess_001','der_site_001'][i],
    target_component: ['bess','pv','bess','site'][i],
    demo_label: labels[i],
    demo_class_key: cls,
    detection: {
      model_name: 'smoke_detector',
      display_name: 'Demo mode — placeholder detection scores',
      anomaly_score: cls === 'normal' ? 0.22 : 0.78,
      threshold: 0.5,
      predicted_label: cls === 'normal' ? 0 : 1,
      smoke_detection_only: true,
    },
    labels: {
      label_anomaly: cls !== 'normal' ? 1 : 0,
      label_cyber_anomaly: ['cyber_physical','cyber_only'].includes(cls) ? 1 : 0,
      label_physical_anomaly: ['cyber_physical','physical_only'].includes(cls) ? 1 : 0,
    },
    timing: {
      window_start_s: 259200, window_end_s: 259259,
      event_start_s: 259200, event_end_s: 259259,
      command_apply_time_s: cls.includes('cyber') ? 259198 : null,
      physical_effect_start_time_s: cls !== 'cyber_only' ? 259200 : null,
      cyber_before_physical: cls === 'cyber_physical' ? true : null,
      timing_alignment_status: cls === 'cyber_physical' ? 'cyber_before_physical_confirmed'
        : cls === 'physical_only' ? 'physical_only_no_cyber_event'
        : cls === 'cyber_only' ? 'cyber_only_no_physical_effect' : 'no_anomaly',
    },
    timeseries: makeTs(cls === 'normal' ? 60 : 50, cls === 'cyber_only' ? 0 : -15),
    cyber_timeline: {
      stages: cls === 'cyber_physical'
        ? [{key:'created',label:'Created',num:1,anomaly:false},{key:'sent',label:'Sent',num:2,anomaly:false},{key:'received',label:'Received',num:3,anomaly:false},{key:'accepted',label:'Accepted',num:4,anomaly:false},{key:'applied',label:'Applied',num:5,anomaly:false},{key:'response',label:'Physical Response',num:6,anomaly:true},{key:'status',label:'Status',num:7,anomaly:false}]
        : cls === 'cyber_only'
        ? [{key:'created',label:'Created',num:1,anomaly:false},{key:'sent',label:'Sent',num:2,anomaly:false},{key:'blocked',label:'Blocked',num:3,anomaly:true},{key:'alert',label:'Security Alert',num:4,anomaly:true},{key:'status',label:'Status',num:5,anomaly:false}]
        : cls === 'physical_only'
        ? [{key:'pre',label:'Pre-event Monitoring',num:1,anomaly:false},{key:'event',label:'Physical Event',num:2,anomaly:true},{key:'post',label:'Post-event Monitoring',num:3,anomaly:false}]
        : [{key:'normal',label:'Normal Monitoring',num:1,anomaly:false},{key:'complete',label:'Normal Status',num:2,anomaly:false}],
      anomaly_description: cls === 'cyber_physical' ? 'Cyber event preceded physical signal change.'
        : cls === 'cyber_only' ? 'Command suppression detected.'
        : cls === 'physical_only' ? 'Physical signal deviated without cyber cause.'
        : 'Normal DER operation.',
      cyber_before_physical: cls === 'cyber_physical' ? true : null,
      timing_alignment_status: 'cyber_before_physical_confirmed',
    },
    explanation: {
      explanation_type: cls,
      confidence: 'high',
      primary_asset: ['bess_001','pv_001','bess_001','der_site_001'][i],
      primary_physical_signals: cls !== 'cyber_only' ? ['pv_p_kw','bess_p_kw'] : [],
      primary_cyber_evidence: cls !== 'physical_only' && cls !== 'normal' ? ['command_apply_flag','cyber_state'] : [],
      timing_summary: cls === 'cyber_physical'
        ? 'Command applied before physical signal changed — timing confirms cyber-driven event.'
        : cls === 'physical_only'
        ? 'Physical effect active throughout window. No cyber event preceded the change.'
        : cls === 'cyber_only'
        ? 'Cyber anomaly active. No physical signal deviation observed.'
        : 'No anomaly active. Normal monitoring lifecycle.',
      operator_summary: cls === 'cyber_physical'
        ? 'A control command coincided with a physical DER change. Verify authorization.'
        : cls === 'physical_only'
        ? 'Physical DER signal changed with no cyber cause. Check inverter fault logs.'
        : cls === 'cyber_only'
        ? 'A control command was blocked. Physical DER was unaffected.'
        : 'Normal DER operation. No action required.',
      recommended_operator_checks: cls === 'normal' ? []
        : cls === 'cyber_physical' ? ['Verify dispatch command authorization at event time','Check SCADA command log for concurrent events']
        : cls === 'physical_only' ? ['Inspect inverter fault log','Check irradiance vs output mismatch']
        : ['Review blocked command log','Check BESS controller access rules'],
      human_explanation: cls === 'cyber_physical'
        ? 'A control command was applied and physical DER output changed concurrently. Timing alignment suggests the physical change was cyber-driven.'
        : cls === 'physical_only'
        ? 'The DER physical signal deviated from expected behavior with no associated cyber event. Consistent with inverter protection or hardware fault.'
        : cls === 'cyber_only'
        ? 'A cyber/control event was detected but no physical DER change occurred. Consistent with command suppression.'
        : 'All signals within normal operating ranges. No anomaly detected.',
      evidence_used: cls === 'cyber_physical'
        ? ['pv_p_kw','bess_p_kw','command_apply_flag','cyber_state','physical_effect_active']
        : cls === 'physical_only'
        ? ['pv_p_kw','irradiance_pu','cyber_anomaly_active','physical_effect_active']
        : cls === 'cyber_only'
        ? ['blocked_flag','command_apply_flag','cyber_state','physical_effect_active']
        : ['cyber_anomaly_active','physical_effect_active','cyber_state'],
    },
    evidence: {
      scenario_class_match: 1, asset_match: 1, physical_signal_match: 1,
      cyber_state_match: 1, timing_match: 1, expected_vs_observed_match: 1,
      unsupported_claim_flag: 0, packet_claim_flag: 0, field_telemetry_claim_flag: 0,
      external_attacker_claim_flag: 0, uses_old_asset_name_flag: 0,
      overall_evidence_score: 0.847, parse_success: 1,
    },
    physical_evidence: {
      top_signals: [
        {signal:'pv_p_kw',mean:48.2,std:8.3,delta:-15.1},
        {signal:'bess_p_kw',mean:12.4,std:6.2,delta:22.3},
      ],
      affected_variables_from_scenario: cls !== 'cyber_only' ? ['pv_p_kw'] : [],
      physical_effect_active_fraction: cls !== 'cyber_only' && cls !== 'normal' ? 0.85 : 0.0,
      expected_behavior_summary: 'Normal DER dispatch consistent with irradiance and SOC.',
      observed_behavior_summary: cls !== 'normal' ? 'Signal deviated from expected behavior.' : 'Normal operation.',
      expected_vs_observed_difference: cls !== 'normal' ? 'Anomalous deviation observed.' : 'No difference.',
    },
    cyber_evidence: {
      cyber_states_seen: cls !== 'normal' && cls !== 'physical_only'
        ? ['command_active','anomaly_active'] : ['normal_monitoring'],
      lifecycle_stages_seen: cls !== 'physical_only' && cls !== 'normal'
        ? ['command_created','command_sent'] : ['monitoring_complete'],
      blocked_flag_seen: cls === 'cyber_only',
      replay_flag_seen: false, mismatch_flag_seen: false,
      stale_command_flag_seen: false,
      timeout_flag_seen: cls === 'cyber_physical',
      cyber_anomaly_active_fraction: cls !== 'normal' && cls !== 'physical_only' ? 0.85 : 0.0,
    },
  }))
}
