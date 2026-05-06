import React from 'react'

const POSITIVE_CHECKS = [
  { key: 'scenario_class_match',  label: 'Class Match',         group: 'positive' },
  { key: 'asset_match',           label: 'Asset Match',         group: 'positive' },
  { key: 'physical_signal_match', label: 'Physical Signal Match', group: 'positive' },
  { key: 'cyber_state_match',     label: 'Cyber State Match',   group: 'positive' },
  { key: 'timing_match',          label: 'Timing Match',        group: 'positive' },
  { key: 'expected_vs_observed_match', label: 'Expected vs Observed', group: 'positive' },
]

const GUARDRAIL_CHECKS = [
  { key: 'unsupported_claim_flag',       label: 'Unsupported Claim',       invert: true },
  { key: 'packet_claim_flag',            label: 'Packet-level Claim',      invert: true },
  { key: 'field_telemetry_claim_flag',   label: 'Field Telemetry Claim',   invert: true },
  { key: 'external_attacker_claim_flag', label: 'External Attacker Claim', invert: true },
  { key: 'uses_old_asset_name_flag',     label: 'Old Asset Name Used',     invert: true },
]

export default function EvidenceScorecard({ evidence, presenterMode }) {
  if (!evidence || Object.keys(evidence).length === 0) {
    return (
      <div className="card" style={{ padding: 40, textAlign: 'center', color: '#90a4ae' }}>
        No evidence matrix data available.
      </div>
    )
  }

  const overall = evidence.overall_evidence_score
  const overallPct = overall != null ? Math.round(overall * 100) : null

  const overallColor = overall == null ? '#607d8b'
    : overall >= 0.8 ? '#2e7d32'
    : overall >= 0.5 ? '#e65100'
    : '#c62828'

  return (
    <div className="card">
      <div className="card-title">Evidence Scorecard</div>

      {/* Overall score display */}
      {overallPct != null && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 24,
          marginBottom: 24,
          padding: '20px 24px',
          background: `linear-gradient(135deg, ${overallColor}22 0%, ${overallColor}08 100%)`,
          border: `2px solid ${overallColor}44`,
          borderRadius: 'var(--radius)',
        }}>
          <div style={{ textAlign: 'center', minWidth: 100 }}>
            <div style={{ fontSize: presenterMode ? 72 : 56, fontWeight: 900, lineHeight: 1, color: overallColor }}>
              {overallPct}
            </div>
            <div style={{ fontSize: 'var(--font-sm)', color: 'var(--color-subtext)', marginTop: 4 }}>
              / 100
            </div>
          </div>
          <div>
            <div style={{ fontSize: 'var(--font-title)', fontWeight: 700, color: overallColor }}>
              Overall Evidence Score
            </div>
            <div style={{ fontSize: 'var(--font-md)', color: 'var(--color-subtext)', marginTop: 4 }}>
              {overall >= 0.8 ? '✓ Strong grounded explanation' :
               overall >= 0.5 ? '~ Moderate — some evidence gaps' :
               '✗ Weak — review evidence manually'}
            </div>
          </div>
        </div>
      )}

      {/* Positive checks */}
      <div className="section-label">Accuracy Checks (higher is better)</div>
      <div className="score-grid">
        {POSITIVE_CHECKS.map(({ key, label }) => {
          const val = evidence[key]
          const passed = val === 1 || val === true
          const na = val == null
          return (
            <div key={key} className={`score-tile ${na ? 'neutral' : passed ? 'pass' : 'fail'}`}>
              <div className="score-icon">{na ? '—' : passed ? '✓' : '✗'}</div>
              <div className="score-label">{label}</div>
              <div className="score-val" style={{ color: na ? '#90a4ae' : passed ? '#2e7d32' : '#c62828' }}>
                {na ? 'N/A' : passed ? 'Pass' : 'Fail'}
              </div>
            </div>
          )
        })}
      </div>

      <div className="divider" />

      {/* Guardrail checks */}
      <div className="section-label">Guardrail Checks (0 violations = pass)</div>
      <div className="score-grid">
        {GUARDRAIL_CHECKS.map(({ key, label }) => {
          const val = evidence[key]
          const violated = val === 1 || val === true
          const na = val == null
          return (
            <div key={key} className={`score-tile ${na ? 'neutral' : !violated ? 'pass' : 'fail'}`}>
              <div className="score-icon">{na ? '—' : !violated ? '✓' : '✗'}</div>
              <div className="score-label">{label}</div>
              <div className="score-val" style={{ color: na ? '#90a4ae' : !violated ? '#2e7d32' : '#c62828' }}>
                {na ? 'N/A' : !violated ? 'Clean' : 'VIOLATED'}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
