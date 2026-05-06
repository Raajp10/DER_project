import React from 'react'

const TYPE_LABELS = {
  cyber_physical:         'Cyber-Physical',
  physical_only:          'Physical Only',
  cyber_only:             'Cyber Only',
  normal:                 'Normal',
  insufficient_evidence:  'Insufficient Evidence',
}

export default function ExplanationCard({ explanation, presenterMode }) {
  if (!explanation || !explanation.explanation_type) {
    return (
      <div className="card" style={{ padding: 40, textAlign: 'center', color: '#90a4ae' }}>
        No explanation available. LLM may not have run yet.
      </div>
    )
  }

  const {
    explanation_type, confidence,
    primary_asset, primary_physical_signals, primary_cyber_evidence,
    timing_summary, operator_summary, recommended_operator_checks,
    human_explanation, evidence_used,
  } = explanation

  const safeList = (v) => Array.isArray(v) ? v : []

  return (
    <div className="card explanation-card">
      <div className="card-title">Grounded LLM Explanation</div>

      {/* Type + confidence */}
      <div className="explanation-type-row">
        <span className={`explanation-type-chip type-${explanation_type}`}>
          {TYPE_LABELS[explanation_type] || explanation_type}
        </span>
        {confidence && (
          <span className={`badge badge-${confidence === 'high' ? 'green' : confidence === 'medium' ? 'yellow' : 'gray'}`}>
            {confidence.toUpperCase()} confidence
          </span>
        )}
        {primary_asset && (
          <span className="badge badge-blue">
            Asset: {primary_asset}
          </span>
        )}
      </div>

      {/* Human explanation */}
      {human_explanation && (
        <div className="human-explanation">
          {human_explanation}
        </div>
      )}

      {/* Physical signals */}
      {safeList(primary_physical_signals).length > 0 && (
        <div>
          <div className="section-label">Physical Signals Referenced</div>
          <div className="flex-row">
            {safeList(primary_physical_signals).map(s => (
              <span key={s} className="badge badge-blue">{s}</span>
            ))}
          </div>
        </div>
      )}

      {/* Cyber evidence */}
      {safeList(primary_cyber_evidence).length > 0 && (
        <div>
          <div className="section-label">Cyber Evidence Referenced</div>
          <div className="flex-row">
            {safeList(primary_cyber_evidence).map(s => (
              <span key={s} className="badge badge-orange">{s}</span>
            ))}
          </div>
        </div>
      )}

      {/* Timing */}
      {timing_summary && (
        <div>
          <div className="section-label">Timing</div>
          <div style={{ fontSize: 'var(--font-md)', color: 'var(--color-subtext)', fontStyle: 'italic' }}>
            {timing_summary}
          </div>
        </div>
      )}

      {/* Operator summary */}
      {operator_summary && (
        <div>
          <div className="section-label">Operator Summary</div>
          <div style={{ fontSize: 'var(--font-md)' }}>{operator_summary}</div>
        </div>
      )}

      {/* Recommended checks */}
      {safeList(recommended_operator_checks).length > 0 && (
        <div>
          <div className="section-label">Recommended Operator Checks</div>
          <div className="op-checks">
            {safeList(recommended_operator_checks).map((check, i) => (
              <div key={i} className="op-check">{check}</div>
            ))}
          </div>
        </div>
      )}

      {/* Evidence used */}
      {safeList(evidence_used).length > 0 && (
        <div>
          <div className="section-label">Evidence Fields Used</div>
          <div className="flex-row">
            {safeList(evidence_used).map(s => (
              <span key={s} className="badge badge-gray" style={{ fontFamily: 'monospace', fontSize: 11 }}>{s}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
