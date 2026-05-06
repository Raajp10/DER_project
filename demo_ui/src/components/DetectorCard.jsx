import React from 'react'

export default function DetectorCard({ detection, presenterMode }) {
  if (!detection) return null

  const { anomaly_score, threshold, predicted_label, display_name } = detection
  const score = anomaly_score ?? 0
  const thresh = threshold ?? 0.5
  const isAnomaly = predicted_label === 1

  const pct = Math.min(100, Math.round(score * 100))
  const threshPct = Math.min(100, Math.round(thresh * 100))

  const barColor = isAnomaly
    ? `hsl(${Math.max(0, 20 - (pct - threshPct) * 0.5)}, 90%, 45%)`
    : '#4caf50'

  return (
    <div className="card detector-card">
      <div className="card-title">Detector Output</div>

      {/* Mode badge */}
      <div className="detector-mode-badge">
        ⚠ {display_name || 'Demo mode — placeholder detection scores'}
      </div>

      {/* Score bar */}
      <div className="score-bar-wrap">
        <div className="score-bar-label">
          <span>Anomaly Score</span>
          <span style={{ fontFamily: 'monospace', fontSize: 'var(--font-title)', fontWeight: 900 }}>
            {score.toFixed(3)}
          </span>
        </div>
        <div className="score-bar-track">
          <div
            className="score-bar-fill"
            style={{ width: `${pct}%`, background: barColor }}
          />
          <div
            className="score-bar-threshold"
            style={{ left: `${threshPct}%` }}
            title={`Threshold: ${thresh}`}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#90a4ae', marginTop: 4 }}>
          <span>0.0</span>
          <span style={{ color: '#455a64', fontWeight: 700 }}>
            Threshold: {thresh.toFixed(2)}
          </span>
          <span>1.0</span>
        </div>
      </div>

      {/* Predicted label */}
      <div className={`predicted-label ${isAnomaly ? 'label-anomaly' : 'label-normal'}`}>
        {isAnomaly ? '⚠  ANOMALY DETECTED' : '✓  NORMAL — No anomaly'}
      </div>

      {/* Score vs threshold comparison */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12,
        marginTop: 4,
      }}>
        <div style={{ textAlign: 'center', background: '#eceff1', borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 11, color: '#90a4ae', fontWeight: 700 }}>SCORE</div>
          <div style={{ fontSize: presenterMode ? 32 : 26, fontWeight: 900, color: barColor }}>
            {score.toFixed(3)}
          </div>
        </div>
        <div style={{ textAlign: 'center', background: '#eceff1', borderRadius: 8, padding: 12 }}>
          <div style={{ fontSize: 11, color: '#90a4ae', fontWeight: 700 }}>THRESHOLD</div>
          <div style={{ fontSize: presenterMode ? 32 : 26, fontWeight: 900, color: '#455a64' }}>
            {thresh.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  )
}
