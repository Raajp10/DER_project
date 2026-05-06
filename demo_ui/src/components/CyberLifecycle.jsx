import React from 'react'

const CLASS_COLORS = {
  cyber_physical: '#e65100',
  physical_only:  '#1565c0',
  cyber_only:     '#6a1b9a',
  normal:         '#2e7d32',
}

export default function CyberLifecycle({ timeline, scenario, presenterMode }) {
  if (!timeline || !timeline.stages || timeline.stages.length === 0) {
    return (
      <div className="card" style={{ padding: 40, textAlign: 'center', color: '#90a4ae' }}>
        No cyber lifecycle data available.
      </div>
    )
  }

  const { stages, anomaly_description, cyber_before_physical, timing_alignment_status } = timeline
  const scls = scenario?.scenario_class || 'unknown'
  const accentColor = CLASS_COLORS[scls] || '#607d8b'

  const circleSize = presenterMode ? 96 : 78
  const numSize    = presenterMode ? 30 : 24
  const lblSize    = presenterMode ? 13 : 11

  return (
    <div className="card">
      <div className="card-title">Cyber / Context Lifecycle</div>

      {/* Class + timing badges */}
      <div className="flex-row" style={{ marginBottom: 20 }}>
        <span className="badge" style={{ background: `${accentColor}20`, color: accentColor, border: `1.5px solid ${accentColor}` }}>
          {scls.replace(/_/g, ' ').toUpperCase()}
        </span>
        {timing_alignment_status && timing_alignment_status !== 'unknown' && (
          <span className="badge badge-gray" style={{ fontFamily: 'monospace', fontSize: 11 }}>
            {timing_alignment_status.replace(/_/g, ' ')}
          </span>
        )}
        {cyber_before_physical === true && (
          <span className="badge badge-orange" style={{ fontWeight: 800 }}>
            Cyber → Physical confirmed
          </span>
        )}
      </div>

      {/* Flow diagram */}
      <div className="lifecycle-wrap">
        <div className="lifecycle-flow">
          {stages.map((stage, i) => (
            <React.Fragment key={stage.key || i}>
              {i > 0 && (
                <div style={{
                  width: presenterMode ? 40 : 28, height: 3,
                  background: stage.anomaly ? '#ff5722' : '#b0bec5',
                  flexShrink: 0, position: 'relative',
                  marginBottom: presenterMode ? 28 : 22,
                }}>
                  <div style={{
                    position: 'absolute', right: -7, top: -6,
                    width: 0, height: 0,
                    borderLeft: `9px solid ${stage.anomaly ? '#ff5722' : '#b0bec5'}`,
                    borderTop: '6px solid transparent',
                    borderBottom: '6px solid transparent',
                  }} />
                </div>
              )}
              <div className="lifecycle-stage">
                {/* Circle */}
                <div
                  className={`lifecycle-circle ${stage.anomaly ? 'anomaly' : scls === 'physical_only' ? 'physical' : 'normal'}`}
                  style={{ width: circleSize, height: circleSize }}
                >
                  <div style={{ fontSize: numSize, fontWeight: 900, lineHeight: 1 }}>{stage.num}</div>
                  <div style={{ fontSize: lblSize, textAlign: 'center', lineHeight: 1.2, marginTop: 3, padding: '0 6px' }}>
                    {stage.label}
                  </div>
                </div>
              </div>
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Anomaly description */}
      {anomaly_description && (
        <div className="lifecycle-desc">
          <strong>Event:</strong> {anomaly_description}
        </div>
      )}

      {/* Extra context for physical_only */}
      {scls === 'physical_only' && (
        <div style={{ marginTop: 12, padding: '10px 14px', background: '#e3f2fd', borderRadius: 8, fontSize: 'var(--font-md)', color: '#1565c0' }}>
          <strong>Note:</strong> Cyber context remained in normal monitoring state throughout the event.
          No cyber command or flag was associated with this physical change.
        </div>
      )}
      {scls === 'normal' && (
        <div style={{ marginTop: 12, padding: '10px 14px', background: '#e8f5e9', borderRadius: 8, fontSize: 'var(--font-md)', color: '#2e7d32' }}>
          <strong>Normal operation:</strong> No anomalous cyber or physical events detected in this window.
        </div>
      )}
    </div>
  )
}
