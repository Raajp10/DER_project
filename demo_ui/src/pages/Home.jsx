import React from 'react'

export default function Home({ onNavigate, summary }) {
  const s = summary || {}

  const badges = [
    { label: 'Phase 2 zero-day dataset ready', color: 'badge-blue' },
    { label: 'Phase 3 explanation demo ready',  color: 'badge-green' },
    { label: 'Qwen2.5 local LLM',               color: 'badge-purple' },
    { label: 'Event-level cyber context only',   color: 'badge-orange' },
    { label: 'Demo mode — placeholder detection scores', color: 'badge-yellow' },
  ]

  return (
    <div className="home-page">
      {/* Hero */}
      <div className="home-hero">
        <h1 className="home-title">
          DER Zero-Day Anomaly<br />Explanation Demo
        </h1>
        <p className="home-subtitle">
          Physical + cyber/context evidence with grounded local LLM explanation
        </p>
        <div className="home-badges">
          {badges.map(b => (
            <span key={b.label} className={`badge ${b.color}`}>{b.label}</span>
          ))}
        </div>
      </div>

      {/* Mode cards */}
      <div className="home-cards">
        <div className="home-demo-card" onClick={() => onNavigate('presenter')} style={{ cursor: 'pointer' }}>
          <div style={{ fontSize: 48 }}>🎤</div>
          <div className="home-demo-card-title" style={{ color: '#1976d2' }}>
            Live Presenter Demo
          </div>
          <div className="home-demo-card-sub">
            Guided 5–7 minute audience walkthrough with 8 story steps.
            Large fonts, visual charts, and lifecycle diagrams.
            Includes 4 demo scenarios: Cyber-Physical, Physical-Only, Cyber-Only, and Normal.
          </div>
          <button className="btn btn-primary" style={{ marginTop: 8 }}>
            Start Presenter Demo →
          </button>
        </div>

        <div className="home-demo-card" onClick={() => onNavigate('technical')} style={{ cursor: 'pointer' }}>
          <div style={{ fontSize: 48 }}>🔬</div>
          <div className="home-demo-card-title" style={{ color: '#2e7d32' }}>
            Technical Review Demo
          </div>
          <div className="home-demo-card-sub">
            Inspect all {s.total_demo_scenarios || 15} detections in detail.
            6 tabs: scenario details, physical evidence, cyber evidence,
            LLM explanation, evidence matrix, and raw audit.
          </div>
          <button className="btn" style={{ background: '#2e7d32', color: 'white', marginTop: 8 }}>
            Open Technical Review →
          </button>
        </div>
      </div>

      {/* Stats */}
      {s.average_evidence_score != null && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-title">Pipeline Status</div>
          <div className="overview-stats">
            <div className="stat-card">
              <div className="stat-num">64</div>
              <div className="stat-label">Phase 2 zero-day scenarios</div>
            </div>
            <div className="stat-card">
              <div className="stat-num">{s.total_demo_scenarios || 15}</div>
              <div className="stat-label">Demo detections</div>
            </div>
            <div className="stat-card">
              <div className="stat-num" style={{ color: '#2e7d32' }}>
                {Math.round((s.average_evidence_score || 0) * 100)}
              </div>
              <div className="stat-label">Avg evidence score (/ 100)</div>
            </div>
            <div className="stat-card">
              <div className="stat-num" style={{ color: '#2e7d32' }}>0</div>
              <div className="stat-label">Guardrail violations</div>
            </div>
            <div className="stat-card">
              <div className="stat-num">4</div>
              <div className="stat-label">Authors (ChatGPT, Claude, Gemini, Grok)</div>
            </div>
          </div>

          <div style={{ marginTop: 20, padding: '14px 18px', background: '#fff3e0', borderRadius: 8,
                        border: '1.5px solid #ffcc80', fontSize: 'var(--font-sm)' }}>
            <strong>Demo mode:</strong> Anomaly detection scores are placeholder values.
            Physical signals, cyber context, and LLM explanations use real Phase 2/3 generated data.
            Replace <code>smoke_model_detections.csv</code> with real frozen-model outputs to enable
            fully real evaluation.
          </div>
        </div>
      )}
    </div>
  )
}
