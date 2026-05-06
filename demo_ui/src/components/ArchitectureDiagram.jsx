import React from 'react'

const NODES = [
  { id: 'scenario',  title: 'Zero-day Scenario', sub: '64 scenarios · 4 models',   color: '#e3f2fd', border: '#1976d2' },
  { id: 'physical',  title: 'Physical Time Series', sub: 'pv_p_kw · bess_p_kw · pcc_v_a_pu',  color: '#e8f5e9', border: '#2e7d32' },
  { id: 'cyber',     title: 'Aligned Cyber Context', sub: 'event-level only · 604,800 rows', color: '#fff3e0', border: '#e65100' },
  { id: 'detector',  title: 'Detector Window', sub: '60s / 10s stride · 22 features', color: '#fce4ec', border: '#c62828' },
  { id: 'evidence',  title: 'Evidence Packet', sub: 'physical + cyber + timing',    color: '#f3e5f5', border: '#6a1b9a' },
  { id: 'qwen',      title: 'Qwen Explanation', sub: 'local Ollama · 3B model',     color: '#fff9c4', border: '#f57f17' },
  { id: 'score',     title: 'Evidence Score', sub: '10 checks · 5 guardrails',      color: '#e0f2f1', border: '#00796b' },
]

export default function ArchitectureDiagram({ presenterMode }) {
  return (
    <div className="card">
      <div className="card-title">Pipeline Architecture</div>
      <div className="arch-flow">
        {NODES.map((node, i) => (
          <React.Fragment key={node.id}>
            {i > 0 && (
              <div className="arch-arrow">→</div>
            )}
            <div
              className="arch-node"
              style={{
                background: node.color,
                borderColor: node.border,
                borderWidth: 2,
              }}
            >
              <div className="arch-node-title" style={{ color: node.border, fontSize: presenterMode ? 15 : 13 }}>
                {node.title}
              </div>
              <div className="arch-node-sub" style={{ fontSize: presenterMode ? 12 : 10 }}>
                {node.sub}
              </div>
            </div>
          </React.Fragment>
        ))}
      </div>
      <div style={{ marginTop: 12, fontSize: 'var(--font-sm)', color: 'var(--color-subtext)', fontStyle: 'italic' }}>
        All physical/cyber/context evidence is from real Phase 2 generated data.
        Only the detector score is a placeholder (demo mode).
      </div>
    </div>
  )
}
