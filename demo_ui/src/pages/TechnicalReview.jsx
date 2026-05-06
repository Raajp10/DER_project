import React, { useState } from 'react'
import PhysicalPlot from '../components/PhysicalPlot.jsx'
import CyberLifecycle from '../components/CyberLifecycle.jsx'
import ExplanationCard from '../components/ExplanationCard.jsx'
import EvidenceScorecard from '../components/EvidenceScorecard.jsx'
import DetectorCard from '../components/DetectorCard.jsx'

const TABS = [
  { id: 'details',     label: 'Scenario Details' },
  { id: 'physical',   label: 'Physical Evidence' },
  { id: 'cyber',      label: 'Cyber Evidence' },
  { id: 'explanation', label: 'LLM Explanation' },
  { id: 'matrix',     label: 'Evidence Matrix' },
  { id: 'audit',      label: 'Raw Audit' },
]

const CLASS_COLOR = {
  cyber_physical: 'badge-orange',
  physical_only:  'badge-blue',
  cyber_only:     'badge-purple',
  normal:         'badge-green',
}

export default function TechnicalReview({ technicalScenarios }) {
  const scenarios = technicalScenarios || []
  const [selIdx, setSelIdx] = useState(0)
  const [tab, setTab] = useState('details')
  const [search, setSearch] = useState('')
  const [filterClass, setFilterClass] = useState('all')

  const filtered = scenarios.filter(s => {
    const matchClass = filterClass === 'all' || s.scenario_class === filterClass
    const matchSearch = !search ||
      (s.scenario_id || '').toLowerCase().includes(search.toLowerCase()) ||
      (s.scenario_family || '').toLowerCase().includes(search.toLowerCase()) ||
      (s.author_model || '').toLowerCase().includes(search.toLowerCase())
    return matchClass && matchSearch
  })

  const current = filtered[selIdx] || filtered[0]

  if (!scenarios.length) {
    return (
      <div className="card" style={{ padding: 40, textAlign: 'center', color: '#90a4ae' }}>
        No technical scenarios loaded. Run <code>prepare_demo_data.py</code> first.
      </div>
    )
  }

  return (
    <div className="technical-page">
      <div className="tech-layout">
        {/* Left sidebar — scenario list */}
        <div className="tech-sidebar">
          <div className="sidebar-header">
            <div style={{ fontWeight: 700, fontSize: 'var(--font-md)', marginBottom: 8 }}>
              Detections ({filtered.length} / {scenarios.length})
            </div>
            <input
              className="search-input"
              placeholder="Search scenarios..."
              value={search}
              onChange={e => { setSearch(e.target.value); setSelIdx(0) }}
            />
            <select
              className="class-filter"
              value={filterClass}
              onChange={e => { setFilterClass(e.target.value); setSelIdx(0) }}
            >
              <option value="all">All Classes</option>
              <option value="cyber_physical">Cyber-Physical</option>
              <option value="physical_only">Physical-Only</option>
              <option value="cyber_only">Cyber-Only</option>
              <option value="normal">Normal</option>
            </select>
          </div>

          <div className="scenario-list">
            {filtered.map((s, i) => {
              const score = s.evidence?.overall_evidence_score
              const scoreColor = score == null ? '#90a4ae'
                : score >= 0.8 ? '#2e7d32'
                : score >= 0.5 ? '#e65100' : '#c62828'
              const isAnomaly = s.detection?.predicted_label === 1
              return (
                <div
                  key={s.scenario_id || i}
                  className={`scenario-item ${selIdx === i ? 'selected' : ''}`}
                  onClick={() => { setSelIdx(i); setTab('details') }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ fontFamily: 'monospace', fontSize: 11, color: '#607d8b' }}>
                      {s.scenario_id || `scenario_${i}`}
                    </div>
                    <span className={`badge ${CLASS_COLOR[s.scenario_class] || 'badge-gray'}`}
                          style={{ fontSize: 9, padding: '1px 5px' }}>
                      {(s.scenario_class || '').replace(/_/g,' ')}
                    </span>
                  </div>
                  <div style={{ fontSize: 'var(--font-sm)', marginTop: 2, color: '#455a64' }}>
                    {s.scenario_family || '—'}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 11 }}>
                    <span style={{ color: isAnomaly ? '#c62828' : '#2e7d32', fontWeight: 700 }}>
                      {isAnomaly ? '⚠ ANOMALY' : '✓ NORMAL'}
                    </span>
                    {score != null && (
                      <span style={{ color: scoreColor, fontWeight: 700 }}>
                        Score: {Math.round(score * 100)}
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right panel — detail tabs */}
        <div className="tech-detail">
          {current && (
            <>
              {/* Scenario header */}
              <div className="tech-scenario-header">
                <span className="badge badge-gray" style={{ fontFamily: 'monospace' }}>
                  {current.scenario_id}
                </span>
                <span className={`badge ${CLASS_COLOR[current.scenario_class] || 'badge-gray'}`}>
                  {(current.scenario_class || '').replace(/_/g, ' ')}
                </span>
                <span className="badge badge-gray">{current.scenario_family}</span>
                <span className="badge badge-gray">by {current.author_model}</span>
                {current.evidence?.overall_evidence_score != null && (
                  <span className="badge badge-green">
                    Score: {Math.round(current.evidence.overall_evidence_score * 100)}/100
                  </span>
                )}
              </div>

              {/* Tab bar */}
              <div className="tab-bar">
                {TABS.map(t => (
                  <button
                    key={t.id}
                    className={`tab-btn ${tab === t.id ? 'active' : ''}`}
                    onClick={() => setTab(t.id)}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {/* Tab content */}
              <div className="tab-content">
                {tab === 'details'     && <TabDetails scenario={current} />}
                {tab === 'physical'    && <TabPhysical scenario={current} />}
                {tab === 'cyber'       && <TabCyber scenario={current} />}
                {tab === 'explanation' && <TabExplanation scenario={current} />}
                {tab === 'matrix'      && <TabMatrix scenario={current} />}
                {tab === 'audit'       && <TabAudit scenario={current} />}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/* ── Tab components ─────────────────────────────────────────── */

function TabDetails({ scenario }) {
  const s = scenario
  const { timing, detection, physical_evidence } = s

  return (
    <div className="flex-col" style={{ gap: 16 }}>
      <div className="two-col">
        <div className="card">
          <div className="card-title">Scenario Metadata</div>
          <table className="kv-table">
            <tbody>
              <tr><td>Scenario ID</td><td style={{ fontFamily: 'monospace' }}>{s.scenario_id}</td></tr>
              <tr><td>Class</td><td>{s.scenario_class}</td></tr>
              <tr><td>Family</td><td>{s.scenario_family}</td></tr>
              <tr><td>Author Model</td><td>{s.author_model}</td></tr>
              <tr><td>Target Asset</td><td style={{ fontFamily: 'monospace' }}>{s.target_asset_id}</td></tr>
              <tr><td>Demo Class Key</td><td style={{ fontFamily: 'monospace' }}>{s.demo_class_key}</td></tr>
              <tr><td>Demo Label</td><td>{s.demo_label}</td></tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <div className="card-title">Detection Window Timing</div>
          <table className="kv-table">
            <tbody>
              <tr><td>Window Start</td><td>{timing?.window_start_s}s</td></tr>
              <tr><td>Window End</td><td>{timing?.window_end_s}s</td></tr>
              <tr><td>Attack Start (relative)</td><td>{timing?.attack_start_rel != null ? `${timing.attack_start_rel}s` : '—'}</td></tr>
              <tr><td>Attack End (relative)</td><td>{timing?.attack_end_rel != null ? `${timing.attack_end_rel}s` : '—'}</td></tr>
              <tr><td>Event Start (abs)</td><td>{timing?.event_start_s}s</td></tr>
              <tr><td>Event End (abs)</td><td>{timing?.event_end_s}s</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {detection && <DetectorCard detection={detection} presenterMode={false} />}

      {physical_evidence && (
        <div className="card">
          <div className="card-title">Physical Evidence Summary</div>
          <table className="kv-table">
            <tbody>
              {Object.entries(physical_evidence).map(([k, v]) => (
                <tr key={k}>
                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{k}</td>
                  <td>{typeof v === 'number' ? v.toFixed(4) : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function TabPhysical({ scenario }) {
  return (
    <div className="flex-col" style={{ gap: 16 }}>
      <div className="card" style={{ padding: '12px 16px' }}>
        <div style={{ fontSize: 'var(--font-sm)', color: 'var(--color-subtext)' }}>
          Red shaded region = anomaly window. Signals extracted ±120s around event. Max 600 points (downsampled if needed).
        </div>
      </div>
      <PhysicalPlot
        timeseries={scenario.timeseries}
        presenterMode={false}
        scenario={scenario}
      />
    </div>
  )
}

function TabCyber({ scenario }) {
  const { cyber_timeline, timeseries } = scenario

  const cyberFields = timeseries
    ? Object.entries(timeseries)
        .filter(([k]) => ['cyber_state', 'cyber_anomaly_active', 'physical_effect_active',
                          'command_type', 'command_source', 'command_accepted',
                          'command_applied', 'expected_value', 'observed_value'].includes(k))
    : []

  return (
    <div className="flex-col" style={{ gap: 16 }}>
      <CyberLifecycle timeline={cyber_timeline} scenario={scenario} presenterMode={false} />

      {cyberFields.length > 0 && (
        <div className="card">
          <div className="card-title">Cyber Context Fields (window sample)</div>
          <table className="kv-table">
            <tbody>
              {cyberFields.map(([k, v]) => (
                <tr key={k}>
                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{k}</td>
                  <td>{Array.isArray(v) ? `[${v.slice(0,3).join(', ')}${v.length > 3 ? ', …' : ''}]` : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function TabExplanation({ scenario }) {
  return (
    <ExplanationCard explanation={scenario.explanation} presenterMode={false} />
  )
}

function TabMatrix({ scenario }) {
  return (
    <EvidenceScorecard evidence={scenario.evidence} presenterMode={false} />
  )
}

function TabAudit({ scenario }) {
  const auditFields = {
    scenario_id: scenario.scenario_id,
    scenario_class: scenario.scenario_class,
    scenario_family: scenario.scenario_family,
    author_model: scenario.author_model,
    target_asset_id: scenario.target_asset_id,
    demo_class_key: scenario.demo_class_key,
    detection: scenario.detection,
    timing: scenario.timing,
    physical_evidence: scenario.physical_evidence,
    evidence: scenario.evidence,
    explanation: scenario.explanation,
    cyber_timeline: scenario.cyber_timeline,
  }

  return (
    <div className="flex-col" style={{ gap: 16 }}>
      <div className="card" style={{ padding: '12px 16px' }}>
        <div style={{ fontSize: 'var(--font-sm)', color: 'var(--color-subtext)' }}>
          Raw JSON fields from the demo data packet (timeseries arrays excluded for readability).
        </div>
      </div>
      <div className="card">
        <div className="card-title">Audit Data</div>
        <pre className="json-preview">
          {JSON.stringify(auditFields, null, 2)}
        </pre>
      </div>
    </div>
  )
}
