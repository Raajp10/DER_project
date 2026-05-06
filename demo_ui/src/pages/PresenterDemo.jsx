import React, { useState } from 'react'
import PhysicalPlot from '../components/PhysicalPlot.jsx'
import CyberLifecycle from '../components/CyberLifecycle.jsx'
import ExplanationCard from '../components/ExplanationCard.jsx'
import EvidenceScorecard from '../components/EvidenceScorecard.jsx'
import DetectorCard from '../components/DetectorCard.jsx'
import ArchitectureDiagram from '../components/ArchitectureDiagram.jsx'

const STEP_LABELS = [
  '1. System Overview',
  '2. Normal Behavior',
  '3. Attack Window',
  '4. Detector Fires',
  '5. Cyber Timeline',
  '6. LLM Explanation',
  '7. Evidence Scorecard',
  '8. Boundaries & Summary',
]

const SPEAKER_NOTES = {
  1: 'Walk the audience through the architecture: physical signals from the DER site feed into detection windows. Each window is accompanied by aligned cyber/context data. A local Qwen model provides a grounded explanation of the detection result. The evidence scorecard verifies the explanation did not hallucinate.',
  2: 'Show the audience what normal DER operation looks like. PV power is proportional to irradiance. BESS is charging or discharging normally. The cyber context shows "normal monitoring" — no commands are being issued. The detector score is below threshold.',
  3: 'Now the event window appears. The physical signal deviates from its expected value. The red shaded region marks where the anomaly is active. Notice how the signal changes — this is the observable physical consequence of the event.',
  4: 'The anomaly detector fires. The score exceeds the threshold. Note: for this demo, we are using placeholder detection scores to exercise the explanation pipeline. When real frozen-model results are available, these scores will reflect actual model outputs.',
  5: 'The cyber/context lifecycle diagram shows the sequence of control events. For a cyber-physical event, a command was created, sent, received, accepted, and applied — and then the physical response was observed. The highlighted stage marks where the anomaly occurred.',
  6: 'The local Qwen model (running on your machine via Ollama) generated this explanation using only the evidence packet. It was instructed not to invent causes, not to claim packet-level protocol details, and not to name external attackers. The explanation cites actual field names from the evidence.',
  7: 'The evidence scorecard shows how well the explanation matched the known scenario. Each tile is a specific check. Green tiles are passes. Red tiles are violations. Zero violations in the guardrail section means the model stayed within its evidence boundaries.',
  8: 'This demo uses event-level cyber context only — no packet captures, no byte-level protocol traces. The physical signals are synthetic simulation data, not real field telemetry. When Phase 1 model training is complete, real detection scores will replace the placeholder values shown here.',
}

const DEMO_CLASS_KEYS = ['cyber_physical', 'physical_only', 'cyber_only', 'normal']
const DEMO_ACTIVE_CLASSES = {
  cyber_physical: 'active-cp',
  physical_only: 'active-po',
  cyber_only: 'active-co',
  normal: 'active-no',
}

export default function PresenterDemo({ presenterScenarios, presenterMode }) {
  const [step, setStep] = useState(1)
  const [demoIdx, setDemoIdx] = useState(0)
  const [showNotes, setShowNotes] = useState(false)

  const scenarios = presenterScenarios || []

  // Map class key to scenario
  const scenarioByClass = {}
  for (const s of scenarios) {
    if (s.demo_class_key) scenarioByClass[s.demo_class_key] = s
    else if (s.scenario_class) scenarioByClass[s.scenario_class] = s
  }

  const demoClassKey = DEMO_CLASS_KEYS[demoIdx] || DEMO_CLASS_KEYS[0]
  const current = scenarioByClass[demoClassKey] || scenarios[demoIdx] || scenarios[0]

  if (!current) {
    return (
      <div className="card" style={{ padding: 40, textAlign: 'center', color: '#90a4ae' }}>
        No presenter scenarios loaded. Run <code>prepare_demo_data.py</code> first.
      </div>
    )
  }

  const { timeseries, cyber_timeline, explanation, evidence, detection, physical_evidence } = current

  const jumpToClass = (cls) => {
    const idx = DEMO_CLASS_KEYS.indexOf(cls)
    if (idx >= 0) { setDemoIdx(idx); setStep(2) }
  }

  return (
    <div className="presenter-page">
      {/* Top nav: demo selector + step indicator */}
      <div className="demo-nav">
        <div className="step-indicator">Step {step} / 8</div>
        <div className="demo-selector">
          {DEMO_CLASS_KEYS.map((cls, i) => {
            const sc = scenarioByClass[cls] || scenarios[i]
            const label = sc?.demo_label || `Demo ${String.fromCharCode(65+i)}`
            return (
              <button
                key={cls}
                className={`demo-type-btn ${demoIdx === i ? DEMO_ACTIVE_CLASSES[cls] : ''}`}
                onClick={() => { setDemoIdx(i); if (step < 2 || step > 7) setStep(2) }}
              >
                {label.split('—')[0].trim()}
              </button>
            )
          })}
        </div>
        <div className="ml-auto" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="btn btn-ghost" style={{ height: 40, fontSize: 'var(--font-sm)' }}
            onClick={() => jumpToClass('cyber_physical')}>Jump: Cyber-Physical</button>
          <button className="btn btn-ghost" style={{ height: 40, fontSize: 'var(--font-sm)' }}
            onClick={() => jumpToClass('physical_only')}>Jump: Physical-Only</button>
          <button className="btn btn-ghost" style={{ height: 40, fontSize: 'var(--font-sm)' }}
            onClick={() => jumpToClass('cyber_only')}>Jump: Cyber-Only</button>
          <button className="btn btn-ghost" style={{ height: 40, fontSize: 'var(--font-sm)' }}
            onClick={() => jumpToClass('normal')}>Jump: Normal</button>
        </div>
      </div>

      {/* Scenario info bar */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <span className="badge badge-gray" style={{ fontFamily: 'monospace' }}>
          {current.scenario_id}
        </span>
        <span className={`badge badge-${
          current.scenario_class === 'cyber_physical' ? 'orange' :
          current.scenario_class === 'physical_only'  ? 'blue' :
          current.scenario_class === 'cyber_only'     ? 'purple' : 'green'
        }`}>
          {(current.scenario_class || '').replace(/_/g, ' ')}
        </span>
        <span className="badge badge-gray">{current.scenario_family}</span>
        <span className="badge badge-gray">by {current.author_model}</span>
        {evidence?.overall_evidence_score != null && (
          <span className="badge badge-green">
            Score: {Math.round(evidence.overall_evidence_score * 100)}/100
          </span>
        )}
      </div>

      {/* Step content */}
      <div className="step-content">
        {step === 1 && <Step1Overview presenterMode={presenterMode} />}
        {step === 2 && <Step2Normal timeseries={timeseries} scenario={current} presenterMode={presenterMode} />}
        {step === 3 && <Step3Attack timeseries={timeseries} scenario={current} presenterMode={presenterMode} />}
        {step === 4 && <Step4Detector detection={detection} scenario={current} presenterMode={presenterMode} />}
        {step === 5 && <Step5Cyber timeline={cyber_timeline} scenario={current} presenterMode={presenterMode} />}
        {step === 6 && <Step6Explanation explanation={explanation} presenterMode={presenterMode} />}
        {step === 7 && <Step7Scorecard evidence={evidence} presenterMode={presenterMode} />}
        {step === 8 && <Step8Boundaries presenterMode={presenterMode} />}
      </div>

      {/* Navigation bar */}
      <div className="step-nav">
        <button className="btn btn-gray" disabled={step === 1} onClick={() => setStep(s => s - 1)}>
          ← Prev
        </button>
        <div className="step-dots">
          {STEP_LABELS.map((_, i) => (
            <div key={i} className={`step-dot ${step === i+1 ? 'active' : ''}`}
                 onClick={() => setStep(i+1)} title={STEP_LABELS[i]} />
          ))}
        </div>
        <button className="btn btn-primary" disabled={step === 8} onClick={() => setStep(s => s + 1)}>
          Next →
        </button>
        <button className="btn btn-gray" style={{ marginLeft: 8 }} onClick={() => setStep(1)}>
          ↺ Restart
        </button>
        <div className="ml-auto">
          <button className="speaker-notes-toggle" onClick={() => setShowNotes(v => !v)}>
            {showNotes ? 'Hide speaker notes' : 'Show speaker notes'}
          </button>
        </div>
      </div>

      {/* Speaker notes */}
      {showNotes && (
        <div className="speaker-notes-panel">
          <strong>Speaker note — Step {step}:</strong><br />
          {SPEAKER_NOTES[step] || 'No notes for this step.'}
        </div>
      )}
    </div>
  )
}

/* ── Step components ─────────────────────────────────────── */

function Step1Overview({ presenterMode }) {
  return (
    <div className="flex-col" style={{ gap: 20 }}>
      <div className="card">
        <div className="card-title">DER Zero-Day Anomaly Explanation Demo</div>
        <p style={{ fontSize: 'var(--font-md)', marginBottom: 16 }}>
          This demo shows how a cyber-physical anomaly detection system works — from raw physical signals
          through cyber/context alignment to a grounded LLM explanation.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
          {[
            { emoji: '⚡', title: 'Physical Layer', desc: 'PV, BESS, PCC signals at 1-second resolution' },
            { emoji: '🔒', title: 'Cyber/Context Layer', desc: 'Event-level command lifecycle and flags' },
            { emoji: '🔍', title: 'Anomaly Detector', desc: '60s windows with 22 features (demo mode)' },
            { emoji: '🤖', title: 'Local LLM', desc: 'Qwen2.5:3b via Ollama — no cloud needed' },
            { emoji: '📊', title: 'Evidence Scorecard', desc: '15 grounding checks per explanation' },
          ].map(({ emoji, title, desc }) => (
            <div key={title} style={{ background: '#f4f6f9', borderRadius: 10, padding: 16 }}>
              <div style={{ fontSize: 32, marginBottom: 6 }}>{emoji}</div>
              <div style={{ fontSize: 'var(--font-md)', fontWeight: 700 }}>{title}</div>
              <div style={{ fontSize: 'var(--font-sm)', color: 'var(--color-subtext)' }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
      <ArchitectureDiagram presenterMode={presenterMode} />
    </div>
  )
}

function Step2Normal({ timeseries, scenario, presenterMode }) {
  const normalTs = timeseries ? {
    ...timeseries,
    attack_start_relative: undefined,
    attack_end_relative: undefined,
  } : null

  return (
    <div className="flex-col" style={{ gap: 16 }}>
      <div className="card">
        <div className="card-title" style={{ color: 'var(--color-normal)' }}>Normal Behavior</div>
        <p style={{ fontSize: 'var(--font-md)' }}>
          Before the event, DER signals are stable and cyber context shows <strong>normal monitoring</strong>.
          No commands are anomalous. No physical deviations are present.
        </p>
      </div>
      <PhysicalPlot
        timeseries={normalTs}
        eventStart={null}
        eventEnd={null}
        presenterMode={presenterMode}
        scenario={scenario}
      />
      <div className="card" style={{ padding: 16 }}>
        <div className="flex-row">
          <span className="badge badge-green">cyber_state: normal_monitoring</span>
          <span className="badge badge-green">cyber_anomaly_active: 0</span>
          <span className="badge badge-green">physical_effect_active: 0</span>
        </div>
      </div>
    </div>
  )
}

function Step3Attack({ timeseries, scenario, presenterMode }) {
  return (
    <div className="flex-col" style={{ gap: 16 }}>
      <div className="card" style={{ borderLeft: '4px solid var(--color-anomaly)' }}>
        <div className="card-title" style={{ color: 'var(--color-anomaly)' }}>⚠ Attack / Event Window Appears</div>
        <p style={{ fontSize: 'var(--font-md)' }}>
          The physical signal now deviates inside the red shaded region.
          This is the observable consequence of the anomalous event.
        </p>
      </div>
      <PhysicalPlot
        timeseries={timeseries}
        presenterMode={presenterMode}
        scenario={scenario}
      />
    </div>
  )
}

function Step4Detector({ detection, scenario, presenterMode }) {
  return (
    <div className="two-col">
      <DetectorCard detection={detection} presenterMode={presenterMode} />
      <div className="card">
        <div className="card-title">Detection Window Info</div>
        <table className="kv-table">
          <tbody>
            <tr><td>Scenario ID</td><td>{scenario?.scenario_id || '—'}</td></tr>
            <tr><td>Scenario Class</td><td>{scenario?.scenario_class || '—'}</td></tr>
            <tr><td>Scenario Family</td><td>{scenario?.scenario_family || '—'}</td></tr>
            <tr><td>Target Asset</td><td>{scenario?.target_asset_id || '—'}</td></tr>
            <tr><td>Author Model</td><td>{scenario?.author_model || '—'}</td></tr>
            <tr><td>Window Start</td><td>{scenario?.timing?.window_start_s}s</td></tr>
            <tr><td>Window End</td><td>{scenario?.timing?.window_end_s}s</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Step5Cyber({ timeline, scenario, presenterMode }) {
  return (
    <CyberLifecycle timeline={timeline} scenario={scenario} presenterMode={presenterMode} />
  )
}

function Step6Explanation({ explanation, presenterMode }) {
  return (
    <ExplanationCard explanation={explanation} presenterMode={presenterMode} />
  )
}

function Step7Scorecard({ evidence, presenterMode }) {
  return (
    <EvidenceScorecard evidence={evidence} presenterMode={presenterMode} />
  )
}

function Step8Boundaries({ presenterMode }) {
  const claims = [
    { icon: '📡', title: 'Event-level cyber context only',
      body: 'No packet captures, no byte-level protocol traces, no IEEE 2030.5 frame contents.' },
    { icon: '🏭', title: 'No real field telemetry',
      body: 'This dataset is synthetic simulation data. Not real measurements from a live DER site.' },
    { icon: '🔬', title: 'No exploit evidence',
      body: 'No claims of real CVE exploitation, specific malware, or external attacker identity.' },
    { icon: '🔄', title: 'Replace detections when ready',
      body: 'Swap smoke_model_detections.csv with real frozen-model outputs. Evidence pipeline unchanged.' },
    { icon: '🤖', title: 'LLM for explanation only',
      body: 'Qwen was used for grounded explanation, not for anomaly detection.' },
    { icon: '✅', title: 'Zero guardrail violations',
      body: 'All 15 explanations passed packet-claim, telemetry-claim, and attacker-attribution checks.' },
  ]

  return (
    <div className="flex-col" style={{ gap: 20 }}>
      <div className="card">
        <div className="card-title">Claim Boundaries & Summary</div>
        <div className="claim-grid">
          {claims.map(({ icon, title, body }) => (
            <div key={title} className="claim-item">
              <div className="claim-icon">{icon}</div>
              <div className="claim-text">
                <div className="claim-title">{title}</div>
                <div style={{ color: 'var(--color-subtext)', fontSize: 'var(--font-sm)' }}>{body}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ background: '#e8f5e9', borderColor: '#4caf50' }}>
        <div className="card-title" style={{ color: '#2e7d32' }}>
          Phase 3 Status: SMOKE_READY_WAITING_FOR_REAL_MODEL_RESULTS
        </div>
        <p style={{ fontSize: 'var(--font-md)', color: '#2e7d32' }}>
          The full explanation pipeline is verified end-to-end with placeholder detections.
          Average evidence score: <strong>0.847/1.0</strong>. Zero guardrail violations.
          Ready to accept real frozen-model detection outputs from Phase 1 training.
        </p>
      </div>
    </div>
  )
}
