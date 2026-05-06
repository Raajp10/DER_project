import React, { useState, useMemo } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceArea, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const SIGNAL_LABELS = {
  pv_p_kw: 'PV Power (kW)',
  bess_p_kw: 'BESS Power (kW)',
  bess_soc_percent: 'BESS SoC (%)',
  pcc_v_a_pu: 'PCC Voltage A (pu)',
  pcc_v_b_pu: 'PCC Voltage B (pu)',
  pcc_v_c_pu: 'PCC Voltage C (pu)',
  pcc_i_a_amp: 'PCC Current A (A)',
  pcc_p_kw: 'PCC Active Power (kW)',
  pcc_q_kvar: 'PCC Reactive Power (kVAr)',
  irradiance_pu: 'Irradiance (pu)',
  temperature_c: 'Temperature (°C)',
  pv_q_kvar: 'PV Reactive (kVAr)',
  bess_q_kvar: 'BESS Reactive (kVAr)',
}

const SIGNAL_COLORS = [
  '#1565c0', '#e65100', '#2e7d32', '#6a1b9a',
  '#00838f', '#ad1457', '#ef6c00', '#1b5e20',
]

export default function PhysicalPlot({ timeseries, eventStart, eventEnd, presenterMode, scenario }) {
  const ts = timeseries || {}
  const timeRel = ts.time_relative || []
  const attackStart = ts.attack_start_relative ?? eventStart
  const attackEnd   = ts.attack_end_relative ?? eventEnd

  // Find available signals
  const availSignals = useMemo(() => {
    return Object.keys(ts).filter(k => {
      if (!Array.isArray(ts[k])) return false
      if (k.startsWith('overlay_') || k === 'time_relative' || k === 'time_absolute') return false
      if (['attack_start_relative','attack_end_relative','window_start_relative','window_end_relative','buffer_start_s','n_points'].includes(k)) return false
      return ts[k].length === timeRel.length
    })
  }, [ts, timeRel.length])

  // Default: prefer scenario-relevant signals
  const affected = scenario?.physical_evidence?.affected_variables_from_scenario || []
  const topSignals = scenario?.physical_evidence?.top_signals?.map(s => s.signal) || []
  const defaultSignals = [...new Set([...affected, ...topSignals, 'pv_p_kw', 'bess_p_kw', 'bess_soc_percent'])]
    .filter(s => availSignals.includes(s)).slice(0, 3)
  const [selectedSignals, setSelectedSignals] = useState(null)
  const active = selectedSignals || defaultSignals.slice(0, 2)

  // Build chart data
  const chartData = useMemo(() => {
    return timeRel.map((t, i) => {
      const row = { t }
      for (const sig of active) {
        const arr = ts[sig]
        if (arr) row[sig] = arr[i] ?? null
      }
      return row
    })
  }, [timeRel, ts, active.join(',')])

  const chartHeight = presenterMode ? 520 : 420

  const toggleSignal = (sig) => {
    setSelectedSignals(prev => {
      const cur = prev || defaultSignals.slice(0,2)
      if (cur.includes(sig)) {
        return cur.length > 1 ? cur.filter(s => s !== sig) : cur
      }
      return [...cur, sig].slice(0, 4)
    })
  }

  if (timeRel.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: 60, color: '#90a4ae' }}>
        <div style={{ fontSize: 48, marginBottom: 12 }}>📊</div>
        <div>No time series data available for this scenario.</div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="plot-controls">
        <span className="card-title" style={{ marginBottom: 0 }}>Physical Signal</span>
        <div className="attack-legend">
          <div className="attack-swatch" />
          <span>Attack / Event Window</span>
        </div>
        <div className="ml-auto" style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {availSignals.map((sig, idx) => (
            <button
              key={sig}
              onClick={() => toggleSignal(sig)}
              style={{
                padding: '4px 10px', borderRadius: 14,
                border: `2px solid ${SIGNAL_COLORS[availSignals.indexOf(sig) % SIGNAL_COLORS.length]}`,
                background: active.includes(sig) ? SIGNAL_COLORS[availSignals.indexOf(sig) % SIGNAL_COLORS.length] : 'white',
                color: active.includes(sig) ? 'white' : SIGNAL_COLORS[availSignals.indexOf(sig) % SIGNAL_COLORS.length],
                cursor: 'pointer', fontSize: 12, fontWeight: 700,
              }}
            >
              {sig.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={chartHeight}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 40, left: 10, bottom: 40 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
          <XAxis
            dataKey="t"
            label={{ value: 'Time (seconds from buffer start)', position: 'insideBottom', offset: -20,
                     fontSize: presenterMode ? 16 : 13, fill: '#546e7a' }}
            tick={{ fontSize: presenterMode ? 14 : 11 }}
            tickFormatter={v => `${v}s`}
          />
          <YAxis
            tick={{ fontSize: presenterMode ? 14 : 11 }}
            width={presenterMode ? 70 : 55}
          />
          <Tooltip
            contentStyle={{ fontSize: presenterMode ? 14 : 12 }}
            labelFormatter={v => `t = ${v}s`}
          />
          <Legend
            wrapperStyle={{ fontSize: presenterMode ? 15 : 12, paddingTop: 8 }}
            formatter={v => SIGNAL_LABELS[v] || v}
          />

          {/* Attack window shading */}
          {attackStart != null && attackEnd != null && (
            <ReferenceArea
              x1={attackStart} x2={attackEnd}
              fill="rgba(198,40,40,0.12)"
              stroke="rgba(198,40,40,0.5)"
              strokeWidth={2}
              label={{ value: '⚠ Event Window', position: 'insideTop',
                       fontSize: presenterMode ? 15 : 12, fill: '#c62828', fontWeight: 700 }}
            />
          )}

          {/* Window start/end lines */}
          {attackStart != null && (
            <ReferenceLine x={attackStart} stroke="#c62828" strokeDasharray="6 3" strokeWidth={2} />
          )}
          {attackEnd != null && (
            <ReferenceLine x={attackEnd} stroke="#c62828" strokeDasharray="6 3" strokeWidth={2} />
          )}

          {active.map((sig, idx) => (
            <Line
              key={sig}
              type="monotone"
              dataKey={sig}
              name={sig}
              stroke={SIGNAL_COLORS[availSignals.indexOf(sig) % SIGNAL_COLORS.length]}
              strokeWidth={presenterMode ? 3.5 : 2.5}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
