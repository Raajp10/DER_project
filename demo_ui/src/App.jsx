import React, { useState, useEffect } from 'react'
import Home from './pages/Home.jsx'
import PresenterDemo from './pages/PresenterDemo.jsx'
import TechnicalReview from './pages/TechnicalReview.jsx'
import { loadDemoData } from './utils/dataLoader.js'

export default function App() {
  const [view, setView] = useState('home')
  const [presenterMode, setPresenterMode] = useState(true)
  const [data, setData] = useState(null)
  const [dataSource, setDataSource] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDemoData().then(result => {
      setData(result)
      setDataSource(result.source)
      setLoading(false)
    })
  }, [])

  const navigate = (v) => setView(v)

  const appClass = presenterMode ? 'presenter-mode' : ''

  if (loading) {
    return (
      <div className={`app-shell ${appClass}`} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
        <div style={{ textAlign: 'center', color: '#90a4ae' }}>
          <div style={{ fontSize: 36, marginBottom: 16 }}>⚡</div>
          <div style={{ fontSize: 'var(--font-md)' }}>Loading demo data…</div>
        </div>
      </div>
    )
  }

  return (
    <div className={`app-shell ${appClass}`}>
      {/* Header */}
      <header className="app-header">
        <div
          className="app-logo"
          onClick={() => setView('home')}
          style={{ cursor: 'pointer' }}
        >
          ⚡ DER Anomaly Demo
        </div>

        <nav className="app-nav">
          <button
            className={`nav-btn ${view === 'home' ? 'active' : ''}`}
            onClick={() => setView('home')}
          >
            Home
          </button>
          <button
            className={`nav-btn ${view === 'presenter' ? 'active' : ''}`}
            onClick={() => setView('presenter')}
          >
            🎤 Presenter Demo
          </button>
          <button
            className={`nav-btn ${view === 'technical' ? 'active' : ''}`}
            onClick={() => setView('technical')}
          >
            🔬 Technical Review
          </button>
        </nav>

        <div className="app-header-right">
          <label className="presenter-toggle" title="Toggle large-font presenter mode">
            <input
              type="checkbox"
              checked={presenterMode}
              onChange={e => setPresenterMode(e.target.checked)}
            />
            <span>Presenter Mode</span>
          </label>
        </div>
      </header>

      {/* Fallback banner */}
      {dataSource && dataSource !== 'live' && (
        <div className={`fallback-banner ${dataSource === 'hardcoded' ? 'fallback-hardcoded' : 'fallback-json'}`}>
          {dataSource === 'fallback'
            ? '⚠ Using fallback JSON data (live fetch failed). Run prepare_demo_data.py to regenerate.'
            : '⚠ Using hardcoded stub data (no data files found). Run prepare_demo_data.py to load real results.'}
        </div>
      )}

      {/* Main content */}
      <main className="app-main">
        {view === 'home' && (
          <Home onNavigate={navigate} summary={data?.summary} />
        )}
        {view === 'presenter' && (
          <PresenterDemo
            presenterScenarios={data?.presenter}
            presenterMode={presenterMode}
          />
        )}
        {view === 'technical' && (
          <TechnicalReview
            technicalScenarios={data?.technical}
          />
        )}
      </main>
    </div>
  )
}
