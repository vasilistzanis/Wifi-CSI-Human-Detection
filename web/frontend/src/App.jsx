import { useEffect, useRef, useState, useCallback } from 'react'
import Nav from './components/Nav'
import Sidebar from './components/Sidebar'
import { StatusBanner, MetricsRow, PredictionCard, Pipeline, Footer, MiniFooter, ActivityLogPage, SignalViewPage, SystemInfoPage, SettingsPage, SignalHealthCard, MiniActivityFeed, MiniSignalCard } from './components/index.jsx'
import './App.css'

const WS_URL = 'ws://localhost:8000/ws'
const RECONNECT_DELAY_MS = 3000
const MAX_LOG_ENTRIES = 200

const INITIAL_STATE = {
  label: 'idle',
  smoothed: 'idle',
  confidence: 0,
  probabilities: { walk: 0, idle: 0 },
  fps: 0,
  latency: 0,
  loss: 0,
  frame_count: 0,
  waveform: Array(60).fill(0),
  subcarrier_map: Array(57).fill(0),
  connected: false,
  error: '',
}

export default function App() {
  const [data, setData] = useState(INITIAL_STATE)
  const [wsStatus, setWsStatus] = useState('connecting')
  const [activePage, setActivePage] = useState('monitor')
  const [activityLog, setActivityLog] = useState([])
  const wsRef = useRef(null)
  const timerRef = useRef(null)
  const lastLabel = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setWsStatus('connecting')
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setWsStatus('live')
      console.log('✅ WebSocket connected')
    }

    ws.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data)
        if (payload.heartbeat) return
        setData(prev => ({ ...prev, ...payload }))

        // Log activity changes (or every ~2s even if unchanged)
        if (payload.smoothed && payload.confidence) {
          const now = new Date()
          const entry = {
            time: now.toLocaleTimeString('el-GR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            timestamp: now.getTime(),
            activity: payload.smoothed,
            raw: payload.label || payload.smoothed,
            confidence: payload.confidence,
            fps: payload.fps || 0,
            frame: payload.frame_count || 0,
          }
          // Only log if activity changed or every 30th frame
          if (payload.smoothed !== lastLabel.current || (payload.frame_count % 30 === 0)) {
            lastLabel.current = payload.smoothed
            setActivityLog(prev => [entry, ...prev].slice(0, MAX_LOG_ENTRIES))
          }
        }
      } catch {
        // malformed JSON
      }
    }

    ws.onerror = () => setWsStatus('error')

    ws.onclose = () => {
      setWsStatus('error')
      console.warn('⚠️ WebSocket closed — reconnecting in 3s…')
      timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const renderPage = () => {
    switch (activePage) {
      case 'signal':
        return <SignalViewPage data={data} />
      case 'activity':
        return <ActivityLogPage log={activityLog} onClear={() => setActivityLog([])} />
      case 'system':
        return <SystemInfoPage data={data} />
      case 'settings':
        return <SettingsPage />
      default: {
        const uptime = data.connected ? (() => {
          const s = Math.floor((Date.now() - (data.start_time || Date.now())) / 1000)
          const m = Math.floor(s / 60)
          const h = Math.floor(m / 60)
          return h > 0 ? `${h}h ${m % 60}m` : `${m}m ${s % 60}s`
        })() : '-'

        return (
          <div style={{ animation: 'fadeIn 0.4s ease', display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
            <StatusBanner data={data} />

            <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 340px', gap: 12, overflow: 'hidden', minHeight: 0 }}>

              {/* LEFT COLUMN: Signal on top (fills), Telemetry strip bottom */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0, overflow: 'hidden' }}>
                <MiniSignalCard data={data} style={{ flex: 1, minHeight: 0 }} />
                <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
                  {[
                    { label: 'Frames', value: data.connected ? (data.frame_count || 0).toLocaleString() : '-', icon: '🔢' },
                    { label: 'Uptime', value: uptime, icon: '⏱️' },
                    { label: 'Model', value: data.connected ? (data.model_name || 'None') : '-', icon: '🧠' },
                    { label: 'Interface', value: data.connected ? (data.port || 'Auto') : '-', icon: '🔌' },
                  ].map(t => (
                    <div key={t.label} className="card" style={{ flex: 1, padding: '8px 10px', display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 10 }}>{t.icon}</span>
                      <div>
                        <div className="label" style={{ fontSize: 7, marginBottom: 0 }}>{t.label}</div>
                        <div className="mono" style={{ fontSize: 11, fontWeight: 700 }}>{t.value}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* RIGHT COLUMN: Everything stacked */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0, overflow: 'hidden' }}>
                {/* Prediction */}
                <PredictionCard data={data} style={{ flexShrink: 0 }} />

                {/* Health + Metrics row */}
                <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
                  <SignalHealthCard data={data} />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flex: 1 }}>
                    {[
                      { l: '⚡ FPS', v: data.connected ? data.fps.toFixed(0) : '-', c: data.connected ? 'var(--accent)' : 'var(--muted)' },
                      { l: '📡 Loss', v: data.connected ? `${(data.loss||0).toFixed(1)}%` : '-', c: data.connected ? ((data.loss||0) > 5 ? 'var(--danger)' : 'var(--success)') : 'var(--muted)' },
                    ].map(m => (
                      <div key={m.l} className="card" style={{ flex: 1, padding: '8px 12px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                        <div className="label" style={{ fontSize: 7, marginBottom: 2 }}>{m.l}</div>
                        <div className="mono" style={{ fontSize: 16, fontWeight: 800, color: m.c }}>{m.v}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Activity Feed fills remaining */}
                <MiniActivityFeed log={activityLog} style={{ flex: 1, minHeight: 0 }} />
              </div>
            </div>
          </div>
        )
      }
    }
  }

  return (
    <div className="app">
      <Nav wsStatus={wsStatus} data={data} />

      <div className="dashboard-layout">
        <Sidebar activePage={activePage} onNavigate={setActivePage} data={data} />

        <main className="main-content">
          {renderPage()}
          <MiniFooter />
        </main>
      </div>
    </div>
  )
}
