import { useEffect, useRef, useState, useCallback } from 'react'
import Nav from './components/Nav'
import Sidebar from './components/Sidebar'
import { StatusBanner, MetricsRow, PredictionCard, Pipeline, Footer, MiniFooter, ActivityLogPage, SignalViewPage, SystemInfoPage, SettingsPage } from './components/index.jsx'
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
      default:
        return (
          <div style={{ animation: 'fadeIn 0.4s ease', display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
            <StatusBanner data={data} />

            <div style={{
              maxWidth: 1000, width: '100%', margin: '0 auto',
              display: 'flex', flexDirection: 'column', justifyContent: 'center',
              flex: 1, gap: 24, overflow: 'hidden'
            }}>
              {/* Core AI Inference & Pipeline */}
              <div className="responsive-grid" style={{ flex: 1, maxHeight: '600px' }}>
                <PredictionCard data={data} />
                <Pipeline />
              </div>
            </div>
          </div>
        )
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
