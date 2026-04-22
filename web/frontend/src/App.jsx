import { useEffect, useRef, useState, useCallback } from 'react'
import Nav from './components/Nav'
import Sidebar from './components/Sidebar'
import { Hero, SignalCard, MetricsRow, PredictionCard, Pipeline, Footer } from './components/index.jsx'
import './App.css'

const WS_URL = 'ws://localhost:8000/ws'
const RECONNECT_DELAY_MS = 3000

const INITIAL_STATE = {
  label:          'idle',
  smoothed:       'idle',
  confidence:     0,
  probabilities:  { walk: 0, idle: 0 },
  fps:            0,
  latency_ms:     0,
  packet_loss:    0,
  frame_count:    0,
  waveform:       Array(60).fill(0.1),
  subcarrier_map: Array(57).fill(0.1),
  connected:      false,
  error:          '',
}

export default function App() {
  const [data, setData]       = useState(INITIAL_STATE)
  const [wsStatus, setWsStatus] = useState('connecting') 
  const wsRef     = useRef(null)
  const timerRef  = useRef(null)

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
      } catch {
        // malformed JSON
      }
    }

    ws.onerror = () => {
      setWsStatus('error')
    }

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

  return (
    <div className="app">
      <Nav wsStatus={wsStatus} data={data} />
      
      <div className="dashboard-layout">
        <Sidebar wsStatus={wsStatus} />
        
        <main className="main-content">
          <Hero />
          
          <div className="content-grid">
            <div className="grid-left">
              <PredictionCard data={data} />
              <MetricsRow data={data} />
            </div>
            
            <div className="grid-right">
              <Pipeline />
            </div>
          </div>

          {/* Full-width detailed analyzer at the bottom */}
          <SignalCard data={data} />
          
          <Footer />
        </main>
      </div>
    </div>
  )
}


