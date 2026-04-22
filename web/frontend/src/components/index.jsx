import { useMemo } from 'react'

// ── Components ─────────────────────────────────

export function Hero() {
  return (
    <div style={{ marginBottom: 48 }}>
      <span className="label" style={{ color: 'var(--accent)', marginBottom: 8 }}>
        CSI Radar Interface
      </span>
      <h1 className="grad-text" style={{ 
        fontSize: 'clamp(28px, 6vw, 48px)', 
        lineHeight: 1.1, 
        marginBottom: 16 
      }}>
        Human Activity<br/>Recognition
      </h1>
      <p style={{ color: 'var(--muted)', fontSize: 15, maxWidth: 500, lineHeight: 1.6 }}>
        Real-time WiFi Channel State Information processing for device-free motion detection.
      </p>
    </div>
  )
}

export function SignalCard({ data }) {
  const { waveform = [], subcarrier_map = [] } = data

  const points = useMemo(() => {
    if (!waveform || waveform.length < 2) return '0,100 100,100'
    const max = Math.max(...waveform, 0.0001)
    const norm = waveform.map(v => (v / max) * 100)
    const len = norm.length
    return norm.map((h, i) => `${(i / (len - 1)) * 100},${100 - h}`).join(' ')
  }, [waveform])

  const getHeatColor = (v) => {
    // Spectral color map (Indigo -> Cyan -> Emerald)
    const opacity = 0.2 + (v * 0.8)
    if (v < 0.3) return `rgba(99, 102, 241, ${opacity})` // Indigo
    if (v < 0.7) return `rgba(6, 182, 212, ${opacity})`  // Cyan
    return `rgba(16, 185, 129, ${opacity})`             // Emerald
  }

  return (
    <div className="card" style={{ padding: 32, marginTop: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 32 }}>
        <div>
          <span className="label" style={{ marginBottom: 4 }}>Signal Spectrogram & Waveform</span>
          <h3 style={{ fontSize: 18, fontWeight: 600 }}>High-Fidelity CSI Analysis</h3>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span className="mono" style={{ fontSize: 11, color: 'var(--accent)', display: 'block' }}>HT40 · 5GHz</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--muted)' }}>114 Active Subcarriers</span>
        </div>
      </div>

      {/* Heatmap Area */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: 'flex', gap: 2, height: 40, background: 'rgba(0,0,0,0.2)', borderRadius: 4, overflow: 'hidden', padding: 4 }}>
          {subcarrier_map.slice(0, 57).map((v, i) => (
            <div key={i} style={{
              flex: 1, 
              background: getHeatColor(v),
              transition: 'background 0.1s ease',
              borderRadius: 1
            }} />
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
          <span className="mono" style={{ fontSize: 9 }}>-20MHz</span>
          <span className="mono" style={{ fontSize: 9 }}>Center (Ch 114)</span>
          <span className="mono" style={{ fontSize: 9 }}>+20MHz</span>
        </div>
      </div>

      {/* Detail Waveform (SVG) */}
      <div style={{ position: 'relative', height: 160, background: 'rgba(255,255,255,0.01)', borderRadius: 12, border: '1px solid rgba(255,255,255,0.03)', overflow: 'hidden' }}>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block' }}>
          <defs>
            <linearGradient id="waveGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.4" />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path 
            d={`M 0,100 L ${points} L 100,100 Z`}
            fill="url(#waveGrad)"
            style={{ transition: 'all 0.1s linear' }}
          />
          <polyline
            fill="none"
            stroke="var(--accent)"
            strokeWidth="0.8"
            points={points}
            style={{ transition: 'all 0.1s linear' }}
          />
        </svg>
      </div>
    </div>
  )
}


function MetricTile({ label, value, unit, color = 'var(--text)' }) {
  return (
    <div className="card" style={{ flex: '1 1 120px', padding: 20 }}>
      <span className="label" style={{ marginBottom: 8 }}>{label}</span>
      <div style={{ fontSize: 24, fontWeight: 700, color }}>
        {value}<small style={{ fontSize: 14, marginLeft: 4, fontWeight: 500, color: 'var(--muted)' }}>{unit}</small>
      </div>
    </div>
  )
}

export function MetricsRow({ data }) {
  return (
    <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
      <MetricTile label="Throughput" value={data.fps.toFixed(1)} unit="FPS" />
      <MetricTile label="Latency" value={data.latency_ms} unit="ms" />
      <MetricTile label="Loss" value={data.packet_loss.toFixed(1)} unit="%" color={data.packet_loss > 5 ? 'var(--warning)' : 'var(--text)'} />
    </div>
  )
}

export function PredictionCard({ data }) {
  const { smoothed = 'idle', confidence = 0 } = data
  const isWalk = smoothed === 'walk'

  return (
    <div className="card mb-8" style={{ borderLeft: `4px solid ${isWalk ? 'var(--accent)' : 'var(--border)'}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span className="label">Activity Inference</span>
          <h2 style={{ fontSize: 36, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {smoothed}
          </h2>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 32, fontWeight: 800, color: 'var(--accent)' }}>
            {(confidence * 100).toFixed(0)}<span style={{ fontSize: 16 }}>%</span>
          </div>
          <span className="label" style={{ marginBottom: 0 }}>Confidence</span>
        </div>
      </div>
    </div>
  )
}

const STEPS = [
  { n: '01', t: 'Null\nRemoval' },
  { n: '02', t: 'Filtering' },
  { n: '03', t: 'Features' },
  { n: '04', t: 'PCA' },
  { n: '05', t: 'Inference' },
]

export function Pipeline() {
  return (
    <div className="mb-8">
      <span className="label">DSP Pipeline</span>
      <div style={{ display: 'flex', gap: 12, overflowX: 'auto', paddingBottom: 8 }}>
        {STEPS.map((s, i) => (
          <div key={i} style={{ flex: 1, minWidth: 100 }}>
            <div className="card" style={{ padding: '16px 12px', textAlign: 'center', background: 'rgba(255,255,255,0.02)' }}>
              <div className="mono" style={{ color: 'var(--accent)', fontSize: 10, marginBottom: 4 }}>{s.n}</div>
              <div style={{ fontSize: 12, fontWeight: 600, whiteSpace: 'pre-line' }}>{s.t}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function Footer() {
  return (
    <footer style={{ 
      marginTop: 'auto', 
      paddingTop: 32, 
      borderTop: '1px solid var(--border)',
      display: 'flex',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: 16
    }}>
      <div className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>
        Hardware: ESP32-C6 · 2.4GHz
      </div>
      <div className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>
        Algorithm: SVM-RBF · Kernel Pulse
      </div>
    </footer>
  )
}

