import { useMemo, useState, useEffect } from 'react'

// ── Status Banner ──────────────────────────────
export function StatusBanner({ data }) {
  if (!data.error && data.connected) return null
  const isError = !data.connected && data.error
  const type = isError ? 'error' : 'warning'
  const icon = isError ? '⚠️' : '🔄'
  const message = data.error || 'Connecting to backend server...'

  return (
    <div className={`status-banner ${type}`}>
      <span className="banner-icon">{icon}</span>
      <span>{message}</span>
    </div>
  )
}

// ── Prediction Card ────────────────────────────
export function PredictionCard({ data, style = {} }) {
  const { smoothed = 'idle', confidence = 0, probabilities = {}, label = 'idle' } = data
  const hasModel = Object.keys(probabilities).length > 0 && confidence > 0

  const themes = {
    walk: { color: '#818cf8', icon: '🚶', bg: 'rgba(99,102,241,0.06)' },
    idle: { color: '#34d399', icon: '🧍', bg: 'rgba(52,211,153,0.04)' },
    sit: { color: '#fbbf24', icon: '🪑', bg: 'rgba(251,191,36,0.04)' },
    fall: { color: '#f87171', icon: '⚡', bg: 'rgba(248,113,113,0.04)' },
  }
  const t = themes[smoothed] || themes.idle

  return (
    <div className="card" style={{ borderLeft: `3px solid ${t.color}`, background: t.bg, animation: 'fadeIn 0.4s ease', ...style }}>
      {/* Header */}
      <span className="label">Activity Inference</span>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: hasModel ? 18 : 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 32 }}>{t.icon}</span>
          <div>
            <h2 style={{ fontSize: 28, textTransform: 'uppercase', letterSpacing: '0.05em', lineHeight: 1 }}>{smoothed}</h2>
            {!hasModel && (
              <span className="mono" style={{ fontSize: 10, color: 'var(--muted)', marginTop: 3, display: 'block' }}>
                {label === 'Hardware Live' ? '📡 Streaming — No ML model' : `Raw: ${label}`}
              </span>
            )}
            {hasModel && label !== smoothed && (
              <span className="mono" style={{ fontSize: 10, color: 'var(--muted)', marginTop: 3, display: 'block' }}>Raw: {label}</span>
            )}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 30, fontWeight: 800, color: t.color, lineHeight: 1 }}>
            {hasModel ? (confidence * 100).toFixed(0) : '—'}
            {hasModel && <span style={{ fontSize: 14, fontWeight: 600 }}>%</span>}
          </div>
          <span className="label" style={{ marginBottom: 0, marginTop: 2, fontSize: 9 }}>Confidence</span>
        </div>
      </div>

      {/* Confidence bar + probabilities */}
      {hasModel && (
        <>
          <div style={{ height: 5, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden', marginBottom: 16 }}>
            <div style={{
              height: '100%', width: `${Math.min(confidence * 100, 100)}%`,
              background: `linear-gradient(90deg, ${t.color}, ${t.color}80)`,
              borderRadius: 3, transition: 'width 0.4s ease',
              boxShadow: `0 0 10px ${t.color}30`,
            }} />
          </div>
          <span className="label" style={{ marginBottom: 8, fontSize: 9 }}>Class Probabilities</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {Object.entries(probabilities).sort((a, b) => b[1] - a[1]).map(([cls, prob]) => {
              const ct = themes[cls] || { color: 'var(--muted)' }
              const win = cls === smoothed
              return (
                <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span className="mono" style={{ width: 44, fontSize: 10, fontWeight: win ? 700 : 500, textTransform: 'uppercase', color: win ? ct.color : 'var(--muted)' }}>{cls}</span>
                  <div style={{ flex: 1, height: 5, background: 'rgba(255,255,255,0.04)', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${(prob * 100).toFixed(1)}%`, background: win ? ct.color : 'rgba(255,255,255,0.1)', borderRadius: 3, transition: 'width 0.3s ease' }} />
                  </div>
                  <span className="mono" style={{ width: 40, textAlign: 'right', fontSize: 10, color: win ? 'var(--text)' : 'var(--muted)' }}>{(prob * 100).toFixed(1)}%</span>
                </div>
              )
            })}
          </div>
        </>
      )}

      {!hasModel && data.connected && (
        <div style={{ marginTop: 14, padding: '10px 14px', background: 'rgba(255,255,255,0.02)', borderRadius: 8, border: '1px solid var(--border)' }}>
          <span className="mono" style={{ fontSize: 10, color: 'var(--muted)', lineHeight: 1.6, display: 'block' }}>
            💡 Place trained models in <span style={{ color: 'var(--accent)' }}>web/backend/models/</span>
          </span>
        </div>
      )}
    </div>
  )
}

// ── Metrics Row (vertical stack) ───────────────
function MetricTile({ label, value, unit, color = 'var(--text)', icon }) {
  return (
    <div className="card" style={{ padding: '16px 18px', flex: '1 1 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
        {icon && <span style={{ fontSize: 11 }}>{icon}</span>}
        <span className="label" style={{ marginBottom: 0, fontSize: 9 }}>{label}</span>
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color, letterSpacing: '-0.02em' }}>
        {value}<span style={{ fontSize: 12, marginLeft: 3, fontWeight: 500, color: 'var(--muted)' }}>{unit}</span>
      </div>
    </div>
  )
}

export function MetricsRow({ data }) {
  const lc = data.packet_loss > 5 ? 'var(--danger)' : data.packet_loss > 2 ? 'var(--warning)' : 'var(--success)'
  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
      <MetricTile label="Throughput" value={data.fps.toFixed(1)} unit="FPS" icon="⚡" />
      <MetricTile label="Latency" value={data.latency_ms} unit="ms" icon="🕐" />
      <MetricTile label="Loss" value={data.packet_loss.toFixed(1)} unit="%" color={lc} icon="📡" />
    </div>
  )
}

// ── Pipeline Steps ─────────────────────────────
const STEPS = [
  { n: '01', t: 'Null\nRemoval', icon: '🧹' },
  { n: '02', t: 'Butterworth\nFilter', icon: '📊' },
  { n: '03', t: 'Temporal\nDiff', icon: '📐' },
  { n: '04', t: 'PCA\nReduction', icon: '🔬' },
  { n: '05', t: 'ML\nInference', icon: '🧠' },
]

export function Pipeline() {
  return (
    <div className="card" style={{ animation: 'fadeIn 0.6s ease' }}>
      <span className="label">DSP Pipeline</span>
      <div style={{ display: 'flex', gap: 6 }}>
        {STEPS.map((s, i) => (
          <div key={i} style={{ flex: 1, textAlign: 'center', position: 'relative' }}>
            <div style={{
              padding: '12px 4px',
              background: 'rgba(255,255,255,0.02)',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--border)'
            }}>
              <div style={{ fontSize: 14, marginBottom: 4 }}>{s.icon}</div>
              <div className="mono" style={{ color: 'var(--accent)', fontSize: 8, marginBottom: 2, fontWeight: 600 }}>STEP {s.n}</div>
              <div style={{ fontSize: 10, fontWeight: 600, whiteSpace: 'pre-line', lineHeight: 1.3, color: 'var(--text-secondary)' }}>{s.t}</div>
            </div>
            {i < STEPS.length - 1 && (
              <span style={{ position: 'absolute', right: -6, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted)', fontSize: 10, zIndex: 2 }}>›</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Signal Card (Full-width, large) ────────────
export function SignalCard({ data }) {
  const { waveform = [], subcarrier_map = [] } = data

  const points = useMemo(() => {
    if (!waveform || waveform.length < 2) return '0,100 100,100'
    const max = Math.max(...waveform, 0.0001)
    const norm = waveform.map(v => (v / max) * 100)
    const len = norm.length
    return norm.map((h, i) => `${(i / (len - 1)) * 100},${100 - h}`).join(' ')
  }, [waveform])

  const heatColor = (v) => {
    if (v < 0.25) return `rgba(99, 102, 241, ${0.15 + v * 2})`
    if (v < 0.5) return `rgba(56, 189, 248, ${0.2 + v})`
    if (v < 0.75) return `rgba(52, 211, 153, ${0.3 + v * 0.7})`
    return `rgba(251, 191, 36, ${0.4 + v * 0.6})`
  }

  return (
    <div className="card" style={{ padding: 28, animation: 'fadeIn 0.7s ease' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 22 }}>
        <div>
          <span className="label" style={{ marginBottom: 4 }}>Real-Time Signal Analysis</span>
          <h3 style={{ fontSize: 16, fontWeight: 700 }}>CSI Waveform & Spectrogram</h3>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <div style={{ textAlign: 'right' }}>
            <span className="mono" style={{ fontSize: 10, color: 'var(--accent)', display: 'block', fontWeight: 500 }}>HT40 · 5GHz</span>
            <span className="mono" style={{ fontSize: 9, color: 'var(--muted)' }}>114 Subcarriers</span>
          </div>
        </div>
      </div>

      {/* Two-panel layout: Heatmap left, info right */}
      <div style={{ display: 'flex', gap: 20, marginBottom: 20, alignItems: 'stretch' }}>
        {/* Heatmap */}
        <div style={{ flex: 1 }}>
          <span className="label" style={{ fontSize: 9, marginBottom: 6 }}>Subcarrier Power Distribution</span>
          <div style={{ display: 'flex', gap: 1.5, height: 48, background: 'rgba(0,0,0,0.25)', borderRadius: 8, overflow: 'hidden', padding: 4 }}>
            {subcarrier_map.slice(0, 57).map((v, i) => (
              <div key={i} style={{ flex: 1, background: heatColor(v), transition: 'background 0.15s ease', borderRadius: 2 }} />
            ))}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
            <span className="mono" style={{ fontSize: 8, color: 'var(--muted)' }}>-20MHz</span>
            <span className="mono" style={{ fontSize: 8, color: 'var(--muted)' }}>Center</span>
            <span className="mono" style={{ fontSize: 8, color: 'var(--muted)' }}>+20MHz</span>
          </div>
        </div>
        {/* Color Legend */}
        <div style={{ width: 100, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 4 }}>
          <span className="label" style={{ fontSize: 9, marginBottom: 2 }}>Power Level</span>
          {[
            { label: 'High', color: 'rgba(251, 191, 36, 0.9)' },
            { label: 'Medium', color: 'rgba(52, 211, 153, 0.7)' },
            { label: 'Low', color: 'rgba(56, 189, 248, 0.5)' },
            { label: 'Quiet', color: 'rgba(99, 102, 241, 0.3)' },
          ].map(item => (
            <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: item.color, flexShrink: 0 }} />
              <span className="mono" style={{ fontSize: 9, color: 'var(--muted)' }}>{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Waveform — large */}
      <div>
        <span className="label" style={{ fontSize: 9, marginBottom: 6 }}>Amplitude Over Time</span>
        <div style={{
          position: 'relative', height: 280,
          background: 'linear-gradient(180deg, rgba(99,102,241,0.03) 0%, rgba(0,0,0,0.18) 100%)',
          borderRadius: 12, border: '1px solid rgba(255,255,255,0.04)', overflow: 'hidden'
        }}>
          {/* Y-axis */}
          <div style={{ position: 'absolute', top: 0, left: 0, bottom: 20, width: 32, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '6px 0', zIndex: 2 }}>
            {['1.0', '.75', '.50', '.25', '0'].map((v, i) => (
              <span key={i} className="mono" style={{ fontSize: 8, color: 'var(--muted)', textAlign: 'right', paddingRight: 4, opacity: 0.5 }}>{v}</span>
            ))}
          </div>

          {/* Grid */}
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0, opacity: 0.1 }}>
            {[20, 40, 60, 80].map(y => <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="white" strokeWidth="0.15" strokeDasharray="1.5,2.5" />)}
            {[25, 50, 75].map(x => <line key={`v${x}`} x1={x} y1="0" x2={x} y2="100" stroke="white" strokeWidth="0.08" strokeDasharray="1,4" />)}
          </svg>

          {/* Wave */}
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block', position: 'relative' }}>
            <defs>
              <linearGradient id="wg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.35" />
                <stop offset="50%" stopColor="var(--accent-vivid)" stopOpacity="0.08" />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="lg" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
                <stop offset="40%" stopColor="var(--accent)" stopOpacity="1" />
                <stop offset="100%" stopColor="#34d399" stopOpacity="1" />
              </linearGradient>
              <filter id="glow"><feGaussianBlur stdDeviation="1.2" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            </defs>
            <path d={`M 0,100 L ${points} L 100,100 Z`} fill="url(#wg)" />
            <polyline fill="none" stroke="url(#lg)" strokeWidth="0.55" strokeLinejoin="round" points={points} filter="url(#glow)" />
            {waveform.length > 1 && (() => {
              const max = Math.max(...waveform, 0.0001)
              const lastY = 100 - (waveform[waveform.length - 1] / max) * 100
              return <circle cx="100" cy={lastY} r="1.3" fill="#34d399" opacity="0.9">
                <animate attributeName="r" values="1;1.8;1" dur="1.5s" repeatCount="indefinite" />
              </circle>
            })()}
          </svg>

          {/* Time axis */}
          <div style={{ position: 'absolute', bottom: 3, left: 32, right: 6, display: 'flex', justifyContent: 'space-between' }}>
            <span className="mono" style={{ fontSize: 8, color: 'var(--muted)', opacity: 0.45 }}>-60 frames</span>
            <span className="mono" style={{ fontSize: 8, color: 'var(--muted)', opacity: 0.45 }}>-30</span>
            <span className="mono" style={{ fontSize: 8, color: 'var(--muted)', opacity: 0.45 }}>now</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Live Signal Widget (Mini Waveform) ────────
export function LiveSignalWidget({ data }) {
  const { waveform = [] } = data
  const points = useMemo(() => {
    if (!waveform || waveform.length < 2) return '0,100 100,100'
    const max = Math.max(...waveform, 0.0001)
    const norm = waveform.map(v => (v / max) * 100)
    const len = norm.length
    return norm.map((h, i) => `${(i / (len - 1)) * 100},${100 - h}`).join(' ')
  }, [waveform])

  return (
    <div className="card" style={{ padding: 20, display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <span className="label" style={{ marginBottom: 0 }}>Live Signal Flow</span>
        <span className="mono" style={{ fontSize: 9, color: 'var(--accent)' }}>Amplitude</span>
      </div>
      <div style={{
        flex: 1, minHeight: 140, position: 'relative',
        background: 'linear-gradient(180deg, rgba(99,102,241,0.02) 0%, rgba(0,0,0,0.15) 100%)',
        borderRadius: 8, border: '1px solid rgba(255,255,255,0.03)', overflow: 'hidden'
      }}>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block' }}>
          <defs>
            <linearGradient id="mwg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.4" />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={`M 0,100 L ${points} L 100,100 Z`} fill="url(#mwg)" />
          <polyline fill="none" stroke="var(--accent)" strokeWidth="0.8" strokeLinejoin="round" points={points} />
        </svg>
      </div>
    </div>
  )
}

// ── Recent Activity Widget ──────────────
const miniColors = {
  walk: { color: '#818cf8' },
  idle: { color: '#34d399' },
  sit: { color: '#fbbf24' },
  fall: { color: '#f87171' },
}

export function RecentActivityWidget({ log }) {
  const recent = log.slice(0, 5)

  return (
    <div className="card" style={{ padding: 20, display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <span className="label" style={{ marginBottom: 0 }}>Recent Activity</span>
        <span className="mono" style={{ fontSize: 9, color: 'var(--muted)' }}>Last {recent.length} events</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flex: 1 }}>
        {recent.length === 0 ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--muted)', fontSize: 11 }}>
            Waiting for activity...
          </div>
        ) : (
          recent.map((entry, i) => {
            const color = (miniColors[entry.activity] || miniColors.idle).color
            return (
              <div key={`${entry.timestamp}-${i}`} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '8px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: 6,
                borderLeft: `2px solid ${color}`,
                animation: i === 0 ? 'fadeSlideIn 0.3s ease' : 'none'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span className="mono" style={{ fontSize: 10, color: 'var(--muted)' }}>{entry.time.split(':').slice(1).join(':')}</span>
                  <span style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', color }}>
                    {entry.activity}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{ width: 40, height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 2 }}>
                    <div style={{ width: `${entry.confidence * 100}%`, height: '100%', background: color, borderRadius: 2 }} />
                  </div>
                  <span className="mono" style={{ fontSize: 9, color: 'var(--text-secondary)', width: 24, textAlign: 'right' }}>
                    {(entry.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

// ── AI Confidence Trend ──────────────────────────────
export function ConfidenceSparkline({ log }) {
  const points = useMemo(() => {
    if (log.length < 2) return '0,100 100,100'
    const recent = log.slice(0, 25).map(e => e.confidence).reverse()
    const len = recent.length
    return recent.map((c, i) => `${(i / (len - 1)) * 100},${100 - (c * 100)}`).join(' ')
  }, [log])

  const currentConf = log.length > 0 ? log[0].confidence : 0

  return (
    <div className="card" style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <span className="label" style={{ marginBottom: 0 }}>Model Confidence</span>
        <span className="mono" style={{ fontSize: 16, fontWeight: 700, color: currentConf > 0.8 ? 'var(--success)' : currentConf > 0.5 ? 'var(--warning)' : 'var(--muted)' }}>
          {(currentConf * 100).toFixed(0)}%
        </span>
      </div>
      <div style={{ flex: 1, minHeight: 60, position: 'relative' }}>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block', overflow: 'visible' }}>
          <defs>
            <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={`M 0,100 L ${points} L 100,100 Z`} fill="url(#cg)" />
          <polyline fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" points={points} />
        </svg>
      </div>
      <span className="mono" style={{ fontSize: 9, color: 'var(--muted)', marginTop: 12, textAlign: 'right' }}>Last 25 frames</span>
    </div>
  )
}

// ── Session Distribution ──────────────────────────────
export function SessionDistribution({ log }) {
  const total = log.length
  const dist = {}
  log.forEach(e => { dist[e.activity] = (dist[e.activity] || 0) + 1 })
  const distEntries = Object.entries(dist).sort((a, b) => b[1] - a[1])

  return (
    <div className="card" style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', height: '100%' }}>
      <span className="label" style={{ marginBottom: 20 }}>Session Overview</span>
      {total === 0 ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: 11 }}>No data yet</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, flex: 1, justifyContent: 'center' }}>
          {distEntries.map(([act, count]) => {
            const pct = (count / total) * 100
            const color = (activityColors[act] || activityColors.idle).color
            return (
              <div key={act}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>{act}</span>
                  <span className="mono" style={{ fontSize: 10, fontWeight: 700 }}>{pct.toFixed(1)}%</span>
                </div>
                <div style={{ width: '100%', height: 4, background: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2 }} />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Footer ─────────────────────────────────────
export function Footer() {
  return (
    <footer style={{
      marginTop: 32, paddingTop: 20,
      borderTop: '1px solid var(--border)',
      display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, opacity: 0.6,
    }}>
      <div className="mono" style={{ fontSize: 10, color: 'var(--muted)' }}>Hardware: ESP32-C6 · 2.4GHz · HT40</div>
      <div className="mono" style={{ fontSize: 10, color: 'var(--muted)' }}>Algorithm: SVM-RBF · Butterworth + PCA Pipeline</div>
    </footer>
  )
}


// ═══════════════════════════════════════════════
// PAGE: Activity Log
// ═══════════════════════════════════════════════
const activityColors = {
  walk: { bg: 'rgba(99,102,241,0.12)', color: '#818cf8', border: 'rgba(99,102,241,0.25)' },
  idle: { bg: 'rgba(52,211,153,0.10)', color: '#34d399', border: 'rgba(52,211,153,0.25)' },
  sit: { bg: 'rgba(251,191,36,0.10)', color: '#fbbf24', border: 'rgba(251,191,36,0.25)' },
  fall: { bg: 'rgba(248,113,113,0.12)', color: '#f87171', border: 'rgba(248,113,113,0.25)' },
}

function ActivityBadge({ activity }) {
  const t = activityColors[activity] || activityColors.idle
  return (
    <span style={{
      display: 'inline-block',
      padding: '3px 10px',
      borderRadius: 6,
      fontSize: 11,
      fontWeight: 700,
      textTransform: 'uppercase',
      letterSpacing: '0.06em',
      background: t.bg,
      color: t.color,
      border: `1px solid ${t.border}`,
    }}>
      {activity}
    </span>
  )
}

export function ActivityLogPage({ log, onClear }) {
  const [filter, setFilter] = useState('all')

  const filtered = filter === 'all' ? log : log.filter(e => e.activity === filter)
  const activities = [...new Set(log.map(e => e.activity))]

  // Stats
  const totalEntries = log.length
  const avgConf = totalEntries > 0 ? (log.reduce((s, e) => s + e.confidence, 0) / totalEntries) : 0

  // Distribution counts
  const dist = {}
  log.forEach(e => { dist[e.activity] = (dist[e.activity] || 0) + 1 })
  const distEntries = Object.entries(dist).sort((a, b) => b[1] - a[1])
  const dominant = distEntries.length > 0 ? distEntries[0] : null

  const icons = { walk: '🚶', idle: '🧍', sit: '🪑', fall: '⚡' }

  return (
    <div style={{ animation: 'fadeIn 0.4s ease' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <span className="label" style={{ marginBottom: 4 }}>Activity Log</span>
          <h2 style={{ fontSize: 22, fontWeight: 700 }}>Prediction History</h2>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {totalEntries > 0 && (
            <button onClick={onClear} style={{
              padding: '6px 14px', borderRadius: 8, border: '1px solid var(--border)',
              background: 'rgba(255,255,255,0.03)', color: 'var(--muted)', fontSize: 11,
              fontWeight: 600, cursor: 'pointer', transition: 'var(--transition)', fontFamily: 'inherit',
            }}
              onMouseEnter={e => { e.target.style.borderColor = 'rgba(248,113,113,0.3)'; e.target.style.color = '#f87171' }}
              onMouseLeave={e => { e.target.style.borderColor = 'var(--border)'; e.target.style.color = 'var(--muted)' }}
            >
              🗑️ Clear All
            </button>
          )}
        </div>
      </div>

      {/* Summary Stats Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, marginBottom: 20 }}>
        {/* Total */}
        <div className="card" style={{ padding: '16px 18px' }}>
          <span className="label" style={{ fontSize: 9, marginBottom: 6 }}>Total Predictions</span>
          <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--accent)', lineHeight: 1 }}>{totalEntries}</div>
          <span className="mono" style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4, display: 'block' }}>
            {filtered.length !== totalEntries ? `${filtered.length} shown` : 'all shown'}
          </span>
        </div>
        {/* Dominant */}
        <div className="card" style={{ padding: '16px 18px' }}>
          <span className="label" style={{ fontSize: 9, marginBottom: 6 }}>Dominant Activity</span>
          {dominant ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 24 }}>{icons[dominant[0]] || '❓'}</span>
              <div>
                <div style={{ fontSize: 18, fontWeight: 700, textTransform: 'uppercase', color: (activityColors[dominant[0]] || activityColors.idle).color }}>
                  {dominant[0]}
                </div>
                <span className="mono" style={{ fontSize: 10, color: 'var(--muted)' }}>
                  {((dominant[1] / totalEntries) * 100).toFixed(0)}% of all
                </span>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--muted)' }}>—</div>
          )}
        </div>
        {/* Avg Confidence */}
        <div className="card" style={{ padding: '16px 18px' }}>
          <span className="label" style={{ fontSize: 9, marginBottom: 6 }}>Avg Confidence</span>
          <div style={{ fontSize: 28, fontWeight: 800, color: avgConf > 0.8 ? 'var(--success)' : avgConf > 0.5 ? 'var(--warning)' : 'var(--muted)', lineHeight: 1 }}>
            {totalEntries > 0 ? `${(avgConf * 100).toFixed(0)}%` : '—'}
          </div>
          <div style={{ height: 4, background: 'rgba(255,255,255,0.04)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${avgConf * 100}%`, background: avgConf > 0.8 ? 'var(--success)' : 'var(--warning)', borderRadius: 2, transition: 'width 0.5s ease' }} />
          </div>
        </div>
      </div>

      {/* Activity Distribution */}
      {distEntries.length > 0 && (
        <div className="card" style={{ padding: '16px 18px', marginBottom: 20 }}>
          <span className="label" style={{ fontSize: 9, marginBottom: 10 }}>Activity Distribution</span>
          <div style={{ display: 'flex', gap: 2, height: 28, borderRadius: 6, overflow: 'hidden' }}>
            {distEntries.map(([act, count]) => {
              const t = activityColors[act] || activityColors.idle
              const pct = (count / totalEntries) * 100
              return (
                <div key={act} title={`${act}: ${count} (${pct.toFixed(1)}%)`} style={{
                  width: `${pct}%`, minWidth: 2,
                  background: t.color, opacity: 0.8,
                  transition: 'width 0.5s ease',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }} >
                  {pct > 12 && <span style={{ fontSize: 9, fontWeight: 700, color: '#000', textTransform: 'uppercase' }}>{act}</span>}
                </div>
              )
            })}
          </div>
          {/* Legend */}
          <div style={{ display: 'flex', gap: 16, marginTop: 10 }}>
            {distEntries.map(([act, count]) => {
              const t = activityColors[act] || activityColors.idle
              return (
                <div key={act} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: t.color }} />
                  <span className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>{act}</span>
                  <span className="mono" style={{ fontSize: 10, color: 'var(--muted)' }}>({count})</span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Filter tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
        <button onClick={() => setFilter('all')} style={{
          padding: '5px 14px', borderRadius: 6, border: '1px solid',
          borderColor: filter === 'all' ? 'var(--accent)' : 'var(--border)',
          background: filter === 'all' ? 'var(--accent-soft)' : 'transparent',
          color: filter === 'all' ? 'var(--accent)' : 'var(--muted)',
          fontSize: 11, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
        }}>
          All ({totalEntries})
        </button>
        {activities.map(act => {
          const t = activityColors[act] || activityColors.idle
          const count = dist[act] || 0
          return (
            <button key={act} onClick={() => setFilter(act)} style={{
              padding: '5px 14px', borderRadius: 6, border: '1px solid',
              borderColor: filter === act ? t.border : 'var(--border)',
              background: filter === act ? t.bg : 'transparent',
              color: filter === act ? t.color : 'var(--muted)',
              fontSize: 11, fontWeight: 600, cursor: 'pointer', textTransform: 'uppercase', fontFamily: 'inherit',
            }}>
              {act} ({count})
            </button>
          )
        })}
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        {/* Header */}
        <div style={{
          display: 'grid', gridTemplateColumns: '80px 95px 1fr 80px 70px',
          padding: '10px 18px', borderBottom: '1px solid var(--border)',
          background: 'rgba(255,255,255,0.02)',
        }}>
          {['Time', 'Activity', 'Confidence', 'Raw', 'Frame'].map(h => (
            <span key={h} className="label" style={{ marginBottom: 0, fontSize: 9 }}>{h}</span>
          ))}
        </div>

        {/* Scrollable body — fixed viewport */}
        <div style={{ height: 420, overflowY: 'auto' }}>
          {filtered.length === 0 ? (
            <div style={{ padding: 60, textAlign: 'center', color: 'var(--muted)' }}>
              <div style={{ fontSize: 48, marginBottom: 12, opacity: 0.4 }}>📋</div>
              <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>No activity recorded yet</div>
              <div className="mono" style={{ fontSize: 11, lineHeight: 1.6 }}>
                Predictions will stream here in real-time<br />
                once the ESP32 is connected and the system is running.
              </div>
            </div>
          ) : (
            filtered.map((entry, i) => {
              const rowColor = activityColors[entry.activity] || activityColors.idle
              return (
                <div key={`${entry.timestamp}-${i}`} style={{
                  display: 'grid', gridTemplateColumns: '80px 95px 1fr 80px 70px',
                  padding: '9px 18px', alignItems: 'center',
                  borderBottom: '1px solid rgba(255,255,255,0.025)',
                  borderLeft: i === 0 ? `3px solid ${rowColor.color}` : '3px solid transparent',
                  transition: 'all 0.2s ease',
                  animation: i === 0 ? 'fadeSlideIn 0.3s ease' : 'none',
                  background: i === 0 ? 'rgba(255,255,255,0.015)' : 'transparent',
                }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.025)'}
                  onMouseLeave={e => e.currentTarget.style.background = i === 0 ? 'rgba(255,255,255,0.015)' : 'transparent'}
                >
                  <span className="mono" style={{ fontSize: 11, color: i === 0 ? 'var(--text)' : 'var(--text-secondary)' }}>{entry.time}</span>
                  <ActivityBadge activity={entry.activity} />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingRight: 12 }}>
                    <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.04)', borderRadius: 2, overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', width: `${(entry.confidence * 100)}%`,
                        background: rowColor.color, borderRadius: 2,
                      }} />
                    </div>
                    <span className="mono" style={{ fontSize: 10, color: i === 0 ? 'var(--text)' : 'var(--text-secondary)', width: 36, textAlign: 'right', fontWeight: i === 0 ? 600 : 400 }}>
                      {(entry.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <span className="mono" style={{ fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase' }}>{entry.raw}</span>
                  <span className="mono" style={{ fontSize: 10, color: 'var(--muted)' }}>#{entry.frame}</span>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}


// ═══════════════════════════════════════════════
// PAGE: Signal View (Expanded)
// ═══════════════════════════════════════════════
export function SignalViewPage({ data }) {
  const { waveform = [], subcarrier_map = [], fps = 0, latency = 0, loss = 0 } = data

  const points = useMemo(() => {
    if (!waveform || waveform.length < 2) return '0,100 100,100'
    const max = Math.max(...waveform, 0.0001)
    const norm = waveform.map(v => (v / max) * 100)
    const len = norm.length
    return norm.map((h, i) => `${(i / (len - 1)) * 100},${100 - h}`).join(' ')
  }, [waveform])

  const heatColor = (v) => {
    if (v < 0.01) return 'rgba(255,255,255,0.02)'
    if (v < 0.25) return `rgba(99, 102, 241, ${0.15 + v * 2})`
    if (v < 0.5) return `rgba(56, 189, 248, ${0.2 + v})`
    if (v < 0.75) return `rgba(52, 211, 153, ${0.3 + v * 0.7})`
    return `rgba(251, 191, 36, ${0.4 + v * 0.6})`
  }

  const hasData = waveform.some(v => v > 0)
  const avg = hasData ? (waveform.reduce((a, b) => a + b, 0) / waveform.length) : 0
  const peak = hasData ? Math.max(...waveform) : 0

  const stats = [
    { label: 'Throughput', value: fps.toFixed(1), unit: 'FPS', icon: '⚡', color: 'var(--accent)' },
    { label: 'Latency', value: latency.toFixed(0), unit: 'ms', icon: '🕐', color: 'var(--text)' },
    { label: 'Packet Loss', value: loss.toFixed(1), unit: '%', icon: '📡', color: loss > 5 ? 'var(--danger)' : 'var(--success)' },
    { label: 'Avg Amp', value: avg.toFixed(3), unit: '', icon: '📊', color: 'var(--text-secondary)' },
    { label: 'Peak Amp', value: peak.toFixed(3), unit: '', icon: '📈', color: 'var(--success)' },
  ]

  return (
    <div style={{ animation: 'fadeIn 0.4s ease', height: 'calc(100vh - 230px)', display: 'flex', flexDirection: 'column', gap: 14, overflow: 'hidden' }}>
      {/* Metrics Bar */}
      <div style={{ display: 'flex', gap: 10 }}>
        {stats.map(s => (
          <div key={s.label} className="card" style={{ flex: 1, padding: '10px 14px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
              <span style={{ fontSize: 10 }}>{s.icon}</span>
              <span className="label" style={{ fontSize: 8, marginBottom: 0 }}>{s.label}</span>
            </div>
            <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: s.color }}>
              {s.value}<span style={{ fontSize: 9, marginLeft: 2, color: 'var(--muted)', fontWeight: 400 }}>{s.unit}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Heatmap Section */}
      <div className="card" style={{ padding: '14px 18px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <span className="label" style={{ marginBottom: 0 }}>Subcarrier Power Distribution</span>
          <div style={{ display: 'flex', gap: 10 }}>
            <span className="mono" style={{ fontSize: 8, color: 'var(--muted)' }}>CH 114 (5.57GHz)</span>
            <span className="mono" style={{ fontSize: 8, color: 'var(--accent)' }}>HT40 Mode</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 1, height: 28, background: 'rgba(0,0,0,0.2)', borderRadius: 4, overflow: 'hidden', padding: 2 }}>
          {subcarrier_map.slice(0, 57).map((v, i) => (
            <div key={i} style={{ flex: 1, background: heatColor(v), transition: 'background 0.15s ease', borderRadius: 1 }} />
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          <span className="mono" style={{ fontSize: 8, color: 'var(--muted)' }}>-20MHz</span>
          <span className="mono" style={{ fontSize: 8, color: 'var(--muted)' }}>CENTER</span>
          <span className="mono" style={{ fontSize: 8, color: 'var(--muted)' }}>+20MHz</span>
        </div>
      </div>

      {/* Waveform Section */}
      <div className="card" style={{ padding: '14px 18px', flex: 1, display: 'flex', flexDirection: 'column' }}>
        <span className="label" style={{ marginBottom: 8 }}>Live Signal Waveform (Amplitude)</span>
        <div style={{ flex: 1, position: 'relative', background: 'rgba(0,0,0,0.15)', borderRadius: 10, border: '1px solid rgba(255,255,255,0.03)', overflow: 'hidden' }}>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0, opacity: 0.1 }}>
            {[25, 50, 75].map(y => <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="white" strokeWidth="0.1" strokeDasharray="2,2" />)}
          </svg>
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%', display: 'block' }}>
            <defs>
              <linearGradient id="swg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.3" />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d={`M 0,100 L ${points} L 100,100 Z`} fill="url(#swg)" />
            <polyline fill="none" stroke="var(--accent)" strokeWidth="0.6" strokeLinejoin="round" points={points} />
          </svg>
        </div>
      </div>
    </div>
  )
}

export function SystemInfoPage({ data }) {
  const [uptime, setUptime] = useState(0)

  useEffect(() => {
    const start = Date.now()
    const timer = setInterval(() => setUptime(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [])

  const formatUptime = (s) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m}m ${sec}s`
  }

  const sections = [
    {
      title: 'WiFi Radar Engine',
      icon: '📡',
      items: [
        { label: 'Chipset', value: 'ESP32-C6' },
        { label: 'Band', value: '2.4/5GHz (HT40)' },
        { label: 'Antennas', value: '1x1 SISO' },
        { label: 'Security', value: 'WPA3-AES' }
      ]
    },
    {
      title: 'Inference Engine',
      icon: '🧠',
      items: [
        { label: 'Model', value: 'SVM-RBF' },
        { label: 'Feature', value: 'PCA (95%)' },
        { label: 'Window', value: '1.0s (100Hz)' },
        { label: 'Latent', value: '8 Dims' }
      ]
    },
    {
      title: 'Software Stack',
      icon: '💻',
      items: [
        { label: 'Frontend', value: 'React/Vite' },
        { label: 'Backend', value: 'FastAPI/Py' },
        { label: 'Protocol', value: 'WebSocket' },
        { label: 'Build', value: 'v2.4.0' }
      ]
    },
    {
      title: 'Connection Meta',
      icon: '🔌',
      items: [
        { label: 'Port', value: '/dev/ttyUSB0' },
        { label: 'Baud', value: '115,200' },
        { label: 'Latency', value: `${(data.latency || 0).toFixed(1)}ms` },
        { label: 'Status', value: data.connected ? 'STABLE' : 'LOST', color: data.connected ? 'var(--success)' : 'var(--danger)' }
      ]
    }
  ]

  return (
    <div style={{ animation: 'fadeIn 0.5s ease', height: 'calc(100vh - 230px)', display: 'flex', flexDirection: 'column', gap: 16, overflow: 'hidden' }}>
      {/* Hero Header & Live Stats */}
      <div className="card" style={{ padding: '20px 28px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'linear-gradient(90deg, rgba(99,102,241,0.04) 0%, transparent 100%)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ position: 'relative' }}>
            <div style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--success)' }} />
            <div style={{
              position: 'absolute', top: -3, left: -3, width: 18, height: 18,
              borderRadius: '50%', border: '2px solid var(--success)',
              animation: 'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite'
            }} />
          </div>
          <div>
            <span className="label" style={{ fontSize: 9, marginBottom: 2 }}>System Health</span>
            <div style={{ fontSize: 16, fontWeight: 800 }}>NOMINAL STATUS</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 40 }}>
          <div>
            <span className="label" style={{ fontSize: 8, marginBottom: 4 }}>Live Uptime</span>
            <div className="mono" style={{ fontSize: 16, fontWeight: 700 }}>{formatUptime(uptime)}</div>
          </div>
          <div>
            <span className="label" style={{ fontSize: 8, marginBottom: 4 }}>CPU Usage</span>
            <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent)' }}>12.4%</div>
          </div>
          <div>
            <span className="label" style={{ fontSize: 8, marginBottom: 4 }}>Buffer Health</span>
            <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--success)' }}>98%</div>
          </div>
        </div>
      </div>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gridTemplateRows: 'repeat(2, 1fr)', gap: 16, flex: 1 }}>
        {sections.map(s => (
          <div key={s.title} className="card" style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <span style={{ fontSize: 18 }}>{s.icon}</span>
              <h3 style={{ fontSize: 14, fontWeight: 800, color: 'var(--text)' }}>{s.title}</h3>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 20px' }}>
              {s.items.map(item => (
                <div key={item.label}>
                  <div style={{ fontSize: 8, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>{item.label}</div>
                  <div className="mono" style={{ fontSize: 12, color: item.color || 'var(--text-secondary)', fontWeight: 600 }}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <style>{`
        @keyframes ping {
          75%, 100% { transform: scale(2); opacity: 0; }
        }
      `}</style>
    </div>
  )
}



export function SettingsPage() {
  return (
    <div style={{ animation: 'fadeIn 0.4s ease' }}>
      <div style={{ marginBottom: 24 }}>
        <span className="label" style={{ marginBottom: 4 }}>Configuration</span>
        <h2 style={{ fontSize: 22, fontWeight: 700 }}>System Settings</h2>
      </div>

      <div className="card" style={{ padding: 60, textAlign: 'center' }}>
        <div style={{ fontSize: 48, marginBottom: 20 }}>⚙️</div>
        <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12 }}>Settings are currently locked</h3>
        <p className="mono" style={{ fontSize: 13, color: 'var(--muted)', maxWidth: 400, margin: '0 auto', lineHeight: 1.6 }}>
          The real-time parameter tuning module is under maintenance. <br />
          Hardware thresholds are currently managed via the backend configuration.
        </p>
      </div>
    </div>
  )
}

