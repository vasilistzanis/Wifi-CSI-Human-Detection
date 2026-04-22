import './Nav.css'

export default function Nav({ wsStatus, data }) {
  const statusColor = wsStatus === 'live' ? 'var(--success)'
    : wsStatus === 'connecting' ? 'var(--warning)'
      : '#ef4444'

  const statusText = wsStatus === 'live'
    ? `System Live · ${data.frame_count.toLocaleString()} fr`
    : wsStatus === 'connecting' ? 'Connecting'
      : 'Offline'

  return (
    <nav className="nav">
      <div className="nav-logo">
        <div className="nav-icon">
          <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
            <path d="M8 0C3.58 0 0 3.58 0 8s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm0 14.5c-3.58 0-6.5-2.92-6.5-6.5S4.42 1.5 8 1.5 14.5 4.42 14.5 8 11.58 14.5 8 14.5z" fill="white" opacity="0.3" />
            <circle cx="8" cy="8" r="3" fill="white" />
            <path d="M8 5V3M8 13v-2M5 8H3m10 0h-2" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
        <div>
          <div className="nav-title">WIFI CSI Analyzer</div>
          <div className="nav-sub mono">Live Monitoring</div>
        </div>
      </div>

      <div className="nav-status" style={{ borderColor: `${statusColor}33`, color: statusColor }}>
        <span className="nav-dot" style={{ background: statusColor, color: statusColor }} />
        <span className="mono">{statusText}</span>
      </div>
    </nav>
  )
}

