import './Sidebar.css'

export default function Sidebar({ activePage, onNavigate, data = {} }) {
  const menuItems = [
    { id: 'monitor', label: 'Monitor', icon: '📊' },
    { id: 'signal', label: 'Signal View', icon: '📡' },
    { id: 'activity', label: 'Activity Log', icon: '📋' },
    { id: 'system', label: 'System Info', icon: '🖥️' },
    { id: 'settings', label: 'Settings', icon: '⚙️' },
  ]

  const isConnected = data.connected || false
  // Το Load υπολογίζεται από το πραγματικό throughput, με ταβάνι τα 110 FPS (το όριο του ESP32)
  const loadPct = isConnected ? Math.min(100, (data.fps / 110) * 100).toFixed(0) : 0
  // Το Buffer Health είναι 100% όταν το latency είναι 0. Όσο αυξάνεται το latency, η υγεία πέφτει.
  const bufferPct = isConnected ? Math.max(0, 100 - (data.latency || 0)).toFixed(0) : 0

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <div className="label" style={{ marginBottom: 24, fontSize: 10 }}>Navigation</div>
        <div className="menu-list">
          {menuItems.map(item => (
            <div
              key={item.id}
              className={`menu-item ${activePage === item.id ? 'active' : ''}`}
              onClick={() => onNavigate(item.id)}
            >
              <span className="menu-icon">{item.icon}</span>
              <span className="menu-label">{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="sidebar-bottom">
        <div className="system-card">
          <div className="label" style={{ marginBottom: 12 }}>System Health</div>
          <div className="health-stat">
            <span>Buffer <span style={{ fontSize: 9, opacity: 0.7, marginLeft: 4 }}>{bufferPct}%</span></span>
            <div className="health-bar"><div className="health-fill" style={{ width: `${bufferPct}%`, transition: 'width 0.5s ease', background: isConnected ? 'var(--success)' : 'var(--muted)' }} /></div>
          </div>
          <div className="health-stat">
            <span>Load <span style={{ fontSize: 9, opacity: 0.7, marginLeft: 4 }}>{loadPct}%</span></span>
            <div className="health-bar"><div className="health-fill" style={{ width: `${loadPct}%`, transition: 'width 0.5s ease', background: loadPct > 80 ? 'var(--warning)' : isConnected ? 'var(--success)' : 'var(--muted)' }} /></div>
          </div>
        </div>
      </div>
    </aside>
  )
}
