import './Sidebar.css'

export default function Sidebar({ wsStatus }) {
  const menuItems = [
    { id: 'dash', label: 'Monitor', icon: '📊', active: true },
    { id: 'logs', label: 'Raw Data', icon: '📝' },
    { id: 'calc', label: 'Calibration', icon: '⚙️' },
    { id: 'stat', label: 'Spectrogram', icon: '📈' },
    { id: 'docs', label: 'Architecture', icon: '📖' },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <div className="label" style={{ marginBottom: 24, fontSize: 10 }}>Navigation</div>
        <div className="menu-list">
          {menuItems.map(item => (
            <div key={item.id} className={`menu-item ${item.active ? 'active' : ''}`}>
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
            <span>Buffer</span>
            <div className="health-bar"><div className="health-fill" style={{ width: '65%' }} /></div>
          </div>
          <div className="health-stat">
            <span>Load</span>
            <div className="health-bar"><div className="health-fill" style={{ width: '12%' }} /></div>
          </div>
        </div>
      </div>
    </aside>
  )
}
