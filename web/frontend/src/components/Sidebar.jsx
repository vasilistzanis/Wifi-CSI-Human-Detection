import './Sidebar.css'

export default function Sidebar({ activePage, onNavigate }) {
  const menuItems = [
    { id: 'monitor',  label: 'Monitor',      icon: '📊' },
    { id: 'signal',   label: 'Signal View',   icon: '📡' },
    { id: 'activity', label: 'Activity Log',  icon: '📋' },
    { id: 'system',   label: 'System Info',   icon: '🖥️' },
    { id: 'settings', label: 'Settings',      icon: '⚙️' },
  ]

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
