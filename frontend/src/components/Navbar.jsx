import React, { useState } from 'react';
import { ShieldCheck, LayoutDashboard, History, Menu, X, Activity } from 'lucide-react';

function Navbar({ activeTab, setActiveTab, healthStatus }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'history', label: 'History', icon: History },
  ];

  const handleNavClick = (id) => {
    setActiveTab(id);
    setMobileMenuOpen(false);
  };

  return (
    <nav className="navbar-container">
      <div className="navbar-inner">
        {/* Brand Logo */}
        <div className="navbar-brand" onClick={() => handleNavClick('dashboard')} style={{ cursor: 'pointer' }}>
          <div className="brand-icon">
            <ShieldCheck size={26} />
          </div>
          <div>
            <div className="brand-title">GODSEYE</div>
            <div className="brand-subtitle">Media Authenticator & Detector</div>
          </div>
        </div>

        {/* Desktop Navigation */}
        <div className="nav-menu-desktop">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => handleNavClick(item.id)}
              >
                <Icon size={17} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>

        {/* Right Status Badge */}
        <div className="nav-right">
          <div className={`status-pill ${healthStatus?.connected ? 'online' : 'offline'}`}>
            <Activity size={14} className={healthStatus?.connected ? 'pulse' : ''} />
            <span>{healthStatus?.connected ? 'Engine Online' : 'Offline'}</span>
          </div>

          {/* Mobile Hamburger Toggle */}
          <button
            className="mobile-menu-btn"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle Navigation Menu"
          >
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="nav-menu-mobile">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item-mobile ${isActive ? 'active' : ''}`}
                onClick={() => handleNavClick(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      )}
    </nav>
  );
}

export default Navbar;
