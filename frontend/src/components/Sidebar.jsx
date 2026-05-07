import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Sparkles, LayoutDashboard, TerminalSquare } from 'lucide-react';

const Sidebar = () => {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path ? 'active' : '';
  };

  return (
    <nav className="sidebar">
      <Link to="/" className="sidebar-logo">
        <Sparkles size={28} />
        <span style={{marginLeft: '10px', fontWeight: 'bold', color: 'white'}}>LexiVision</span>
      </Link>
      
      <div className="sidebar-nav">
        <Link to="/" className={`nav-link ${isActive('/')}`} title="Home">
          <Sparkles size={22} />
        </Link>
        <Link to="/dashboard" className={`nav-link ${isActive('/dashboard')}`} title="Research Dashboard">
          <LayoutDashboard size={22} />
        </Link>
        <Link to="/app" className={`nav-link ${isActive('/app')}`} title="Workspace">
          <TerminalSquare size={22} />
        </Link>
      </div>
    </nav>
  );
};

export default Sidebar;
