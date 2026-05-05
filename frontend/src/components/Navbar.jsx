import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Microscope, LayoutDashboard, TerminalSquare } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path ? 'active' : '';
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <Microscope className="logo-icon" />
          <span>Sci-Fig Analyzer</span>
        </Link>
        
        <div className="navbar-links">
          <Link to="/" className={`nav-link ${isActive('/')}`}>
            Home
          </Link>
          <Link to="/dashboard" className={`nav-link ${isActive('/dashboard')}`}>
            <LayoutDashboard size={18} />
            Research Dashboard
          </Link>
          <Link to="/app" className={`nav-link ${isActive('/app')} primary-nav-btn`}>
            <TerminalSquare size={18} />
            Workspace
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
