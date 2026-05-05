import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import AppPage from './pages/AppPage';
import './index.css';

function App() {
  return (
    <Router>
      <div className="main-layout">
        <Navbar />
        <div className="page-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/app" element={<AppPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
