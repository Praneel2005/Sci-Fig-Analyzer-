import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowRight } from 'lucide-react';

const HomePage = () => {
  return (
    <div className="home-container">
      <div className="hero-section">
        {/* Decorative Floating Images */}
        <img src="/img1.png" alt="Researcher" className="floating-image img-left" />
        <img src="/img2.png" alt="Abstract Data" className="floating-image img-right" />
        
        <div className="main-card hero-card">
          <div className="badge">
            <Sparkles size={16} /> Powered by Qwen2.5-VL-7B
          </div>
          <h1>Comprehend the Unseen in Scientific Literature</h1>
          <p className="subtitle">
            Upload any complex research PDF. Our Vision-Language model instantly extracts figures, analyzes the data, and generates publication-ready paragraph descriptions.
          </p>
          
          <div className="hero-buttons">
            <Link to="/app" className="launch-btn">
              Launch Workspace <ArrowRight size={20} />
            </Link>
            <Link to="/dashboard" className="secondary-btn">
              View Research Architecture
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
