import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowRight } from 'lucide-react';
import m1 from "../../public/m1.png";

const HomePage = () => {
  return (
    <div className="home-container">
      <div className="hero-section">
        
        <h1 className="hero-title">
          <span className="title-primary">LexiVision</span>
          <span className="title-secondary">Grounded Scientific Figure Understanding</span>
        </h1>
        
        <p className="hero-subtitle">
          Generate professional scientific figures, research diagrams, and academic illustrations with AI. 
          Perfect for papers, posters, and presentations. No design skills needed.
        </p>
        
        <div className="hero-buttons">
          <Link to="/app" className="launch-btn">
            <Sparkles size={18} /> Generate Illustration Now
          </Link>
          <Link to="/dashboard" className="secondary-btn">
            See Examples <ArrowRight size={18} />
          </Link>
        </div>
        
        <img src={m1} alt="Architecture Flow" className="hero-diagram" />
      </div>
    </div>
  );
};

export default HomePage;
