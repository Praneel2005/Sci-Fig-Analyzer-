import React from 'react';
import { BarChart3, Image as ImageIcon, Zap, Cpu } from 'lucide-react';


const DashboardPage = () => {
  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>Research Architecture & Findings</h1>
        <p>A deep dive into the LexiVision end-to-end scientific figure comprehension pipeline.</p>
      </div>

      {/* Section 1: Methodology */}
      <section className="dashboard-section">
        <div className="section-title">
          <Cpu className="section-icon" />
          <h2>System Architecture</h2>
        </div>
        <div className="architecture-card">
          <div className="architecture-images-wrapper">
            <img src="/diagram1.png" alt="Methodology Part 1" className="architecture-image" />
            <img src="/diagram2.png" alt="System Methodology" className="architecture-image" />
          </div>
          <div className="architecture-text">
            <h3>Two-Phase Pipeline</h3>
            <p>Our methodology seamlessly bridges the gap between raw PDF extraction and advanced Vision-Language reasoning.</p>
            <ul>
              <li><strong>Phase 1:</strong> OCR-Injected BLIP for highly accurate, short-form literal captions.</li>
              <li><strong>Phase 2:</strong> Zero-Shot Qwen2.5-VL for deep, analytical paragraph generation without fine-tuning hallucination.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Section 2: Quantitative Results */}
      <section className="dashboard-section">
        <div className="section-title">
          <BarChart3 className="section-icon" />
          <h2>Quantitative Superiority</h2>
        </div>
        <p className="section-desc">Evaluation on the full 13,355-sample FigCaps test set.</p>

        <div className="metrics-grid">
          <div className="metric-card highlight">
            <div className="metric-value">+166.4%</div>
            <div className="metric-label">BLEU-4 Improvement</div>
            <div className="metric-context">vs. Original Baseline (Phase 1)</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">37.82%</div>
            <div className="metric-label">ROUGE-L Recall</div>
            <div className="metric-context">Zero-Shot Paragraph Generation</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">0.1342</div>
            <div className="metric-label">METEOR Score</div>
            <div className="metric-context">High Semantic Alignment</div>
          </div>
        </div>
      </section>

      {/* Section 3: Visual Showcase */}
      <section className="dashboard-section">
        <div className="section-title">
          <ImageIcon className="section-icon" />
          <h2>Qualitative Analysis</h2>
        </div>
        <div className="qualitative-card">
          <div className="qualitative-content">
            <div className="q-label">Generated Analytical Paragraph (Qwen2.5-VL)</div>
            <p>
              "The presented graph is an outage probability versus Signal to Information Ratio (SINR) plot, typically used in communication systems analysis. On the x-axis, it represents SINR in dB values ranging from -40 to +40. The y-axis shows outage probability on a logarithmic scale. Three curves are plotted for different standard deviations, demonstrating that as SINR increases, the outage probability significantly decreases..."
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default DashboardPage;
