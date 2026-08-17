import React from 'react';
import { Sparkles, History, Cpu } from 'lucide-react';

function HeroSection({ onAnalyzeClick, onHistoryClick }) {
  return (
    <section className="hero-container">
      <div className="hero-content">
        <div className="hero-badge">
          <Cpu size={14} />
          <span>AI Media Authenticity & Detection</span>
        </div>

        <h1 className="hero-title">
          Detect AI-Generated Media
        </h1>

        <p className="hero-subtitle">
          Analyze images and videos for patterns associated with synthetic media generation.
        </p>

        <div className="hero-actions">
          <button className="btn btn-primary" onClick={onAnalyzeClick}>
            <Sparkles size={18} />
            <span>Analyze Media</span>
          </button>

          <button className="btn btn-secondary" onClick={onHistoryClick}>
            <History size={18} />
            <span>View History</span>
          </button>
        </div>
      </div>
    </section>
  );
}

export default HeroSection;
