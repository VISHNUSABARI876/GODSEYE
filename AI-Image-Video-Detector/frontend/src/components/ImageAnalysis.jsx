import React from 'react';
import ProbabilityRing from './ProbabilityRing';
import ConfidenceBadge from './ConfidenceBadge';
import Disclaimer from './Disclaimer';
import { FileImage, ShieldCheck, ShieldAlert, AlertTriangle, Eye, Layers } from 'lucide-react';

function ImageAnalysis({ data, previewUrl, onReset }) {
  if (!data) return null;

  const {
    result = 'Real',
    label = 'Likely real',
    ai_probability = 0,
    real_probability = 0,
    confidence = 0,
    confidence_category = 'High Confidence',
    filename = 'image.jpg',
    heatmap_url = null,
  } = data;

  const isAI = result === 'AI-Generated' || result === 'AI';
  const isUncertain = result === 'UNCERTAIN' || result === 'Uncertain';

  let verdictTitle = 'Authentic Real Media';
  let verdictColor = 'var(--accent-emerald)';
  let VerdictIcon = ShieldCheck;

  if (isAI) {
    verdictTitle = 'AI-Generated Media';
    verdictColor = 'var(--accent-rose)';
    VerdictIcon = ShieldAlert;
  } else if (isUncertain) {
    verdictTitle = 'Inconclusive / Uncertain Result';
    verdictColor = 'var(--accent-amber)';
    VerdictIcon = AlertTriangle;
  }

  return (
    <div className="analysis-result-panel">
      {/* Header Banner */}
      <div className="result-header-card" style={{ borderColor: `${verdictColor}40` }}>
        <div className="result-header-left">
          <div className="verdict-icon-box" style={{ background: verdictColor }}>
            <VerdictIcon size={32} />
          </div>
          <div>
            <div className="verdict-label-top">Analysis Verdict</div>
            <h2 className="verdict-title-text" style={{ color: verdictColor }}>
              {verdictTitle}
            </h2>
            <div className="verdict-subtitle">{label} • File: {filename}</div>
          </div>
        </div>

        <div className="result-header-right">
          <ConfidenceBadge confidenceCategory={confidence_category} result={result} />
          <div className="confidence-score-box">
            <span className="c-score-label">Confidence Score</span>
            <span className="c-score-val">{confidence}%</span>
          </div>
        </div>
      </div>

      {/* Main Breakdown Section */}
      <div className="result-body-grid">
        {/* Left Column: Image Preview & Visual Analysis */}
        <div className="media-display-card">
          <div className="card-section-title">
            <FileImage size={18} className="text-cyan" />
            <span>Target Image & Visual Analysis</span>
          </div>

          <div className="image-comparison-box">
            <div className="img-frame">
              <span className="frame-label">Original Target Image</span>
              <img src={previewUrl} alt="Original" className="analysis-img" />
            </div>

            {heatmap_url && (
              <div className="img-frame">
                <span className="frame-label">Visual Analysis Map</span>
                <img src={heatmap_url} alt="Heatmap" className="analysis-img" />
              </div>
            )}
          </div>

          {heatmap_url && (
            <p className="attention-caption">
              <strong>Visual Analysis Map:</strong> The highlighted regions represent areas evaluated during analysis. They are not proof of AI generation.
            </p>
          )}
        </div>

        {/* Right Column: Probabilities & Rings */}
        <div className="probabilities-card">
          <div className="card-section-title">
            <Eye size={18} className="text-violet" />
            <span>Probabilistic Distribution</span>
          </div>

          <div className="ring-wrapper">
            <ProbabilityRing aiProb={ai_probability} realProb={real_probability} result={result} />
          </div>

          <div className="probability-bars-group">
            <div className="prob-bar-item">
              <div className="prob-bar-header">
                <span className="text-rose font-semibold">AI-Generated Probability</span>
                <span className="font-bold">{ai_probability}%</span>
              </div>
              <div className="prob-track">
                <div className="prob-fill bg-rose" style={{ width: `${ai_probability}%` }}></div>
              </div>
            </div>

            <div className="prob-bar-item">
              <div className="prob-bar-header">
                <span className="text-emerald font-semibold">Real / Authentic Probability</span>
                <span className="font-bold">{real_probability}%</span>
              </div>
              <div className="prob-track">
                <div className="prob-fill bg-emerald" style={{ width: `${real_probability}%` }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Disclaimer />

      <div className="result-actions-bar">
        <button className="btn btn-secondary" onClick={onReset}>
          Analyze Another File
        </button>
      </div>
    </div>
  );
}

export default ImageAnalysis;
