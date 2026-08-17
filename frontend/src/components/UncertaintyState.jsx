import React from 'react';
import ProbabilityRing from './ProbabilityRing';
import ConfidenceBadge from './ConfidenceBadge';
import Disclaimer from './Disclaimer';
import { AlertTriangle, HelpCircle } from 'lucide-react';

function UncertaintyState({ data, previewUrl, mediaType, onReset }) {
  const {
    ai_probability = 50,
    real_probability = 50,
    filename = 'media_file',
  } = data;

  return (
    <div className="analysis-result-panel uncertainty-panel">
      <div className="result-header-card uncertainty-header">
        <div className="result-header-left">
          <div className="verdict-icon-box bg-amber">
            <AlertTriangle size={32} />
          </div>
          <div>
            <div className="verdict-label-top">Analysis Status</div>
            <h2 className="verdict-title-text text-amber">
              UNCERTAIN / INCONCLUSIVE
            </h2>
            <div className="verdict-subtitle">File: {filename}</div>
          </div>
        </div>

        <div className="result-header-right">
          <ConfidenceBadge confidenceCategory="Low Confidence" result="UNCERTAIN" />
        </div>
      </div>

      <div className="uncertainty-body-box">
        <div className="uncertainty-notice-card">
          <HelpCircle size={28} className="text-amber" />
          <div>
            <h4 className="notice-title">Unable to confidently classify this media.</h4>
            <p className="notice-text">
              The analysis detected conflicting indicators lying within the calibrated uncertainty boundary.
              <strong> Consider this result inconclusive.</strong>
            </p>
          </div>
        </div>

        <div className="uncertainty-ring-grid">
          <ProbabilityRing aiProb={ai_probability} realProb={real_probability} result="UNCERTAIN" />

          <div className="probability-bars-group">
            <div className="prob-bar-item">
              <div className="prob-bar-header">
                <span className="text-rose font-semibold">AI Probability Indicator</span>
                <span className="font-bold">{ai_probability}%</span>
              </div>
              <div className="prob-track">
                <div className="prob-fill bg-rose" style={{ width: `${ai_probability}%` }}></div>
              </div>
            </div>

            <div className="prob-bar-item">
              <div className="prob-bar-header">
                <span className="text-emerald font-semibold">Real Probability Indicator</span>
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

export default UncertaintyState;
