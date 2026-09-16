import React, { useRef } from 'react';
import ProbabilityRing from './ProbabilityRing';
import ConfidenceBadge from './ConfidenceBadge';
import VideoTimeline from './VideoTimeline';
import Disclaimer from './Disclaimer';
import { Film, ShieldCheck, ShieldAlert, AlertTriangle, Cpu, Layers, Activity, Eye } from 'lucide-react';

function VideoAnalysis({ data, previewUrl, onReset }) {
  const videoRef = useRef(null);

  if (!data) return null;

  const {
    result = 'Real',
    label = 'Likely real',
    ai_probability = 0,
    real_probability = 0,
    confidence = 0,
    confidence_category = 'High Confidence',
    spatial_probability = 0,
    temporal_probability = 0,
    frames_analyzed = 16,
    sequences_analyzed = 1,
    suspicious_timestamps = [],
    filename = 'video.mp4'
  } = data;

  const isAI = result === 'AI-Generated' || result === 'AI';
  const isUncertain = result === 'UNCERTAIN' || result === 'Uncertain';

  let verdictTitle = 'Authentic Real Video';
  let verdictColor = 'var(--accent-emerald)';
  let VerdictIcon = ShieldCheck;

  if (isAI) {
    verdictTitle = 'AI-Generated Video';
    verdictColor = 'var(--accent-rose)';
    VerdictIcon = ShieldAlert;
  } else if (isUncertain) {
    verdictTitle = 'Inconclusive / Uncertain Video Result';
    verdictColor = 'var(--accent-amber)';
    VerdictIcon = AlertTriangle;
  }

  const handleSeekTimestamp = (seconds) => {
    if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play().catch(() => {});
    }
  };

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

      {/* Main Grid: Video Player + Analysis Breakdown */}
      <div className="result-body-grid">
        {/* Left Column: Video Player & Timeline */}
        <div className="media-display-card">
          <div className="card-section-title">
            <Film size={18} className="text-cyan" />
            <span>Video Target & Frame Player</span>
          </div>

          <div className="video-player-box">
            <video ref={videoRef} src={previewUrl} controls className="analysis-video-player" />
          </div>

          <VideoTimeline timestamps={suspicious_timestamps} onSeekTimestamp={handleSeekTimestamp} />
        </div>

        {/* Right Column: Breakdown & Ring */}
        <div className="probabilities-card">
          <div className="card-section-title">
            <Layers size={18} className="text-violet" />
            <span>Analysis Breakdown</span>
          </div>

          <div className="ring-wrapper">
            <ProbabilityRing aiProb={ai_probability} realProb={real_probability} result={result} />
          </div>

          {/* Analysis Component Visual Comparison */}
          <div className="dual-branch-box">
            <h5 className="branch-title">Analysis Component Evaluation</h5>

            <div className="branch-metric-item">
              <div className="branch-metric-header">
                <span className="b-label">Frame Analysis</span>
                <span className="b-val">{spatial_probability}% AI</span>
              </div>
              <div className="branch-track">
                <div className="branch-fill bg-cyan" style={{ width: `${spatial_probability}%` }}></div>
              </div>
            </div>

            <div className="branch-metric-item">
              <div className="branch-metric-header">
                <span className="b-label">Sequence Continuity</span>
                <span className="b-val">{temporal_probability}% AI</span>
              </div>
              <div className="branch-track">
                <div className="branch-fill bg-violet" style={{ width: `${temporal_probability}%` }}></div>
              </div>
            </div>

            <div className="branch-metric-item">
              <div className="branch-metric-header">
                <span className="b-label font-bold text-primary">Combined Result</span>
                <span className="b-val font-bold text-primary">{ai_probability}% AI</span>
              </div>
              <div className="branch-track">
                <div className="branch-fill bg-rose" style={{ width: `${ai_probability}%` }}></div>
              </div>
            </div>
          </div>

          {/* Sequence Statistics */}
          <div className="video-stats-grid">
            <div className="v-stat-card">
              <span className="v-stat-num">{frames_analyzed}</span>
              <span className="v-stat-lbl">Frames Analyzed</span>
            </div>
            <div className="v-stat-card">
              <span className="v-stat-num">{sequences_analyzed}</span>
              <span className="v-stat-lbl">Sequences Processed</span>
            </div>
            <div className="v-stat-card">
              <span className="v-stat-num">{suspicious_timestamps.length}</span>
              <span className="v-stat-lbl">Suspicious Timestamps</span>
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

export default VideoAnalysis;
