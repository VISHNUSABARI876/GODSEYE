import React from 'react';

function ProbabilityRing({ aiProb = 0, realProb = 0, size = 160, result = 'Real' }) {
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  // AI probability fraction
  const aiStrokeDashoffset = circumference - (aiProb / 100) * circumference;

  let ringColor = 'var(--accent-emerald)';
  if (result === 'AI-Generated' || result === 'AI') {
    ringColor = 'var(--accent-rose)';
  } else if (result === 'UNCERTAIN' || result === 'Uncertain') {
    ringColor = 'var(--accent-amber)';
  }

  return (
    <div className="probability-ring-container" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background Circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth={strokeWidth}
          fill="transparent"
        />

        {/* AI Probability Arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={ringColor}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={aiStrokeDashoffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1)' }}
        />
      </svg>

      <div className="ring-content">
        <span className="ring-val">{aiProb}%</span>
        <span className="ring-lbl">AI Probability</span>
      </div>
    </div>
  );
}

export default ProbabilityRing;
