import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react';

function ConfidenceBadge({ confidenceCategory = 'High Confidence', result = 'Real' }) {
  let badgeStyle = 'badge-emerald';
  let Icon = ShieldCheck;

  if (result === 'AI-Generated' || result === 'AI') {
    badgeStyle = 'badge-rose';
    Icon = ShieldAlert;
  } else if (result === 'UNCERTAIN' || result === 'Uncertain') {
    badgeStyle = 'badge-amber';
    Icon = AlertTriangle;
  }

  return (
    <div className={`confidence-badge-pill ${badgeStyle}`}>
      <Icon size={15} />
      <span>{confidenceCategory}</span>
    </div>
  );
}

export default ConfidenceBadge;
