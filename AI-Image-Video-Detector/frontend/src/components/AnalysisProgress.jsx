import React, { useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';

function AnalysisProgress({ mediaType, uploadProgress }) {
  const [stageIndex, setStageIndex] = useState(0);

  const stages = [
    'Preparing your media...',
    'Analyzing your media...',
    'Checking the results...',
    'Finalizing analysis...',
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setStageIndex((prev) => (prev + 1) % stages.length);
    }, 1800);
    return () => clearInterval(interval);
  }, [stages.length]);

  const isVideo = mediaType === 'video';
  const title = isVideo ? 'Analyzing Video' : 'Analyzing Image';
  const defaultSubtitle = isVideo
    ? 'Please wait while we analyze your video...'
    : 'Please wait while we analyze your image...';

  return (
    <div className="analysis-progress-panel">
      <div className="progress-header">
        <div className="progress-spinner-box">
          <RefreshCw size={26} className="animate-spin text-cyan" />
        </div>
        <div>
          <h4 className="progress-title">{title}</h4>
          <p className="progress-subtitle">{stages[stageIndex] || defaultSubtitle}</p>
        </div>
      </div>

      {/* Indeterminate Bar with Upload Progress */}
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill indeterminate"
          style={{ width: uploadProgress > 0 && uploadProgress < 100 ? `${uploadProgress}%` : '100%' }}
        ></div>
      </div>

      <div className="progress-meta">
        <span>
          {uploadProgress > 0 && uploadProgress < 100
            ? `Uploading media: ${uploadProgress}%`
            : 'Analysis in progress'}
        </span>
        <span>Please wait...</span>
      </div>
    </div>
  );
}

export default AnalysisProgress;
