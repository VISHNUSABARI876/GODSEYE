import React, { useState } from 'react';
import { Clock, Play, AlertTriangle } from 'lucide-react';

function VideoTimeline({ timestamps = [], onSeekTimestamp }) {
  const [selectedTimestamp, setSelectedTimestamp] = useState(null);

  if (!timestamps || timestamps.length === 0) {
    return (
      <div className="video-timeline-panel empty">
        <Clock size={18} className="text-muted" />
        <span>No suspicious anomaly timestamps detected across video frames.</span>
      </div>
    );
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins.toString().padStart(2, '0')}:${secs.padStart(4, '0')}`;
  };

  const handleTimestampClick = (ts) => {
    setSelectedTimestamp(ts);
    if (onSeekTimestamp) {
      onSeekTimestamp(ts);
    }
  };

  return (
    <div className="video-timeline-panel">
      <div className="timeline-header">
        <div className="timeline-title">
          <Clock size={18} className="text-amber" />
          <span>Suspicious Frame Timeline</span>
        </div>
        <span className="timeline-count-tag">{timestamps.length} Anomaly Frame(s) Flagged</span>
      </div>

      <div className="timeline-markers-row">
        {timestamps.map((ts, index) => {
          const isSelected = selectedTimestamp === ts;
          return (
            <button
              key={index}
              className={`timeline-marker-btn ${isSelected ? 'active' : ''}`}
              onClick={() => handleTimestampClick(ts)}
              title={`Jump video player to ${formatTime(ts)}`}
            >
              <Play size={12} />
              <span>{formatTime(ts)}</span>
            </button>
          );
        })}
      </div>

      {selectedTimestamp !== null && (
        <div className="timestamp-detail-box">
          <AlertTriangle size={16} className="text-amber" />
          <div>
            <strong>Suspicious frame selected at {formatTime(selectedTimestamp)}</strong>
            <p className="timestamp-notice">
              Elevated anomaly detected at this timestamp. Jumped video player to frame. Timestamp markers represent statistical indicators, not definitive proof of synthetic manipulation.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default VideoTimeline;
