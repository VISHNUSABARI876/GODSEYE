import React from 'react';
import ImageAnalysis from './ImageAnalysis';
import VideoAnalysis from './VideoAnalysis';
import UncertaintyState from './UncertaintyState';

function ResultCard({ data, previewUrl, mediaType, onReset }) {
  if (!data) return null;

  if (data.result === 'UNCERTAIN' || data.result === 'Uncertain') {
    return <UncertaintyState data={data} previewUrl={previewUrl} mediaType={mediaType} onReset={onReset} />;
  }

  if (mediaType === 'video' || data.type === 'video') {
    return <VideoAnalysis data={data} previewUrl={previewUrl} onReset={onReset} />;
  }

  return <ImageAnalysis data={data} previewUrl={previewUrl} onReset={onReset} />;
}

export default ResultCard;
