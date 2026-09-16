import React, { useState, useRef } from 'react';
import { analyzeFile } from '../services/api';
import { UploadCloud, FileImage, FileVideo, CheckCircle2, AlertTriangle, RefreshCw, Sparkles, Film, Database, Info, ShieldAlert, Cpu } from 'lucide-react';

function UploadDetector() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mediaType, setMediaType] = useState(null); // 'image' or 'video'
  const [dragActive, setDragActive] = useState(false);

  const [analyzing, setAnalyzing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [resultData, setResultData] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const fileInputRef = useRef(null);

  const handleFileSelect = (file) => {
    if (!file) return;

    const fileExt = file.name.split('.').pop().toLowerCase();
    const imageExts = ['png', 'jpg', 'jpeg', 'webp', 'bmp'];
    const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'webm'];

    if (!imageExts.includes(fileExt) && !videoExts.includes(fileExt)) {
      setErrorMessage(`Unsupported file format .${fileExt}. Please select a valid image or video file.`);
      return;
    }

    setErrorMessage(null);
    setResultData(null);
    setSelectedFile(file);
    const isVid = videoExts.includes(fileExt);
    setMediaType(isVid ? 'video' : 'image');

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyzeClick = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setUploadProgress(0);
    setErrorMessage(null);

    const response = await analyzeFile(selectedFile, (progress) => {
      setUploadProgress(progress);
    });

    setAnalyzing(false);

    if (response.success) {
      setResultData(response.data);
    } else {
      setErrorMessage(response.error);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setMediaType(null);
    setResultData(null);
    setErrorMessage(null);
    setUploadProgress(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* File Upload & Dropzone Area */}
      {!resultData && (
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Sparkles size={22} color="var(--accent-cyan)" /> AI Media Authenticator
          </h2>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
            Upload any image or video clip to detect whether it is synthetic/AI-generated or authentic real media.
          </p>

          {/* Dropzone Container */}
          <div
            id="dropzone-media-upload"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragActive ? 'var(--accent-cyan)' : 'rgba(255, 255, 255, 0.15)'}`,
              borderRadius: 'var(--radius-md)',
              padding: '2.5rem 1.5rem',
              textAlign: 'center',
              cursor: 'pointer',
              background: dragActive ? 'rgba(6, 182, 212, 0.08)' : 'rgba(0, 0, 0, 0.25)',
              transition: 'all 0.25s ease',
              position: 'relative'
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              style={{ display: 'none' }}
              onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
            />

            {!selectedFile ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.85rem' }}>
                <div style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '16px',
                  background: 'rgba(6, 182, 212, 0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--accent-cyan)',
                  border: '1px solid rgba(6, 182, 212, 0.25)'
                }}>
                  <UploadCloud size={32} />
                </div>
                <div>
                  <h4 style={{ fontSize: '1.05rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                    Drag & Drop your Image or Video here
                  </h4>
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    Supports PNG, JPG, JPEG, WEBP, MP4, AVI, MOV, MKV, WEBM (Max 100MB)
                  </p>
                </div>
                <button type="button" className="btn btn-secondary" style={{ marginTop: '0.5rem' }}>
                  Browse Local Files
                </button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                {mediaType === 'image' ? (
                  <img
                    src={previewUrl}
                    alt="Preview"
                    style={{ maxHeight: '220px', maxWidth: '100%', borderRadius: '12px', objectFit: 'contain', border: '1px solid var(--border-color)' }}
                  />
                ) : (
                  <video
                    src={previewUrl}
                    controls
                    style={{ maxHeight: '240px', maxWidth: '100%', borderRadius: '12px', border: '1px solid var(--border-color)' }}
                  />
                )}

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div className="brand-tag">
                    {mediaType === 'image' ? <FileImage size={14} /> : <FileVideo size={14} />}
                    {selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
                  </div>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{ padding: '0.3rem 0.75rem', fontSize: '0.8rem' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReset();
                    }}
                  >
                    Change File
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Error Banner */}
          {errorMessage && (
            <div style={{
              marginTop: '1.25rem',
              padding: '0.85rem 1rem',
              borderRadius: 'var(--radius-sm)',
              background: 'rgba(244, 63, 94, 0.12)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              color: 'var(--accent-rose)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
              fontSize: '0.88rem'
            }}>
              <AlertTriangle size={18} />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Action Button & Progress */}
          {selectedFile && !resultData && (
            <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <button
                id="btn-run-detection"
                className="btn btn-primary"
                style={{ width: '100%', justifyContent: 'center', padding: '0.85rem', fontSize: '1rem' }}
                onClick={handleAnalyzeClick}
                disabled={analyzing}
              >
                {analyzing ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" />
                    <span>Analyzing Media... ({uploadProgress}%)</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    <span>Start Media Analysis</span>
                  </>
                )}
              </button>

              {analyzing && (
                <div style={{ width: '100%', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '999px', height: '8px', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${uploadProgress}%`,
                      background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-violet))',
                      transition: 'width 0.3s ease'
                    }}
                  ></div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Detection Results Card */}
      {resultData && (
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h2 style={{ fontSize: '1.3rem', fontWeight: 800, marginBottom: '0.2rem' }}>
                Detection Analysis Results
              </h2>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Target File: <strong>{resultData.filename}</strong> ({resultData.media_type.toUpperCase()})
              </p>
            </div>
            <button className="btn btn-secondary" onClick={handleReset}>
              <RefreshCw size={16} /> Analyze Another File
            </button>
          </div>

          {/* Verdict Banner */}
          <div style={{
            padding: '1.75rem',
            borderRadius: 'var(--radius-md)',
            background: resultData.result === 'AI-Generated'
              ? 'linear-gradient(135deg, rgba(244, 63, 94, 0.18), rgba(15, 23, 42, 0.9))'
              : 'linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(15, 23, 42, 0.9))',
            border: `1px solid ${resultData.result === 'AI-Generated' ? 'rgba(244, 63, 94, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
            marginBottom: '1.75rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '1.25rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '16px',
                background: resultData.result === 'AI-Generated' ? 'var(--accent-rose)' : 'var(--accent-emerald)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                boxShadow: resultData.result === 'AI-Generated' ? '0 0 25px rgba(244, 63, 94, 0.5)' : '0 0 25px rgba(16, 185, 129, 0.5)'
              }}>
                {resultData.result === 'AI-Generated' ? <ShieldAlert size={32} /> : <CheckCircle2 size={32} />}
              </div>
              <div>
                <div style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: '0.15rem' }}>
                  Analysis Verdict
                </div>
                <div style={{
                  fontSize: '1.75rem',
                  fontWeight: 800,
                  color: resultData.result === 'AI-Generated' ? 'var(--accent-rose)' : 'var(--accent-emerald)',
                  letterSpacing: '-0.02em'
                }}>
                  {resultData.result === 'AI-Generated' ? 'AI-Generated Media' : 'Authentic Real Media'}
                </div>
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', uppercase: 'true' }}>
                Classification Confidence
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {resultData.confidence}%
              </div>
            </div>
          </div>

          {/* Dual Bar Probability Breakdown */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginBottom: '1.75rem' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                <span style={{ color: 'var(--accent-rose)' }}>AI Probability</span>
                <span>{resultData.ai_probability}%</span>
              </div>
              <div style={{ height: '12px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '999px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${resultData.ai_probability}%`, background: 'linear-gradient(90deg, #f43f5e, #e11d48)', transition: 'width 0.6s ease' }}></div>
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                <span style={{ color: 'var(--accent-emerald)' }}>Real / Authentic Probability</span>
                <span>{resultData.real_probability}%</span>
              </div>
              <div style={{ height: '12px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '999px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${resultData.real_probability}%`, background: 'linear-gradient(90deg, #10b981, #059669)', transition: 'width 0.6s ease' }}></div>
              </div>
            </div>
          </div>

          {/* Video Metrics & DB Record Confirmation Grid */}
          <div className="grid-3" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' }}>
            <div className="stat-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
                <Film size={16} color="var(--accent-cyan)" />
                <span className="stat-label">Frames Analyzed</span>
              </div>
              <div className="stat-value">{resultData.frames_analyzed} frame(s)</div>
            </div>

            <div className="stat-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
                <AlertTriangle size={16} color="var(--accent-rose)" />
                <span className="stat-label">Suspicious AI Frames</span>
              </div>
              <div className="stat-value">{resultData.suspicious_frames} frame(s)</div>
            </div>

            <div className="stat-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
                <Database size={16} color="var(--accent-violet)" />
                <span className="stat-label">Record ID</span>
              </div>
              <div className="stat-value">Record #{resultData.detection_id}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadDetector;
