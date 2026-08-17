import React, { useRef, useState } from 'react';
import { UploadCloud, FileImage, FileVideo, X, Sparkles, AlertCircle } from 'lucide-react';

function UploadZone({ onFileSelect, selectedFile, previewUrl, mediaType, onReset, error }) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const imageExts = ['jpg', 'jpeg', 'png', 'webp'];
  const videoExts = ['mp4', 'mov', 'avi', 'webm'];

  const validateAndPass = (file) => {
    if (!file) return;

    const ext = file.name.split('.').pop().toLowerCase();
    if (!imageExts.includes(ext) && !videoExts.includes(ext)) {
      onFileSelect(null, `This file format (.${ext}) isn't supported. Please select a valid image (JPG, PNG, WEBP) or video (MP4, MOV, AVI, WEBM).`);
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      onFileSelect(null, `This file exceeds the maximum upload size of 100MB.`);
      return;
    }

    const type = videoExts.includes(ext) ? 'video' : 'image';
    onFileSelect(file, null, type);
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
      validateAndPass(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="upload-card-panel">
      <div className="upload-header">
        <div className="upload-title-group">
          <UploadCloud size={22} className="text-cyan" />
          <h3>Media Upload & Analysis Target</h3>
        </div>
        <div className="upload-meta-info">
          <span>Max size: 100MB</span>
          <span>•</span>
          <span>Formats: JPG, PNG, WEBP, MP4, MOV, AVI, WEBM</span>
        </div>
      </div>

      <div
        className={`dropzone-container ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp,.mp4,.mov,.avi,.webm"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files && validateAndPass(e.target.files[0])}
        />

        {!selectedFile ? (
          <div className="dropzone-prompt">
            <div className="upload-icon-circle">
              <UploadCloud size={32} />
            </div>
            <div className="prompt-text">
              <h4>Drop your media here</h4>
              <p>or click to browse files from your computer</p>
            </div>
            <button type="button" className="btn btn-secondary">
              Browse Files
            </button>
          </div>
        ) : (
          <div className="dropzone-preview-box">
            {mediaType === 'image' ? (
              <img src={previewUrl} alt="Selected preview" className="media-preview-img" />
            ) : (
              <video src={previewUrl} controls className="media-preview-video" />
            )}

            <div className="preview-details-bar">
              <div className="file-info-badge">
                {mediaType === 'image' ? <FileImage size={16} /> : <FileVideo size={16} />}
                <span className="file-name-text">{selectedFile.name}</span>
                <span className="file-size-tag">({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)</span>
              </div>

              <button type="button" className="btn btn-secondary btn-sm" onClick={(e) => { e.stopPropagation(); onReset(); }}>
                <X size={16} />
                <span>Remove</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="upload-error-alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}

export default UploadZone;
