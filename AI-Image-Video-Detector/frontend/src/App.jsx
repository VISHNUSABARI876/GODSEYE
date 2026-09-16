import React, { useState, useEffect, useRef } from 'react';
import { checkHealth, checkDatabaseHealth, analyzeFile, fetchHistory } from './services/api';
import {
  Navbar,
  HeroSection,
  UploadZone,
  AnalysisProgress,
  ResultCard,
  HistoryTable,
  ErrorState,
} from './components';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard', 'history'

  // Health state
  const [healthStatus, setHealthStatus] = useState({
    loading: true,
    connected: false,
    data: null,
  });

  // Media upload & analysis state
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mediaType, setMediaType] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);

  // History state
  const [historyItems, setHistoryItems] = useState([]);
  const uploadRef = useRef(null);

  // System health check
  const refreshSystemHealth = async () => {
    setHealthStatus((prev) => ({ ...prev, loading: true }));
    const healthRes = await checkHealth();
    const dbRes = await checkDatabaseHealth();

    setHealthStatus({
      loading: false,
      connected: healthRes.success && dbRes.success,
      data: {
        backend: healthRes.data?.status || (healthRes.success ? 'online' : 'offline'),
        database: dbRes.data?.connection || (dbRes.success ? 'connected' : 'disconnected'),
      },
    });
  };

  // Load history from backend / local state
  const loadHistoryRecords = async () => {
    const res = await fetchHistory();
    if (res.success && res.data && res.data.length > 0) {
      setHistoryItems(res.data);
    }
  };

  useEffect(() => {
    refreshSystemHealth();
    loadHistoryRecords();
  }, []);

  // Handle file selection from UploadZone
  const handleFileSelect = (file, error, type) => {
    if (error) {
      setUploadError(error);
      setSelectedFile(null);
      setPreviewUrl(null);
      setMediaType(null);
      return;
    }

    setUploadError(null);
    setAnalysisError(null);
    setAnalysisResult(null);

    if (file) {
      setSelectedFile(file);
      setMediaType(type);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleResetUpload = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setMediaType(null);
    setAnalysisResult(null);
    setUploadError(null);
    setAnalysisError(null);
    setUploadProgress(0);
  };

  // Trigger analysis execution
  const handleRunAnalysis = async () => {
    if (!selectedFile || analyzing) return;

    setAnalyzing(true);
    setUploadProgress(0);
    setAnalysisError(null);

    const response = await analyzeFile(selectedFile, (progress) => {
      setUploadProgress(progress);
    });

    setAnalyzing(false);

    if (response.success) {
      const record = {
        ...response.data,
        id: response.data.detection_id || Date.now(),
        filename: response.data.filename || selectedFile.name,
        media_type: mediaType,
        created_at: new Date().toLocaleTimeString(),
      };
      setAnalysisResult(record);

      // Add to local history list
      setHistoryItems((prev) => [record, ...prev]);
    } else {
      setAnalysisError(response.error);
    }
  };

  const handleDeleteHistoryItem = (id) => {
    setHistoryItems((prev) => prev.filter((item) => (item.id || item.detection_id) !== id));
  };

  const handleViewHistoryItem = (item) => {
    setAnalysisResult(item);
    setMediaType(item.media_type || item.type || 'image');
    setActiveTab('dashboard');
    window.scrollTo({ top: 300, behavior: 'smooth' });
  };

  const scrollToUpload = () => {
    setActiveTab('dashboard');
    if (uploadRef.current) {
      uploadRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="app-wrapper">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} healthStatus={healthStatus} />

      <main>
        {activeTab === 'dashboard' && (
          <>
            <HeroSection
              onAnalyzeClick={scrollToUpload}
              onHistoryClick={() => setActiveTab('history')}
            />

            <div ref={uploadRef}>
              {!analysisResult && (
                <>
                  <UploadZone
                    onFileSelect={handleFileSelect}
                    selectedFile={selectedFile}
                    previewUrl={previewUrl}
                    mediaType={mediaType}
                    onReset={handleResetUpload}
                    error={uploadError}
                  />

                  {selectedFile && !analyzing && (
                    <div style={{ marginBottom: '2rem' }}>
                      <button
                        id="btn-run-detection"
                        className="btn btn-primary"
                        style={{ width: '100%', justifyContent: 'center', padding: '0.9rem', fontSize: '1.05rem' }}
                        onClick={handleRunAnalysis}
                      >
                        Start AI Detection Scan
                      </button>
                    </div>
                  )}

                  {analyzing && <AnalysisProgress mediaType={mediaType} uploadProgress={uploadProgress} />}
                </>
              )}

              {analysisError && <ErrorState message={analysisError} onRetry={handleRunAnalysis} />}

              {analysisResult && (
                <ResultCard
                  data={analysisResult}
                  previewUrl={previewUrl}
                  mediaType={mediaType}
                  onReset={handleResetUpload}
                />
              )}
            </div>
          </>
        )}

        {activeTab === 'history' && (
          <HistoryTable
            historyItems={historyItems}
            onDeleteItem={handleDeleteHistoryItem}
            onViewItem={handleViewHistoryItem}
          />
        )}
      </main>
    </div>
  );
}

export default App;
