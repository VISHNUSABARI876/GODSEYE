import axios from 'axios';

const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000/api';
const API_BASE_URL = rawApiBaseUrl.replace(/\/+$/, '').endsWith('/api')
  ? rawApiBaseUrl.replace(/\/+$/, '')
  : `${rawApiBaseUrl.replace(/\/+$/, '')}/api`;

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 120 seconds timeout for video frames processing
});

/**
 * Health Check API Call
 * GET /api/health
 */
export const checkHealth = async () => {
  const startTime = performance.now();
  try {
    const response = await apiClient.get('/health');
    const endTime = performance.now();
    const latency = Math.round(endTime - startTime);

    return {
      success: response.data?.success ?? true,
      data: response.data,
      latency,
    };
  } catch (error) {
    const endTime = performance.now();
    const latency = Math.round(endTime - startTime);

    return {
      success: false,
      error: error.response?.data?.message || 'Unable to connect to the analysis server.',
      latency,
    };
  }
};

/**
 * Database Health Check API Call
 * GET /api/health/database
 */
export const checkDatabaseHealth = async () => {
  const startTime = performance.now();
  try {
    const response = await apiClient.get('/health/database');
    const endTime = performance.now();
    const latency = Math.round(endTime - startTime);

    return {
      success: response.data?.success ?? false,
      data: response.data,
      latency,
    };
  } catch (error) {
    const endTime = performance.now();
    const latency = Math.round(endTime - startTime);

    return {
      success: false,
      error: error.response?.data?.connection || 'PostgreSQL database is offline',
      latency,
    };
  }
};

/**
 * Upload Image or Video File for AI Detection
 * POST /api/detection/analyze
 */
export const analyzeFile = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await apiClient.post('/detection/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onUploadProgress) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onUploadProgress(percent);
        }
      },
    });

    return {
      success: true,
      data: response.data,
    };
  } catch (error) {
    let errorMsg = 'Analysis could not be completed.';
    if (error.response) {
      if (error.response.status === 413) {
        errorMsg = 'This file exceeds the maximum upload size.';
      } else if (error.response.status === 503) {
        errorMsg = 'The analysis service is currently unavailable.';
      } else {
        const serverMsg = error.response.data?.message;
        const isTechnical = serverMsg && /pytorch|model|exception|traceback|500|cuda|algorithm|embedding/i.test(serverMsg);
        errorMsg = (serverMsg && !isTechnical) ? serverMsg : 'Something went wrong while analyzing your media. Please try again.';
      }
    } else if (error.code === 'ECONNABORTED') {
      errorMsg = 'Analysis timed out. Please try again.';
    } else if (!window.navigator.onLine) {
      errorMsg = 'No internet connection detected.';
    } else {
      errorMsg = 'Unable to connect to the analysis server.';
    }

    return {
      success: false,
      error: errorMsg,
    };
  }
};

/**
 * Fetch History API Call
 * GET /api/history
 */
export const fetchHistory = async () => {
  try {
    const response = await apiClient.get('/history');
    return {
      success: true,
      data: response.data?.history || [],
    };
  } catch (error) {
    return {
      success: false,
      data: [],
      error: 'Could not load analysis history.',
    };
  }
};

export default apiClient;
