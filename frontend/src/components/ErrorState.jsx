import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

function ErrorState({ message, onRetry }) {
  const isTechnicalError =
    !message ||
    /pytorch|model|exception|traceback|500|cuda|algorithm|embedding/i.test(message);

  const displayMessage = isTechnicalError
    ? 'Something went wrong while analyzing your media. Please try again.'
    : message;

  return (
    <div className="error-state-card">
      <AlertCircle size={28} className="text-rose" />
      <div>
        <h4 className="error-state-title">Analysis could not be completed.</h4>
        <p className="error-state-desc">{displayMessage}</p>
      </div>
      {onRetry && (
        <button className="btn btn-secondary btn-sm" onClick={onRetry}>
          <RefreshCw size={14} /> Try Again
        </button>
      )}
    </div>
  );
}

export default ErrorState;
