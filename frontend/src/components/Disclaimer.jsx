import React from 'react';
import { Info } from 'lucide-react';

function Disclaimer() {
  return (
    <div className="disclaimer-panel">
      <Info size={18} className="disclaimer-icon" />
      <p className="disclaimer-text">
        <strong>Forensic Notice:</strong> Detection is probabilistic and may produce false positives or false negatives.
        Results are generated using automated media evaluations and should be used as supporting evidence rather than definitive proof of media authenticity.
      </p>
    </div>
  );
}

export default Disclaimer;
