import React from 'react';
import { ShieldCheck } from 'lucide-react';

function Footer({ setActiveTab }) {
  return (
    <footer className="footer-container">
      <div className="footer-inner">
        <div className="footer-brand">
          <div className="brand-logo">
            <div className="brand-icon small">
              <ShieldCheck size={18} />
            </div>
            <span className="brand-title-sm">GODSEYE</span>
          </div>
          <p className="footer-tagline">
            AI-generated image & video detection platform.
          </p>
        </div>

        <div className="footer-links">
          <button onClick={() => setActiveTab('dashboard')}>Dashboard</button>
          <button onClick={() => setActiveTab('history')}>History</button>
        </div>

        <div className="footer-copy">
          <span>&copy; {new Date().getFullYear()} AI-Image-Video-Detector. Academic & Forensic Research Project.</span>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
