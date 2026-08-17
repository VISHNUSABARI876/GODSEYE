import React, { useState } from 'react';
import { Search, Filter, Trash2, Eye, FileImage, FileVideo, History as HistoryIcon, ShieldAlert, ShieldCheck, AlertTriangle } from 'lucide-react';

function HistoryTable({ historyItems = [], onDeleteItem, onViewItem }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('ALL');

  const filteredItems = historyItems.filter((item) => {
    // Filter by type
    if (filterType === 'IMAGES' && item.media_type !== 'image' && item.type !== 'image') return false;
    if (filterType === 'VIDEOS' && item.media_type !== 'video' && item.type !== 'video') return false;
    if (filterType === 'AI' && item.result !== 'AI-Generated' && item.result !== 'AI') return false;
    if (filterType === 'REAL' && item.result !== 'Real' && item.result !== 'REAL') return false;
    if (filterType === 'UNCERTAIN' && item.result !== 'UNCERTAIN' && item.result !== 'Uncertain') return false;

    // Search by filename
    if (searchTerm.trim()) {
      return (item.filename || '').toLowerCase().includes(searchTerm.toLowerCase());
    }

    return true;
  });

  const getResultBadge = (result) => {
    if (result === 'AI-Generated' || result === 'AI') {
      return (
        <span className="badge-pill bg-rose-light text-rose">
          <ShieldAlert size={13} /> AI-Generated
        </span>
      );
    }
    if (result === 'UNCERTAIN' || result === 'Uncertain') {
      return (
        <span className="badge-pill bg-amber-light text-amber">
          <AlertTriangle size={13} /> Uncertain
        </span>
      );
    }
    return (
      <span className="badge-pill bg-emerald-light text-emerald">
        <ShieldCheck size={13} /> Likely Real
      </span>
    );
  };

  return (
    <div className="history-panel">
      {/* Toolbar: Search & Filters */}
      <div className="history-toolbar">
        <div className="search-input-box">
          <Search size={18} className="search-icon" />
          <input
            type="text"
            placeholder="Search analyzed media filename..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
        </div>

        <div className="filter-pills-row">
          {['ALL', 'IMAGES', 'VIDEOS', 'AI', 'REAL', 'UNCERTAIN'].map((f) => (
            <button
              key={f}
              className={`filter-pill-btn ${filterType === f ? 'active' : ''}`}
              onClick={() => setFilterType(f)}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Main Table / Mobile Cards */}
      {filteredItems.length === 0 ? (
        <div className="history-empty-state">
          <HistoryIcon size={40} className="text-muted" />
          <h4>
            {historyItems.length === 0 ? 'No analyses recorded yet' : 'No matching analyses found'}
          </h4>
          <p>
            {historyItems.length === 0
              ? 'Upload your first image or video to begin generating analysis records.'
              : 'Try clearing your search query or changing filter settings.'}
          </p>
        </div>
      ) : (
        <div className="history-table-wrapper">
          {/* Desktop Table View */}
          <table className="history-table desktop-only">
            <thead>
              <tr>
                <th>Media</th>
                <th>Filename</th>
                <th>Type</th>
                <th>Result Verdict</th>
                <th>AI Prob</th>
                <th>Confidence</th>
                <th>Date</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredItems.map((item) => (
                <tr key={item.id || item.detection_id}>
                  <td>
                    <div className="thumb-box">
                      {item.media_type === 'video' || item.type === 'video' ? (
                        <FileVideo size={20} className="text-cyan" />
                      ) : (
                        <FileImage size={20} className="text-violet" />
                      )}
                    </div>
                  </td>
                  <td className="font-semibold">{item.filename}</td>
                  <td>
                    <span className="type-tag">{item.media_type || item.type}</span>
                  </td>
                  <td>{getResultBadge(item.result)}</td>
                  <td>
                    <strong className="text-primary">{item.ai_probability}%</strong>
                  </td>
                  <td>{item.confidence}%</td>
                  <td className="text-muted text-sm">{item.created_at || 'Just now'}</td>
                  <td style={{ textAlign: 'right' }}>
                    <div className="actions-flex">
                      <button className="icon-action-btn" title="View details" onClick={() => onViewItem(item)}>
                        <Eye size={16} />
                      </button>
                      <button className="icon-action-btn danger" title="Delete record" onClick={() => onDeleteItem(item.id || item.detection_id)}>
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Mobile Card Grid View */}
          <div className="mobile-only history-cards-list">
            {filteredItems.map((item) => (
              <div key={item.id || item.detection_id} className="history-card-item">
                <div className="h-card-header">
                  <div className="h-card-title">
                    {item.media_type === 'video' ? <FileVideo size={16} /> : <FileImage size={16} />}
                    <span className="font-semibold">{item.filename}</span>
                  </div>
                  {getResultBadge(item.result)}
                </div>

                <div className="h-card-body">
                  <div>
                    <span className="lbl">AI Prob:</span>
                    <strong className="val">{item.ai_probability}%</strong>
                  </div>
                  <div>
                    <span className="lbl">Confidence:</span>
                    <span className="val">{item.confidence}%</span>
                  </div>
                </div>

                <div className="h-card-footer">
                  <span className="text-muted text-xs">{item.created_at || 'Just now'}</span>
                  <div className="actions-flex">
                    <button className="btn btn-secondary btn-sm" onClick={() => onViewItem(item)}>
                      View
                    </button>
                    <button className="btn btn-secondary btn-sm danger" onClick={() => onDeleteItem(item.id || item.detection_id)}>
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default HistoryTable;
