import React, { useState } from 'react';

function ClaimItem({ claim }) {
  const [expanded, setExpanded] = useState(false);
  const label = claim.label.toLowerCase();

  return (
    <div className={`claim-item ${label}`}>
      <div className="claim-header-row" onClick={() => setExpanded(!expanded)}>
        <span className={`claim-badge ${label}`}>{claim.label}</span>
        <span className="claim-text">{claim.claim}</span>
        {claim.evidence && (
          <span className="claim-chevron">{expanded ? '▲' : '▼'}</span>
        )}
      </div>

      {expanded && claim.evidence && (
        <div className="claim-evidence-panel">
          <div className="evidence-label">Evidence</div>
          <div className="evidence-text">{claim.evidence}</div>
          {claim.score > 0 && (
            <div className="evidence-score">
              NLI confidence: {(claim.score * 100).toFixed(1)}%
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ClaimItem;
