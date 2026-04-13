import React from 'react';

function Sidebar({ apiKey, onApiKeyChange, result }) {
  const score = result ? Math.round(result.confidence_score * 100) : null;

  const getScoreColor = (pct) => {
    if (pct >= 70) return '#16a34a';
    if (pct >= 40) return '#d97706';
    return '#dc2626';
  };

  const getScoreLabel = (pct) => {
    if (pct >= 70) return 'High reliability';
    if (pct >= 40) return 'Moderate reliability';
    return 'Low reliability — review carefully';
  };

  const supported = result ? result.scored_claims.filter(c => c.label === 'Supported').length : 0;
  const contradicted = result ? result.scored_claims.filter(c => c.label === 'Contradicted').length : 0;
  const unverifiable = result ? result.scored_claims.filter(c => c.label === 'Unverifiable').length : 0;

  return (
    <aside className="sidebar">
      <div className="sidebar-section">
        <div className="sidebar-label">Gemini API Key</div>
        <input
          className="api-input"
          type="password"
          placeholder="AIza..."
          value={apiKey}
          onChange={(e) => onApiKeyChange(e.target.value)}
        />
        <p className="api-note">Never stored or logged.</p>
      </div>

      {result && (
        <>
          <div className="sidebar-section">
            <div className="sidebar-label">Reliability Score</div>
            <div className="score-value" style={{ color: getScoreColor(score) }}>
              {score}%
            </div>
            <div className="meter-track">
              <div
                className="meter-fill"
                style={{
                  width: `${score}%`,
                  background: getScoreColor(score)
                }}
              />
            </div>
            <div className="score-label">{getScoreLabel(score)}</div>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-label">Claim Breakdown</div>
            <div className="stat-row">
              <span className="stat-label">
                <span className="stat-dot" style={{ background: '#16a34a' }} />
                Supported
              </span>
              <span className="stat-count" style={{ color: '#16a34a' }}>{supported}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">
                <span className="stat-dot" style={{ background: '#dc2626' }} />
                Contradicted
              </span>
              <span className="stat-count" style={{ color: '#dc2626' }}>{contradicted}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">
                <span className="stat-dot" style={{ background: '#d97706' }} />
                Unverifiable
              </span>
              <span className="stat-count" style={{ color: '#d97706' }}>{unverifiable}</span>
            </div>
          </div>

          <div className="sidebar-section">
            <div className="sidebar-label">Sources</div>
            {result.abstracts.slice(0, 5).map((a, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.4, marginBottom: 2 }}>
                  {a.title}
                </div>
                <div style={{ fontFamily: 'DM Mono, monospace', fontSize: 10, color: 'var(--text-3)' }}>
                  PMID {a.pmid}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {!result && (
        <div className="sidebar-section">
          <div className="sidebar-label">About</div>
          <p style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6 }}>
            Submit a clinical question. The system retrieves PubMed literature via RAG and verifies each LLM claim using NLI scoring.
          </p>
        </div>
      )}
    </aside>
  );
}

export default Sidebar;
