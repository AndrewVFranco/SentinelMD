import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import ClaimItem from './ClaimItem';
import DrugCarousel from './DrugCarousel';

const FHIR_RESOURCE_TYPES = ['Condition', 'MedicationRequest', 'DiagnosticReport'];

function ChatWindow({ messages, query, onQueryChange, onSubmit, loading }) {
  const bottomRef = useRef(null);
  const [showFhir, setShowFhir] = useState(false);
  const [fhirType, setFhirType] = useState('Condition');
  const [fhirId, setFhirId] = useState('');

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    const fhirData = showFhir && fhirId.trim() ? {
      has_fhir: true,
      fhir_resource_type: fhirType,
      fhir_resource_id: fhirId.trim(),
    } : {
      has_fhir: false,
      fhir_resource_type: null,
      fhir_resource_id: null,
    };
    onSubmit(fhirData);
  };

  const canSubmit = query.trim() || (showFhir && fhirId.trim());

  return (
    <main className="chat-window">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Ask a clinical question</h2>
            <p>
              The system retrieves PubMed abstracts and verifies each claim
              against medical literature using NLI scoring. Optionally attach
              a FHIR resource for structured clinical context.
            </p>
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.role === 'user') {
            return (
              <div key={i} className="message-user">
                <div className="bubble">{msg.text}</div>
              </div>
            );
          }

          if (msg.role === 'error') {
            return (
              <div key={i} className="message-error">
                <div className="bubble">{msg.text}</div>
              </div>
            );
          }

          if (msg.role === 'assistant') {
            return (
              <div key={i} className="message-assistant">
                <div className="assistant-response">
                  <ReactMarkdown>{msg.data.response}</ReactMarkdown>
                </div>
                <DrugCarousel abstracts={msg.data.abstracts} />
                {msg.data.scored_claims && msg.data.scored_claims.length > 0 && (
                  <div className="claims-section">
                    <div className="claims-heading">Claim Verification</div>
                    {msg.data.scored_claims.map((claim, j) => (
                      <ClaimItem key={j} claim={claim} />
                    ))}
                  </div>
                )}
              </div>
            );
          }

          return null;
        })}

        {loading && (
          <div className="message-assistant">
            <div className="typing-indicator">
              <div className="typing-dot" />
              <div className="typing-dot" />
              <div className="typing-dot" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="input-area">
        {showFhir && (
          <div style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
              <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-3)' }}>
                FHIR Resource
              </span>
              <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 10, color: 'var(--text-3)' }}>
                · HAPI R4 Server
              </span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <select
                value={fhirType}
                onChange={(e) => setFhirType(e.target.value)}
                style={{
                  background: 'var(--bg)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  color: 'var(--text)',
                  fontFamily: 'DM Mono, monospace',
                  fontSize: 12,
                  padding: '6px 10px',
                  outline: 'none',
                  cursor: 'pointer',
                  flexShrink: 0,
                }}
              >
                {FHIR_RESOURCE_TYPES.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
              <input
                type="text"
                placeholder="Resource ID (e.g. 123456)"
                value={fhirId}
                onChange={(e) => setFhirId(e.target.value)}
                style={{
                  flex: 1,
                  background: 'var(--bg)',
                  border: '1px solid var(--border)',
                  borderRadius: 6,
                  color: 'var(--text)',
                  fontFamily: 'DM Mono, monospace',
                  fontSize: 12,
                  padding: '6px 10px',
                  outline: 'none',
                }}
              />
            </div>
          </div>
        )}

        <div className="input-row">
          <button
            onClick={() => setShowFhir(!showFhir)}
            title="Attach FHIR resource"
            style={{
              background: showFhir ? 'var(--text)' : 'transparent',
              color: showFhir ? 'var(--bg)' : 'var(--text-3)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              width: 32,
              height: 32,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              flexShrink: 0,
              fontSize: 13,
              transition: 'all 0.15s',
            }}
          >
            ⬡
          </button>
          <textarea
            className="chat-textarea"
            placeholder={showFhir ? 'Optional: add a specific question about this resource…' : 'Ask a clinical question…'}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          <button
            className="send-btn"
            onClick={handleSubmit}
            disabled={loading || !canSubmit}
          >
            ↑
          </button>
        </div>
        <div className="input-hint">Enter to send · Shift+Enter for new line</div>
      </div>
    </main>
  );
}

export default ChatWindow;
