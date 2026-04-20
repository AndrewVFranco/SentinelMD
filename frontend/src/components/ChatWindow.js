import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import ClaimItem from './ClaimItem';
import DrugCarousel from './DrugCarousel';

function ChatWindow({ messages, query, onQueryChange, onSubmit, loading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <main className="chat-window">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <h2>Ask a clinical question</h2>
            <p>
              The system will retrieve relevant PubMed abstracts and verify each
              claim in the response against medical literature using NLI scoring.
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
        <div className="input-row">
          <textarea
            className="chat-textarea"
            placeholder="Ask a clinical question…"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          <button
            className="send-btn"
            onClick={onSubmit}
            disabled={loading || !query.trim()}
          >
            ↑
          </button>
        </div>
      </div>
    </main>
  );
}

export default ChatWindow;
