import React, { useState } from 'react';

const SECTION_ORDER = [
  'indications_and_usage',
  'dosage_and_administration',
  'warnings',
  'contraindications',
  'adverse_reactions',
  'drug_interactions',
];

const SECTION_LABELS = {
  indications_and_usage: 'Indications & Usage',
  dosage_and_administration: 'Dosage & Administration',
  warnings: 'Warnings',
  contraindications: 'Contraindications',
  adverse_reactions: 'Adverse Reactions',
  drug_interactions: 'Drug Interactions',
};

const SECTION_ACCENT = {
  indications_and_usage: '#2563eb',
  dosage_and_administration: '#16a34a',
  warnings: '#d97706',
  contraindications: '#dc2626',
  adverse_reactions: '#b45309',
  drug_interactions: '#7c3aed',
};

function getSectionKey(pmid) {
  const parts = pmid.split('-');
  return parts.slice(2).join('_');
}

function getDrugName(pmid) {
  const parts = pmid.split('-');
  return parts[1];
}

function DrugCard({ drug, sections }) {
  const sorted = [...sections].sort((a, b) => {
    const aKey = getSectionKey(a.pmid);
    const bKey = getSectionKey(b.pmid);
    const aIdx = SECTION_ORDER.indexOf(aKey);
    const bIdx = SECTION_ORDER.indexOf(bKey);
    return (aIdx === -1 ? 99 : aIdx) - (bIdx === -1 ? 99 : bIdx);
  });

  const [activeSection, setActiveSection] = useState(0);
  const current = sorted[activeSection];
  const sectionKey = getSectionKey(current.pmid);
  const accent = SECTION_ACCENT[sectionKey] || 'var(--text-3)';

  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <div style={{
        padding: '10px 14px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
      }}>
        <span style={{
          fontFamily: 'DM Mono, monospace',
          fontSize: 10,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--text-3)',
        }}>FDA Drug Label</span>
        <span style={{
          fontFamily: 'Instrument Serif, serif',
          fontSize: 15,
          color: 'var(--text)',
          textTransform: 'capitalize',
        }}>{drug}</span>
      </div>

      <div style={{
        display: 'flex',
        gap: 6,
        padding: '10px 14px',
        overflowX: 'auto',
        borderBottom: '1px solid var(--border)',
        scrollbarWidth: 'none',
      }}>
        {sorted.map((s, i) => {
          const key = getSectionKey(s.pmid);
          const a = SECTION_ACCENT[key] || 'var(--text-3)';
          const isActive = i === activeSection;
          return (
            <button
              key={i}
              onClick={() => setActiveSection(i)}
              style={{
                flexShrink: 0,
                padding: '4px 10px',
                borderRadius: 100,
                border: `1px solid ${isActive ? a : 'var(--border)'}`,
                background: 'transparent',
                color: isActive ? a : 'var(--text-2)',
                fontSize: 11,
                fontFamily: 'DM Mono, monospace',
                cursor: 'pointer',
                transition: 'all 0.15s',
                whiteSpace: 'nowrap',
              }}
            >
              {SECTION_LABELS[key] || key}
            </button>
          );
        })}
      </div>

      <div style={{ padding: '12px 14px' }}>
        <div style={{
          padding: '10px 14px',
          background: 'var(--bg)',
          borderLeft: `3px solid ${accent}`,
          borderRadius: 4,
        }}>
          <div style={{
            fontSize: 10,
            fontFamily: 'DM Mono, monospace',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            color: accent,
            marginBottom: 6,
          }}>
            {SECTION_LABELS[sectionKey] || sectionKey}
          </div>
          <div style={{
            fontSize: 12,
            lineHeight: 1.7,
            color: 'var(--text)',
            maxHeight: 180,
            overflowY: 'auto',
          }}>
            {current.abstract}
          </div>
        </div>
      </div>

      <div style={{
        padding: '0 14px 10px',
        fontFamily: 'DM Mono, monospace',
        fontSize: 10,
        color: 'var(--text-3)',
      }}>
        Source: U.S. Food & Drug Administration · {current.pmid}
      </div>
    </div>
  );
}

function DrugCarousel({ abstracts }) {
  const [currentDrug, setCurrentDrug] = useState(0);

  if (!abstracts || abstracts.length === 0) return null;

  const fdaAbstracts = abstracts.filter(a => a.pmid && a.pmid.startsWith('FDA-'));
  if (fdaAbstracts.length === 0) return null;

  const drugMap = {};
  fdaAbstracts.forEach(a => {
    const drug = getDrugName(a.pmid);
    if (!drugMap[drug]) drugMap[drug] = [];
    drugMap[drug].push(a);
  });

  const drugs = Object.keys(drugMap);
  if (drugs.length === 0) return null;

  return (
    <div>
      <div style={{
        fontFamily: 'DM Mono, monospace',
        fontSize: 10,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: 'var(--text-3)',
        marginBottom: 8,
      }}>
        FDA Drug Labels · {drugs.length} medication{drugs.length > 1 ? 's' : ''} detected
      </div>

      {drugs.length > 1 && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
          {drugs.map((drug, i) => (
            <button
              key={drug}
              onClick={() => setCurrentDrug(i)}
              style={{
                padding: '4px 12px',
                borderRadius: 100,
                border: `1px solid ${i === currentDrug ? 'var(--text)' : 'var(--border)'}`,
                background: i === currentDrug ? 'var(--text)' : 'transparent',
                color: i === currentDrug ? 'var(--bg)' : 'var(--text-2)',
                fontSize: 12,
                fontFamily: 'DM Sans, sans-serif',
                cursor: 'pointer',
                textTransform: 'capitalize',
                transition: 'all 0.15s',
              }}
            >
              {drug}
            </button>
          ))}
        </div>
      )}

      <DrugCard drug={drugs[currentDrug]} sections={drugMap[drugs[currentDrug]]} />
    </div>
  );
}

export default DrugCarousel;