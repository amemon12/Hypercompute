import { useId } from 'react';
import { OpenComputeDiagram } from './OpenComputeDiagram';
import './session-section.css';

function PrintDetails() {
  const grainId = `${useId()}-print-grain`;

  return (
    <div className="hc-session-print" aria-hidden="true">
      <svg className="hc-session-registration" viewBox="0 0 52 52" focusable="false">
        <path d="M26 2V50M2 26H50" />
      </svg>
      <svg className="hc-session-corner hc-session-corner-top" viewBox="0 0 250 220" focusable="false">
        <defs>
          <pattern id={grainId} width="60" height="60" patternUnits="userSpaceOnUse">
            {Array.from({ length: 32 }, (_, index) => (
              <circle key={index} cx={(index * 37 + 9) % 60} cy={(index * index * 13 + 7) % 60} r={index % 3 === 0 ? .65 : .35} fill="#f4efe5" opacity=".3" />
            ))}
          </pattern>
        </defs>
        <rect x="85" width="165" height="150" fill="#c78251" />
        <rect x="85" width="165" height="150" fill={`url(#${grainId})`} />
        <path className="hc-session-ink" d="M1 105C92 35 173 7 188 25C204 45 130 139 116 185" />
      </svg>
      <svg className="hc-session-corner hc-session-corner-bottom" viewBox="0 0 330 250" focusable="false">
        <path d="M0 33C113 32 205 112 205 227V250H0Z" fill="#7c9b68" />
        <path d="M0 33C113 32 205 112 205 227V250H0Z" fill={`url(#${grainId})`} />
        <path className="hc-session-ink" d="M-10 98C62 50 148 17 165 35C187 58 94 152 80 226M74 183H320" />
      </svg>
    </div>
  );
}

export function SessionSection() {
  return (
    <section className="hc-session-section hc-panel" id="how-it-works" aria-labelledby="session-title">
      <PrintDetails />
      <div className="hc-session-layout">
        <div className="hc-session-copy">
          <h2 id="session-title">Runs where<br />your model lives.</h2>
          <p>The model stays on your machines. Hypercompute assigns layers across selected GPUs and routes each token through them in sequence.</p>
        </div>
        <OpenComputeDiagram />
      </div>
    </section>
  );
}
