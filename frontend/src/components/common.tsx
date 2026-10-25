import { useState, type ReactNode } from 'react';
import type { ApiError } from '../api/client';

interface ErrorNoteProps {
  error: ApiError | null;
  onDismiss?: () => void;
}

/** Friendly rendering of the gateway's {detail, code, fix} errors. */
export function ErrorNote({ error, onDismiss }: ErrorNoteProps) {
  if (!error) return null;
  return (
    <div className="error-note" role="alert">
      <p className="error-detail">{error.message}</p>
      {error.fix && (
        <p className="error-fix">
          <strong>How to fix:</strong> {error.fix}
        </p>
      )}
      <p className="muted small">
        {error.code ? `code ${error.code}` : 'no error code'}
        {error.status > 0 ? ` · HTTP ${error.status}` : ''}
        {onDismiss && (
          <>
            {' · '}
            <button type="button" className="link-button" onClick={onDismiss}>
              dismiss
            </button>
          </>
        )}
      </p>
    </div>
  );
}

interface CopyBlockProps {
  text: string;
  label: string;
}

export function CopyBlock({ text, label }: CopyBlockProps) {
  const [copied, setCopied] = useState<'idle' | 'copied' | 'failed'>('idle');
  const copy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied('copied');
    } catch {
      setCopied('failed');
    }
    setTimeout(() => setCopied('idle'), 2000);
  };
  return (
    <div className="copy-block">
      <pre aria-label={label}>
        <code>{text}</code>
      </pre>
      <button type="button" onClick={() => void copy()}>
        {copied === 'copied' ? 'Copied' : copied === 'failed' ? 'Select and copy manually' : 'Copy'}
      </button>
    </div>
  );
}

interface CardProps {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Card({ title, subtitle, actions, children, className }: CardProps) {
  return (
    <section className={`card ${className ?? ''}`}>
      <header className="card-head">
        <div>
          <h2>{title}</h2>
          {subtitle && <p className="muted small">{subtitle}</p>}
        </div>
        {actions && <div className="card-actions">{actions}</div>}
      </header>
      {children}
    </section>
  );
}

interface FieldProps {
  label: string;
  children: ReactNode;
}

/** One label/value row inside a <dl className="fields">. */
export function Field({ label, children }: FieldProps) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </>
  );
}

interface StatusTextProps {
  tone: 'ok' | 'warn' | 'bad' | 'neutral';
  children: ReactNode;
}

/** Status is carried by the words; tone only adjusts weight/marker, never the meaning. */
export function StatusText({ tone, children }: StatusTextProps) {
  return <span className={`status status-${tone}`}>{children}</span>;
}
