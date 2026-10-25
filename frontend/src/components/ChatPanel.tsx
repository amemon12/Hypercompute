import { useEffect, useRef, useState, type FormEvent } from 'react';
import { streamChat, type ApiError, type ChatOutcome } from '../api/client';
import { EMPTY_STREAM_STATE, type ChatStreamState } from '../api/chatStream';
import type { ChatMessage, Deployment } from '../api/types';
import { TRIAL_NOTE, isTuningTrial, readyRun } from '../lib/deployments';
import { formatCount, formatMs, formatRate } from '../lib/format';
import { Card, ErrorNote } from './common';

interface Turn {
  user: string;
  reply: ChatStreamState;
  ttftMs: number | null;
  outcome: ChatOutcome | null; // null while streaming
}

interface ChatPanelProps {
  deployment: Deployment | null; // the newest deployment; chat uses it only when it is the owner's ready run
  maxTokens: number | undefined;
}

function TurnMeta({ turn }: { turn: Turn }) {
  const o = turn.outcome;
  if (!o) {
    return <p className="meta small">streaming… · TTFT {turn.ttftMs === null ? 'waiting for first token' : formatMs(turn.ttftMs)}</p>;
  }
  const t = o.state.timings;
  const finish = o.status === 'stopped' ? 'stopped by you' : (o.state.finishReason ?? 'none reported');
  return (
    <p className="meta small">
      finish: {finish} · TTFT {formatMs(o.ttftMs)} (measured in this browser) · total {formatMs(o.totalMs)}
      {' · '}
      {t ? `llama.cpp: ${formatCount(t.predicted_n)} tokens at ${formatRate(t.predicted_per_second)}` : 'llama.cpp timings: unavailable'}
    </p>
  );
}

export function ChatPanel({ deployment, maxTokens }: ChatPanelProps) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [error, setError] = useState<ApiError | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streaming = turns.at(-1)?.outcome === null;
  const deploymentId = deployment?.id ?? null;

  // A different deployment means a different model; start a fresh conversation.
  useEffect(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setTurns([]);
    setError(null);
  }, [deploymentId]);

  const updateLast = (patch: Partial<Turn>): void =>
    setTurns((prev) => prev.map((t, i) => (i === prev.length - 1 ? { ...t, ...patch } : t)));

  const run = readyRun(deployment); // a tuning trial is busy being measured and is never the owner's chat
  const send = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    const text = input.trim();
    if (!run || !text || streaming) return;
    const history: ChatMessage[] = turns.flatMap((t) =>
      t.reply.content ? [{ role: 'user' as const, content: t.user }, { role: 'assistant' as const, content: t.reply.content }] : [],
    );
    const controller = new AbortController();
    abortRef.current = controller;
    setInput('');
    setError(null);
    setTurns((prev) => [...prev, { user: text, reply: EMPTY_STREAM_STATE, ttftMs: null, outcome: null }]);
    const outcome = await streamChat({
      model: run.plan.model_id,
      messages: [...history, { role: 'user', content: text }],
      maxTokens,
      signal: controller.signal,
      onUpdate: (reply, ttftMs) => updateLast({ reply, ttftMs }),
    });
    if (abortRef.current === controller) abortRef.current = null;
    updateLast({ reply: outcome.state, ttftMs: outcome.ttftMs, outcome });
    if (outcome.error) setError(outcome.error);
  };

  const stop = (): void => abortRef.current?.abort();
  const enabled = run !== null;
  const trial = isTuningTrial(deployment);
  const idle = trial ? `Chat is off. ${TRIAL_NOTE}` : 'Start a run and wait for the Ready step to chat with the model.';

  return (
    <Card
      title="Chat"
      subtitle={run ? `Talking to ${run.plan.model_id}` : trial ? 'Off during a tuning trial' : 'Available when a run is ready'}
      actions={
        turns.length > 0 && !streaming ? (
          <button type="button" onClick={() => setTurns([])}>
            Clear
          </button>
        ) : undefined
      }
    >
      {!enabled && <p className="muted">{idle}</p>}
      <div className="chat-log" aria-live="polite">
        {turns.map((t, i) => (
          <div key={i} className="turn">
            <p className="msg msg-user">{t.user}</p>
            {t.reply.reasoning && (
              <details className="reasoning" open={t.outcome === null && !t.reply.content}>
                <summary className="small">model reasoning</summary>
                <p className="muted small pre">{t.reply.reasoning}</p>
              </details>
            )}
            <p className="msg msg-model pre">{t.reply.content || (t.outcome === null ? '…' : '')}</p>
            <TurnMeta turn={t} />
          </div>
        ))}
      </div>
      <ErrorNote error={error} onDismiss={() => setError(null)} />
      <form className="chat-form" onSubmit={(e) => void send(e)}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={enabled ? 'Ask something…' : 'Chat is disabled until a run is ready'}
          disabled={!enabled}
          rows={3}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              e.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <div className="chat-buttons">
          {streaming ? (
            <button type="button" onClick={stop}>
              Stop
            </button>
          ) : (
            <button type="submit" className="primary" disabled={!enabled || input.trim() === ''}>
              Send
            </button>
          )}
        </div>
      </form>
    </Card>
  );
}
