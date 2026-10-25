import { useMutation } from "convex/react";
import { ConvexError } from "convex/values";
import { type FormEvent, useState } from "react";
import { api } from "../../convex/_generated/api";
import type { Id } from "../../convex/_generated/dataModel";
import type { Identity } from "../lib/identity";
import { seconds, useNow } from "../lib/format";

const MAX = 500;
const SUGGESTIONS = [
  "Why split one model across laptops?",
  "Explain Convex in one sentence.",
  "Write a haiku about borrowed GPUs.",
  "What is a lease in a job queue?",
];

interface ComposerProps {
  code: string;
  identity: Identity;
  open: boolean;
  onSent: (id: Id<"prompts">) => void;
}

export function Composer({ code, identity, open, onSent }: ComposerProps) {
  const submit = useMutation(api.prompts.submit);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryAt, setRetryAt] = useState<number | null>(null);
  const now = useNow(500);
  const waiting = retryAt !== null && retryAt > now;

  async function send(e: FormEvent) {
    e.preventDefault();
    if (!text.trim() || sending || waiting) return;
    setSending(true);
    setError(null);
    try {
      const { id } = await submit({ code, author: identity.name, sessionId: identity.sessionId, text });
      setText("");
      setRetryAt(null);
      onSent(id);
    } catch (err) {
      const { message, retryAfter } = explain(err);
      setError(message);
      setRetryAt(retryAfter !== null ? Date.now() + retryAfter : null);
    } finally {
      setSending(false);
    }
  }

  const left = MAX - text.length;
  return (
    <form className="composer" onSubmit={send}>
      <label className="composer__label" htmlFor="prompt">
        Ask the pool
      </label>
      {open && !text && (
        <div className="composer__ideas" role="group" aria-label="Ideas">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" className="idea" onClick={() => setText(s)}>
              {s}
            </button>
          ))}
        </div>
      )}
      <textarea
        id="prompt"
        value={text}
        maxLength={MAX}
        rows={3}
        placeholder={open ? "Ask a short question. Every screen in the room sees it." : "The host closed this room."}
        disabled={!open}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void send(e);
        }}
      />
      <div className="composer__row">
        <span className={`composer__count${left < 50 ? " composer__count--low" : ""}`}>{left}</span>
        {error && (
          <span className="composer__error" role="alert">
            {waiting ? `${error} Try again in ${seconds(retryAt - now)}.` : error}
          </span>
        )}
        <button type="submit" disabled={!open || sending || waiting || !text.trim()}>
          {sending ? "Sending" : waiting ? seconds(retryAt - now) : "Send"}
        </button>
      </div>
    </form>
  );
}

function explain(err: unknown): { message: string; retryAfter: number | null } {
  if (err instanceof ConvexError && typeof err.data === "object" && err.data !== null) {
    const data = err.data as Record<string, unknown>;
    if (data.kind === "RateLimited") {
      const room = data.name === "promptRoom";
      return {
        message: room ? "The room is busy." : "Slow down a little.",
        retryAfter: typeof data.retryAfter === "number" ? data.retryAfter : 30_000,
      };
    }
    if (typeof data.message === "string") return { message: data.message, retryAfter: null };
  }
  return { message: "Could not send. Check your connection and try again.", retryAfter: null };
}
