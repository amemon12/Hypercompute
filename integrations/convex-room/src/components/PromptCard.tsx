import type { CSSProperties } from "react";
import { hue, seconds, tps } from "../lib/format";
import { type Prompt, leaseView } from "../lib/types";

interface PromptCardProps {
  p: Prompt;
  mine: boolean;
  now: number;
  leaseMs: number;
  position?: number;
  big?: boolean;
}

const STATUS_LABEL: Record<Prompt["status"], string> = {
  queued: "queued",
  running: "answering",
  done: "done",
  failed: "failed",
};

export function PromptCard({ p, mine, now, leaseMs, position, big = false }: PromptCardProps) {
  const { quiet } = leaseView(p, now, leaseMs);
  const prefill = p.status === "running" && p.firstTokenAt === null && !quiet;
  const label = quiet ? "host quiet" : prefill ? "reading prompt" : STATUS_LABEL[p.status];
  return (
    <article
      className={`card card--${p.status}${quiet ? " card--quiet" : ""}${mine ? " card--mine" : ""}${big ? " card--big" : ""}`}
      style={{ "--h": hue(p.author) } as CSSProperties}
    >
      <header className="card__head">
        <span className="card__author">
          <span className="card__avatar" aria-hidden="true" />
          {p.author}
          {mine && <span className="card__you">you</span>}
        </span>
        <span className={`pill pill--${p.status}${prefill ? " pill--prefill" : ""}${quiet ? " pill--quiet" : ""}`}>
          {label}
          {p.status === "queued" && position !== undefined && <> #{position}</>}
        </span>
      </header>
      <p className="card__q">{p.text}</p>
      {p.reasoning && (
        <p className="card__thinking">
          <span>thinking</span>
          {big ? p.reasoning.slice(-280) : p.reasoning}
        </p>
      )}
      {(p.answer || p.status === "running") && (
        <p className="card__a">
          {p.answer}
          {p.status === "running" && <span className="caret" aria-hidden="true" />}
        </p>
      )}
      {p.note && <p className="card__note">{p.note}</p>}
      {p.error && p.status === "failed" && <p className="card__error">{p.error}</p>}
      <CardFooter p={p} now={now} leaseMs={leaseMs} />
    </article>
  );
}

function CardFooter({ p, now, leaseMs }: { p: Prompt; now: number; leaseMs: number }) {
  if (p.status === "running") {
    const { left, quiet } = leaseView(p, now, leaseMs);
    const frac = Math.min(1, left / leaseMs);
    return (
      <footer className="card__foot">
        <div className="lease" title="Every token the host sends renews its lease. If it goes quiet, Convex requeues the prompt.">
          <div className="lease__bar" style={{ transform: `scaleX(${frac})` }} />
        </div>
        {quiet ? (
          <span className="mono card__quiet">
            no word from the host · Convex requeues this in <strong>{seconds(left)}</strong>
          </span>
        ) : (
          <span className="mono">
            lease {seconds(left)} · {p.startedAt ? seconds(now - p.startedAt) : "0s"} elapsed
            {p.liveTps !== null && <> · <strong>{tps(p.liveTps)}</strong> tok/s live</>}
          </span>
        )}
      </footer>
    );
  }
  if (p.status === "done") {
    return (
      <footer className="card__foot mono">
        {p.tokens ?? "?"} tokens · <strong>{tps(p.decodeTps)}</strong> tok/s decode · {tps(p.prefillTps)} tok/s prefill
        <span className="card__source">llama.cpp timings</span>
      </footer>
    );
  }
  if (p.status === "queued") {
    return <footer className="card__foot mono">waiting {seconds(now - p.createdAt)}</footer>;
  }
  return null;
}
