import type { CSSProperties } from "react";
import type { Machine } from "../lib/types";

interface LayerRibbonProps {
  machines: Machine[];
  colorOf: (id: string) => string;
  /** "decode": tokens are flowing; "prefill": the prompt is being read; null: idle. */
  flow: "decode" | "prefill" | null;
  liveTps: number | null;
  big?: boolean;
}

/**
 * The served model drawn as one strip of layers, split by the machine that holds each run. While a prompt decodes,
 * a pulse walks the whole strip once per token, because every token passes through every machine in turn.
 */
export function LayerRibbon({ machines, colorOf, flow, liveTps, big = false }: LayerRibbonProps) {
  const used = machines.filter((m) => m.layers > 0);
  const total = used.reduce((sum, m) => sum + m.layers, 0);
  if (total === 0) {
    return <div className="ribbon ribbon--empty">No model is loaded on the pool right now.</div>;
  }
  const period = flow === "decode" ? Math.min(Math.max(liveTps ? 1 / liveTps : 1.2, 0.35), 3) : 2.4;
  return (
    <figure className={`ribbon${big ? " ribbon--big" : ""}`} aria-label={`${total} layers across ${used.length} machines`}>
      <div className="ribbon__track">
        {used.map((m) => (
          <div
            key={m.id}
            className="ribbon__seg"
            style={{ flexGrow: m.layers, "--c": colorOf(m.id) } as CSSProperties}
            title={`${m.name}: layers ${m.layerRange ?? "?"} on ${m.device}`}
          >
            <span className="ribbon__name">{m.name}</span>
            <span className="ribbon__range">{m.layerRange}</span>
          </div>
        ))}
        {flow && (
          <span
            className={`ribbon__pulse ribbon__pulse--${flow}`}
            style={{ animationDuration: `${period}s` }}
            aria-hidden="true"
          />
        )}
      </div>
      <figcaption className="ribbon__scale">
        <span>layer 0</span>
        <span>
          {flow === "decode" ? "each token crosses every machine" : flow === "prefill" ? "reading the prompt" : `${total} layers`}
        </span>
        <span>layer {total - 1}</span>
      </figcaption>
    </figure>
  );
}
