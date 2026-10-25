import type { Prompt } from "../lib/types";
import { PromptCard } from "./PromptCard";

interface QueueProps {
  prompts: Prompt[];
  mine: Set<string>;
  now: number;
  leaseMs: number;
}

export function Queue({ prompts, mine, now, leaseMs }: QueueProps) {
  if (prompts.length === 0) {
    return <p className="empty">The queue is empty. Ask the pool something.</p>;
  }
  let position = 0;
  return (
    <div className="queue">
      {prompts.map((p) => {
        if (p.status === "queued") position += 1;
        return (
          <PromptCard
            key={p.id}
            p={p}
            mine={mine.has(p.id)}
            now={now}
            leaseMs={leaseMs}
            position={p.status === "queued" ? position : undefined}
          />
        );
      })}
    </div>
  );
}
