import { useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import { seconds } from "../lib/format";

interface ActivityLogProps {
  code: string;
  now: number;
  leaseMs: number;
  max?: number;
}

const KIND_LABEL: Record<string, string> = {
  submit: "queue",
  claim: "claim",
  done: "done",
  busy: "busy",
  requeue: "requeue",
  lease: "lease",
  failed: "failed",
  fence: "fence",
  host: "host",
  admin: "owner",
};

/** What Convex just did for the room, written in the same transactions as the changes themselves. */
export function ActivityLog({ code, now, leaseMs, max = 7 }: ActivityLogProps) {
  const events = useQuery(api.events.list, { code }) ?? [];
  const shown = events.slice(-max);
  return (
    <section className="log" aria-label="Convex activity">
      <h2 className="log__title">
        <span className="log__live" aria-hidden="true" />
        Convex, live
      </h2>
      {shown.length === 0 ? (
        <p className="log__empty">Claims, leases and requeues will show up here.</p>
      ) : (
        <ol className="log__list">
          {shown.map((e) => (
            <li key={e.id} className={`log__row log__row--${e.kind}`}>
              <span className="log__kind">{KIND_LABEL[e.kind] ?? e.kind}</span>
              <span className="log__text">{e.text}</span>
              <span className="log__age">{seconds(now - e.at)}</span>
            </li>
          ))}
        </ol>
      )}
      <p className="log__mech">
        transactional claim · {Math.round(leaseMs / 1000)} s lease · scheduled requeue · fencing · rate limits ·
        presence
      </p>
    </section>
  );
}
