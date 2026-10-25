import type { CSSProperties } from "react";
import { gib } from "../lib/format";
import type { Machine } from "../lib/types";

const PATH_LABEL: Record<Machine["path"], string> = {
  local: "host",
  direct: "direct",
  relay: "relay",
  connecting: "connecting",
  lan: "LAN",
  unknown: "link ?",
};

const OS_LABEL: Record<string, string> = { linux: "Linux", windows: "Windows", macos: "macOS" };

interface MachineListProps {
  machines: Machine[];
  colorOf: (id: string) => string;
  compact?: boolean;
}

export function MachineList({ machines, colorOf, compact = false }: MachineListProps) {
  if (machines.length === 0) {
    return <p className="empty">No machines have reported in yet.</p>;
  }
  return (
    <ul className={`machines${compact ? " machines--compact" : ""}`}>
      {machines.map((m, i) => (
        <li
          key={m.id}
          className={`machine${m.layers === 0 ? " machine--idle" : ""}`}
          style={{ "--c": colorOf(m.id), animationDelay: `${i * 60}ms` } as CSSProperties}
        >
          <div className="machine__top">
            <span className="machine__swatch" aria-hidden="true" />
            <span className="machine__name">{m.name}</span>
            <span className={`badge badge--${m.path}`}>{PATH_LABEL[m.path]}</span>
          </div>
          <div className="machine__stats">
            <span className="chip">{m.device}</span>
            {m.os && <span className="machine__os">{OS_LABEL[m.os] ?? m.os}</span>}
          </div>
          {m.layers > 0 ? (
            <div className="machine__load">
              <strong>{m.layers}</strong> layers
              <span className="machine__range">{m.layerRange}</span>
              <span className="machine__gib">
                {gib(m.weightBytes)} <small>GiB</small>
              </span>
              {m.holdsOutput && <span className="machine__head">output head</span>}
            </div>
          ) : (
            <div className="machine__load machine__load--idle">joined, not in this plan</div>
          )}
        </li>
      ))}
    </ul>
  );
}
