import type { CSSProperties } from "react";
import { displayName } from "../lib/identity";
import { hue, initials } from "../lib/format";
import { PROJECTOR_ID } from "../lib/types";

export interface Person {
  userId: string;
  online: boolean;
}

interface FacePileProps {
  people: Person[] | undefined;
  me?: string;
  big?: boolean;
  max?: number;
}

export function FacePile({ people, me, big = false, max = 14 }: FacePileProps) {
  const online = (people ?? []).filter((p) => p.online && p.userId !== PROJECTOR_ID);
  const shown = online.slice(0, max);
  const extra = online.length - shown.length;
  return (
    <div className={`faces${big ? " faces--big" : ""}`} aria-label={`${online.length} people in the room`}>
      <ul className="faces__row">
        {shown.map((p) => {
          const name = displayName(p.userId);
          return (
            <li
              key={p.userId}
              className={`face${p.userId === me ? " face--me" : ""}`}
              style={{ "--h": hue(p.userId) } as CSSProperties}
              title={name}
            >
              <span className="face__dot">{initials(name)}</span>
              {big && <span className="face__name">{name}</span>}
            </li>
          );
        })}
        {extra > 0 && (
          <li className="face face--more">
            <span className="face__dot">+{extra}</span>
          </li>
        )}
      </ul>
      <span className="faces__count">
        {online.length === 0 ? "nobody here yet" : online.length === 1 ? "1 person here" : `${online.length} people here`}
      </span>
    </div>
  );
}
