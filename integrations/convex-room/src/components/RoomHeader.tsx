import { tps } from "../lib/format";
import { HOST_LABEL, type Room, hostState } from "../lib/types";

interface RoomHeaderProps {
  room: Room;
  now: number;
  big?: boolean;
}

export function RoomHeader({ room, now, big = false }: RoomHeaderProps) {
  const state = hostState(room, now);
  return (
    <header className={`bar${big ? " bar--big" : ""}`}>
      <div className="bar__id">
        <span className="bar__kicker">Hypercompute Pool Room · {room.code}</span>
        <h1 className="bar__title">{room.title}</h1>
      </div>
      <dl className="bar__stats">
        <div>
          <dt>model</dt>
          <dd className="mono">{room.modelId ?? "none"}</dd>
        </div>
        <div>
          <dt>last decode</dt>
          <dd className="mono">
            {room.lastDecodeTps !== null ? (
              <>
                <strong>{tps(room.lastDecodeTps)}</strong> tok/s
              </>
            ) : (
              "n/a"
            )}
          </dd>
        </div>
        <div>
          <dt>answered</dt>
          <dd className="mono">{room.servedCount}</dd>
        </div>
        <div>
          <dt>host</dt>
          <dd className={`host host--${state}`}>
            <span className="host__dot" aria-hidden="true" />
            {HOST_LABEL[state]}
            {!room.open && " · closed"}
          </dd>
        </div>
      </dl>
    </header>
  );
}
