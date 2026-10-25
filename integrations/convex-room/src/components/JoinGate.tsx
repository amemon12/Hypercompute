import { type FormEvent, useState } from "react";
import type { Room } from "../lib/types";
import { useRoom } from "../lib/useRoom";
import { LayerRibbon } from "./LayerRibbon";

interface JoinGateProps {
  room: Room;
  initialName: string;
  onJoin: (name: string) => void;
}

export function JoinGate({ room, initialName, onJoin }: JoinGateProps) {
  const [name, setName] = useState(initialName);
  const { machines, colorOf } = useRoom(room.code);
  const used = machines.filter((m) => m.layers > 0).length;
  const clean = name
    .replace(/[#\u0000-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]/g, "")
    .replace(/\s+/g, " ")
    .trim();

  function join(e: FormEvent) {
    e.preventDefault();
    if (clean) onJoin(clean.slice(0, 32));
  }

  return (
    <main className="gate">
      <p className="gate__kicker">Hypercompute · room {room.code}</p>
      <h1 className="gate__title">{room.title}</h1>
      <p className="gate__lede">
        {used > 1 ? `${used} machines are` : "Friends' machines are"} running{" "}
        {room.modelId ? <span className="gate__model">{room.modelId}</span> : "one model"} together right now. Join to
        watch it think and ask it something from your phone.
      </p>
      <div className="gate__ribbon">
        <LayerRibbon machines={machines} colorOf={colorOf} flow={null} liveTps={null} />
      </div>
      <form className="gate__form" onSubmit={join}>
        <label htmlFor="name">What should the room call you?</label>
        <div className="gate__row">
          <input
            id="name"
            autoFocus
            autoComplete="nickname"
            maxLength={32}
            value={name}
            placeholder="Your name"
            onChange={(e) => setName(e.target.value)}
          />
          <button type="submit" disabled={!clean}>
            Join
          </button>
        </div>
        <p className="gate__fine">Your name and questions are shown to everyone in the room.</p>
      </form>
    </main>
  );
}
