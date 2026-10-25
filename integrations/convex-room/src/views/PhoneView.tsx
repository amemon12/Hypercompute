import usePresence from "@convex-dev/presence/react";
import { useState } from "react";
import { api } from "../../convex/_generated/api";
import { Composer } from "../components/Composer";
import { FacePile } from "../components/FacePile";
import { LayerRibbon } from "../components/LayerRibbon";
import { MachineList } from "../components/MachineList";
import { Queue } from "../components/Queue";
import { RoomHeader } from "../components/RoomHeader";
import { useServerNow } from "../lib/clock";
import { type Identity, loadMine, presenceId, rememberMine } from "../lib/identity";
import { type Room, flowOf } from "../lib/types";
import { useRoom } from "../lib/useRoom";

interface PhoneViewProps {
  room: Room;
  identity: Identity;
  onRename: () => void;
}

export function PhoneView({ room, identity, onRename }: PhoneViewProps) {
  const { machines, prompts, colorOf, running } = useRoom(room.code);
  const me = presenceId(identity);
  const people = usePresence(api.presence, room.code, me);
  const now = useServerNow(room.bridgeSeenAt);
  const [mine, setMine] = useState(() => new Set(loadMine()));
  const flow = flowOf(running, now, room.leaseMs);

  return (
    <div className="phone">
      <RoomHeader room={room} now={now} />
      <section className="phone__people">
        <FacePile people={people} me={me} />
        <button type="button" className="link" onClick={onRename}>
          you are {identity.name}
        </button>
      </section>
      <LayerRibbon machines={machines} colorOf={colorOf} flow={flow} liveTps={running?.liveTps ?? null} />
      <Composer
        code={room.code}
        identity={identity}
        open={room.open}
        onSent={(id) => {
          const next = new Set(mine).add(id);
          setMine(next);
          rememberMine([...next]);
        }}
      />
      <section className="panel panel--queue">
        <h2 className="panel__title">Shared queue</h2>
        <Queue prompts={prompts} mine={mine} now={now} leaseMs={room.leaseMs} />
      </section>
      <section className="panel">
        <h2 className="panel__title">Who holds which layers</h2>
        <MachineList machines={machines} colorOf={colorOf} compact />
      </section>
      <footer className="phone__foot">
        Live state on Convex. The host sits behind NAT with no open port; it claims prompts over outbound HTTPS.
      </footer>
    </div>
  );
}
