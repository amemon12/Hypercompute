import { useQuery } from "convex/react";
import { useState } from "react";
import { api } from "../convex/_generated/api";
import { JoinGate } from "./components/JoinGate";
import { createIdentity, type Identity, loadIdentity } from "./lib/identity";
import { PhoneView } from "./views/PhoneView";
import { PresentView } from "./views/PresentView";

const CODE = /^[a-z0-9][a-z0-9-]{2,31}$/;

function params(): { code: string | null; present: boolean } {
  const q = new URLSearchParams(location.search);
  const raw = (q.get("room") ?? (import.meta.env.VITE_DEFAULT_ROOM as string | undefined) ?? "").trim().toLowerCase();
  return { code: CODE.test(raw) ? raw : null, present: q.get("view") === "present" };
}

export default function App() {
  const [{ code, present }] = useState(params);
  if (code === null) {
    return (
      <Notice title="Which room?">
        Open the link from the projector, or scan its QR code. Links look like <code>?room=calhacks</code>.
      </Notice>
    );
  }
  return <RoomLoader code={code} present={present} />;
}

function RoomLoader({ code, present }: { code: string; present: boolean }) {
  const room = useQuery(api.rooms.get, { code });
  const [identity, setIdentity] = useState<Identity | null>(loadIdentity);
  const [renaming, setRenaming] = useState(false);

  if (room === undefined) return <div className="loading" aria-label="Loading the room" />;
  if (room === null) {
    return (
      <Notice title={`No room called “${code}”`}>
        The host has not opened it yet, or the link has a typo. Ask whoever showed you the code.
      </Notice>
    );
  }
  if (present) return <PresentView room={room} />;
  if (identity === null || renaming) {
    return (
      <JoinGate
        room={room}
        initialName={identity?.name ?? ""}
        onJoin={(name) => {
          setIdentity(createIdentity(name, identity));
          setRenaming(false);
        }}
      />
    );
  }
  return <PhoneView room={room} identity={identity} onRename={() => setRenaming(true)} />;
}

function Notice({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <main className="notice">
      <h1>{title}</h1>
      <p>{children}</p>
    </main>
  );
}
