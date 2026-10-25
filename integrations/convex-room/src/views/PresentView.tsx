import usePresence from "@convex-dev/presence/react";
import QRCode from "qrcode";
import { useEffect, useState } from "react";
import { api } from "../../convex/_generated/api";
import { ActivityLog } from "../components/ActivityLog";
import { FacePile } from "../components/FacePile";
import { LayerRibbon } from "../components/LayerRibbon";
import { MachineList } from "../components/MachineList";
import { PromptCard } from "../components/PromptCard";
import { RoomHeader } from "../components/RoomHeader";
import { useServerNow } from "../lib/clock";
import { PROJECTOR_ID, type Room, flowOf } from "../lib/types";
import { roomUrl, useRoom } from "../lib/useRoom";

/** The projector: a big QR code, who is in the room, the layer ribbon, the answer streaming in, and Convex's log. */
export function PresentView({ room }: { room: Room }) {
  const { machines, prompts, colorOf, running } = useRoom(room.code);
  const people = usePresence(api.presence, room.code, PROJECTOR_ID);
  const now = useServerNow(room.bridgeSeenAt);
  const url = roomUrl(room.code);
  const qr = useQr(url);
  const flow = flowOf(running, now, room.leaseMs);
  const queued = prompts.filter((p) => p.status === "queued");
  const finished = prompts.filter((p) => p.status === "done" || p.status === "failed");
  const spotlight = running ?? finished[0];
  const unreachable = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])([:/]|$)/.test(url);

  return (
    <div className="present">
      <aside className="present__side">
        <div className="qr">{qr ? <img src={qr} alt={`QR code for ${url}`} /> : <div className="qr__wait" />}</div>
        <p className="present__scan">Scan to join</p>
        <p className="present__url mono">{url.replace(/^https?:\/\//, "")}</p>
        {unreachable && (
          <p className="present__warn">
            This code points at localhost, which phones cannot reach. Set VITE_PUBLIC_URL, or open this page by the
            laptop's LAN address with npm run dev:lan.
          </p>
        )}
        <FacePile people={people} big max={24} />
      </aside>
      <main className="present__main">
        <RoomHeader room={room} now={now} big />
        <LayerRibbon machines={machines} colorOf={colorOf} flow={flow} liveTps={running?.liveTps ?? null} big />
        <MachineList machines={machines} colorOf={colorOf} />
        <section className="present__stage">
          {spotlight ? (
            <PromptCard p={spotlight} mine={false} now={now} leaseMs={room.leaseMs} big />
          ) : (
            <div className="present__idle">
              <p>Nobody has asked anything yet.</p>
              <p className="present__idle-sub">Scan the code, pick a name, and ask the pool a question.</p>
            </div>
          )}
          <div className="present__rail">
            {queued.length > 0 && (
              <ol className="present__queue" aria-label="Up next">
                {queued.slice(0, 4).map((p, i) => (
                  <li key={p.id}>
                    <span className="mono">#{i + 1}</span>
                    <strong>{p.author}</strong>
                    <span className="present__qtext">{p.text}</span>
                    {p.note && <em>{p.note}</em>}
                  </li>
                ))}
              </ol>
            )}
            <ActivityLog code={room.code} now={now} leaseMs={room.leaseMs} />
          </div>
        </section>
      </main>
    </div>
  );
}

function useQr(url: string): string | null {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    QRCode.toDataURL(url, { margin: 1, width: 720, errorCorrectionLevel: "M", color: { dark: "#0d0e0c", light: "#ecebe4" } })
      .then((data) => live && setSrc(data))
      .catch(() => live && setSrc(null));
    return () => {
      live = false;
    };
  }, [url]);
  return src;
}
