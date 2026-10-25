import type { FunctionReturnType } from "convex/server";
import type { api } from "../../convex/_generated/api";

export type Room = NonNullable<FunctionReturnType<typeof api.rooms.get>>;
export type Machine = FunctionReturnType<typeof api.machines.list>[number];
export type Prompt = FunctionReturnType<typeof api.prompts.list>[number];
export type RoomEvent = FunctionReturnType<typeof api.events.list>[number];

export const BRIDGE_ONLINE_MS = 30_000;
export const PROJECTOR_ID = "projector#0000";

export type HostState = "ready" | "warming" | "offline" | "no-model";

export function hostState(room: Room, now: number): HostState {
  if (now - room.bridgeSeenAt > BRIDGE_ONLINE_MS) return "offline";
  if (room.deploymentState === "ready") return "ready";
  if (room.deploymentState === "gateway offline" || room.deploymentState === "no model loaded") return "no-model";
  return "warming";
}

/**
 * A running prompt's lease, read from the client: every append renews it to now + leaseMs, so the time since the last
 * renewal says whether the host is still talking. Tokens renew it several times a second and a long prefill sends a
 * keepalive every 20 s, so silence past those marks means the host went quiet.
 */
export function leaseView(p: Prompt, now: number, leaseMs: number): { left: number; quiet: boolean } {
  if (p.status !== "running" || p.leaseUntil === null) return { left: 0, quiet: false };
  const left = Math.max(0, p.leaseUntil - now);
  const sinceRenewal = leaseMs - left;
  const keepalive = Math.min(20_000, leaseMs / 3); // the bridge's keepalive during a prefill (runner.py)
  return { left, quiet: sinceRenewal > (p.firstTokenAt === null ? keepalive + 6_000 : 6_000) };
}

export function flowOf(p: Prompt | undefined, now: number, leaseMs: number): "decode" | "prefill" | null {
  if (p === undefined || leaseView(p, now, leaseMs).quiet) return null;
  return p.firstTokenAt === null ? "prefill" : "decode";
}

export const HOST_LABEL: Record<HostState, string> = {
  ready: "host online",
  warming: "model loading",
  offline: "host offline",
  "no-model": "no model loaded",
};
