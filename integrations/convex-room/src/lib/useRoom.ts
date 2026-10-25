import { useQuery } from "convex/react";
import { useMemo } from "react";
import { api } from "../../convex/_generated/api";
import { machineColor } from "./format";
import type { Machine, Prompt, Room } from "./types";

export interface RoomData {
  room: Room | null | undefined; // undefined while loading, null for an unknown code
  machines: Machine[];
  prompts: Prompt[];
  colorOf: (id: string) => string;
  running: Prompt | undefined;
}

/** The three live queries every view subscribes to. Convex pushes each change to every open screen at once. */
export function useRoom(code: string): RoomData {
  const room = useQuery(api.rooms.get, { code });
  const machines = useQuery(api.machines.list, { code }) ?? [];
  const prompts = useQuery(api.prompts.list, { code }) ?? [];
  const colors = useMemo(() => new Map(machines.map((m, i) => [m.id as string, machineColor(i)])), [machines]);
  return {
    room,
    machines,
    prompts,
    colorOf: (id: string) => colors.get(id) ?? "#7c7b73",
    running: prompts.find((p) => p.status === "running"),
  };
}

export function roomUrl(code: string): string {
  const base = (import.meta.env.VITE_PUBLIC_URL as string | undefined) || `${location.origin}${location.pathname}`;
  return `${base.replace(/\/$/, "")}/?room=${encodeURIComponent(code)}`;
}
