import type { Id } from "../_generated/dataModel";
import type { MutationCtx } from "../_generated/server";

export type EventKind =
  | "submit"
  | "claim"
  | "done"
  | "busy"
  | "requeue"
  | "lease"
  | "failed"
  | "fence"
  | "host"
  | "admin";

const KEEP = 60;
const REPEAT_MS = 60_000;

/**
 * Appends one line to the room's activity log, in the same transaction as the change it describes, so the log never
 * shows something that rolled back. A line that repeats one of the newest three within a minute is skipped (a busy
 * engine retried every 2 s would otherwise fill the log with claim/busy pairs), and the log keeps its newest 60 lines.
 */
export async function logEvent(ctx: MutationCtx, roomId: Id<"rooms">, kind: EventKind, text: string): Promise<void> {
  const recent = await ctx.db
    .query("events")
    .withIndex("by_room", (q) => q.eq("roomId", roomId))
    .order("desc")
    .take(KEEP + 5);
  const now = Date.now();
  const line = text.slice(0, 200);
  const repeat = recent.slice(0, 3).some((e) => e.kind === kind && e.text === line && now - e.at < REPEAT_MS);
  if (repeat) return;
  await ctx.db.insert("events", { roomId, kind, text: line, at: now });
  for (const old of recent.slice(KEEP - 1)) await ctx.db.delete(old._id);
}
