import { v } from "convex/values";
import { query } from "./_generated/server";
import { roomByCode } from "./rooms";

/** The newest lines of the room's activity log, oldest first, for the projector. */
export const list = query({
  args: { code: v.string() },
  handler: async (ctx, { code }) => {
    const room = await roomByCode(ctx, code);
    if (room === null) return [];
    const rows = await ctx.db
      .query("events")
      .withIndex("by_room", (q) => q.eq("roomId", room._id))
      .order("desc")
      .take(10);
    return rows.reverse().map((e) => ({ id: e._id, kind: e.kind, text: e.text, at: e.at }));
  },
});
