import { v } from "convex/values";
import { query } from "./_generated/server";
import { roomByCode } from "./rooms";

/** Which machine holds which layers of the served model, and how each one is linked to the host. */
export const list = query({
  args: { code: v.string() },
  handler: async (ctx, { code }) => {
    const room = await roomByCode(ctx, code);
    if (room === null) return [];
    const rows = await ctx.db
      .query("machines")
      .withIndex("by_room", (q) => q.eq("roomId", room._id))
      .collect();
    return rows
      .sort((a, b) => a.order - b.order)
      .map((m) => ({
        id: m._id,
        nodeId: m.nodeId,
        name: m.name,
        device: m.device,
        layers: m.layers,
        layerRange: m.layerRange,
        weightBytes: m.weightBytes,
        holdsOutput: m.holdsOutput,
        path: m.path,
        state: m.state,
        os: m.os,
      }));
  },
});
