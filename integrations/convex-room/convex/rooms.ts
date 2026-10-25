import { v } from "convex/values";
import type { Doc } from "./_generated/dataModel";
import { query, type QueryCtx } from "./_generated/server";
import { ROOM_CODE, leaseMs } from "./lib/limits";

export async function roomByCode(ctx: QueryCtx, code: string): Promise<Doc<"rooms"> | null> {
  if (!ROOM_CODE.test(code)) return null;
  return await ctx.db
    .query("rooms")
    .withIndex("by_code", (q) => q.eq("code", code))
    .unique();
}

/** The room header: model, deployment state and the bridge's last check-in. Null for an unknown code. */
export const get = query({
  args: { code: v.string() },
  handler: async (ctx, { code }) => {
    const room = await roomByCode(ctx, code);
    if (room === null) return null;
    return {
      code: room.code,
      title: room.title,
      open: room.open,
      modelId: room.modelId,
      deploymentState: room.deploymentState,
      tier: room.tier,
      offloaded: room.offloaded,
      slots: room.slots,
      bridgeSeenAt: room.bridgeSeenAt,
      lastDecodeTps: room.lastDecodeTps,
      lastPrefillTps: room.lastPrefillTps,
      servedCount: room.servedCount,
      leaseMs: leaseMs(),
    };
  },
});
