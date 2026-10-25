import { v } from "convex/values";
import type { Doc, Id } from "./_generated/dataModel";
import { type DatabaseReader, mutation, query } from "./_generated/server";
import { logEvent } from "./lib/events";
import { QUEUE_CAP, RECENT_FINISHED, checkPrompt, ownerKeyOf, refuse } from "./lib/limits";
import { rateLimiter } from "./rateLimits";
import { roomByCode } from "./rooms";

/**
 * A phone sends a prompt. Order matters: cheap checks first, then the queue cap, then the two token buckets. A
 * refusal throws, so the whole mutation rolls back and a refused prompt spends no rate-limit tokens.
 */
export const submit = mutation({
  args: { code: v.string(), author: v.string(), sessionId: v.string(), text: v.string() },
  handler: async (ctx, args) => {
    const room = await roomByCode(ctx, args.code);
    if (room === null) refuse("bad_room", "no such room; check the link");
    if (!room.open) refuse("room_closed", "the host closed this room");
    const { text, author } = checkPrompt(args.text, args.author, args.sessionId);

    const active = await activeCount(ctx, room._id);
    if (active >= QUEUE_CAP) refuse("queue_full", `the queue is full (${QUEUE_CAP}); try again in a minute`);

    await rateLimiter.limit(ctx, "promptSession", { key: args.sessionId, throws: true });
    await rateLimiter.limit(ctx, "promptRoom", { key: room.code, throws: true });

    const now = Date.now();
    const id = await ctx.db.insert("prompts", {
      roomId: room._id,
      author,
      sessionId: args.sessionId,
      ownerKey: await ownerKeyOf(args.sessionId),
      text,
      status: "queued",
      answer: "",
      reasoning: "",
      tokens: null,
      promptTokens: null,
      decodeTps: null,
      prefillTps: null,
      liveTps: null,
      finishReason: null,
      leaseUntil: null,
      attempts: 0,
      claimSeq: 0,
      requeues: 0,
      bridgeId: null,
      note: null,
      error: null,
      createdAt: now,
      startedAt: null,
      firstTokenAt: null,
      finishedAt: null,
    });
    await logEvent(ctx, room._id, "submit", `${author} asked: ${text.length > 60 ? `${text.slice(0, 60)}...` : text}`);
    return { id, position: active + 1 };
  },
});

export async function activeCount(ctx: { db: DatabaseReader }, roomId: Id<"rooms">): Promise<number> {
  let n = 0;
  for (const status of ["queued", "running"] as const) {
    const rows = await ctx.db
      .query("prompts")
      .withIndex("by_room_status", (q) => q.eq("roomId", roomId).eq("status", status))
      .take(QUEUE_CAP + 1);
    n += rows.length;
  }
  return n;
}

/**
 * The shared queue every screen subscribes to: everything queued or running, oldest first, then the most recent
 * finished prompts. One query for every viewer (no per-user arguments), so Convex serves all of them from one cache
 * entry and pushes each answer patch to every phone at once.
 */
export const list = query({
  args: { code: v.string() },
  handler: async (ctx, { code }) => {
    const room = await roomByCode(ctx, code);
    if (room === null) return [];
    const byStatus = async (status: Doc<"prompts">["status"], order: "asc" | "desc", n: number) =>
      await ctx.db
        .query("prompts")
        .withIndex("by_room_status", (q) => q.eq("roomId", room._id).eq("status", status))
        .order(order)
        .take(n);
    const [running, queued, done, failed] = await Promise.all([
      byStatus("running", "asc", QUEUE_CAP),
      byStatus("queued", "asc", QUEUE_CAP),
      byStatus("done", "desc", RECENT_FINISHED),
      byStatus("failed", "desc", RECENT_FINISHED),
    ]);
    const finished = [...done, ...failed]
      .sort((a, b) => (b.finishedAt ?? b.createdAt) - (a.finishedAt ?? a.createdAt))
      .slice(0, RECENT_FINISHED);
    return [...running, ...queued, ...finished].map(publicPrompt);
  },
});

function publicPrompt(p: Doc<"prompts">) {
  return {
    id: p._id,
    author: p.author,
    ownerKey: p.ownerKey,
    text: p.text,
    status: p.status,
    answer: p.answer,
    reasoning: p.reasoning,
    tokens: p.tokens,
    promptTokens: p.promptTokens,
    decodeTps: p.decodeTps,
    prefillTps: p.prefillTps,
    liveTps: p.liveTps,
    finishReason: p.finishReason,
    leaseUntil: p.leaseUntil,
    attempts: p.attempts,
    requeues: p.requeues,
    note: p.note,
    error: p.error,
    createdAt: p.createdAt,
    startedAt: p.startedAt,
    firstTokenAt: p.firstTokenAt,
    finishedAt: p.finishedAt,
  };
}
