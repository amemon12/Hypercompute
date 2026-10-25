/**
 * What the host bridge does, as internal mutations behind the secret-checked HTTP routes in http.ts.
 *
 * A claim is one transaction: the oldest queued prompt becomes running with a lease, and a scheduled expireLease
 * call is armed for when the lease ends. Every append or finish from the bridge extends the lease. If the host goes
 * quiet (bridge killed, laptop asleep, Wi-Fi gone), expireLease finds the lease in the past and requeues the prompt
 * once; a second expiry marks it failed. claimSeq is a fencing token: it bumps on every claim, and a write that
 * carries an older claimSeq is refused, so a bridge that comes back after its prompt was requeued cannot overwrite
 * the new attempt.
 */
import { ConvexError, v } from "convex/values";
import { internal } from "./_generated/api";
import type { Doc, Id } from "./_generated/dataModel";
import { internalMutation, type MutationCtx } from "./_generated/server";
import { type EventKind, logEvent } from "./lib/events";
import { BRIDGE_ONLINE_MS, MAX_ANSWER_CHARS, MAX_ATTEMPTS, MAX_DELTA_CHARS, ROOM_CODE, leaseMs } from "./lib/limits";
import { linkPath } from "./schema";
import { roomByCode } from "./rooms";

type Stale = { ok: false; reason: "stale" };
const STALE: Stale = { ok: false, reason: "stale" };
type Repeat = { ok: true; repeat: true };
const REPEAT: Repeat = { ok: true, repeat: true };

async function heldPrompt(ctx: MutationCtx, promptId: Id<"prompts">, claimSeq: number): Promise<Doc<"prompts"> | null> {
  const prompt = await ctx.db.get(promptId);
  if (prompt === null || prompt.status !== "running" || prompt.claimSeq !== claimSeq) return null;
  return prompt;
}

/** A write from an older claim: refuse it, and say so in the room's log (the fencing is the point of claimSeq). */
async function refuseStale(ctx: MutationCtx, promptId: Id<"prompts">, claimSeq: number,
                           settled: Array<Doc<"prompts">["status"]> = []): Promise<Stale | Repeat> {
  const prompt = await ctx.db.get(promptId);
  if (prompt !== null && claimSeq === prompt.claimSeq && settled.includes(prompt.status)) {
    return REPEAT; // a retried finish, release or fail whose first try landed and whose reply was lost
  }
  if (prompt !== null && claimSeq < prompt.claimSeq) {
    await logEvent(ctx, prompt.roomId, "fence",
      `Refused a stale write for ${prompt.author}'s prompt: claim #${claimSeq} was superseded by #${prompt.claimSeq}`);
  } else if (prompt !== null && claimSeq === prompt.claimSeq && prompt.status !== "running") {
    await logEvent(ctx, prompt.roomId, "fence",
      `Refused a late write for ${prompt.author}'s prompt: claim #${claimSeq} had lost its lease`);
  }
  return STALE;
}

function rate(x: number | null): string {
  return x === null ? "n/a" : x >= 10 ? x.toFixed(1) : x.toFixed(2);
}

function grow(answer: string, delta: string): string {
  if (answer.length >= MAX_ANSWER_CHARS) return answer;
  return (answer + delta.slice(0, MAX_DELTA_CHARS)).slice(0, MAX_ANSWER_CHARS);
}

/**
 * Places `delta` at `offset`, the number of characters the bridge had sent before it. A retried append whose first
 * try landed writes the same characters over themselves, so a lost reply never doubles the text. An offset past the
 * end (the answer hit its cap) adds nothing.
 */
function placeAt(current: string, offset: number | undefined, delta: string): string {
  if (!delta) return current;
  if (offset === undefined) return grow(current, delta);
  if (!Number.isInteger(offset) || offset < 0 || offset > current.length) return current;
  return grow(current.slice(0, offset), delta);
}

function sane(rate: number | null | undefined): number | null {
  return typeof rate === "number" && Number.isFinite(rate) && rate > 0 && rate < 100_000 ? rate : null;
}

export const claim = internalMutation({
  args: { code: v.string(), bridgeId: v.string(), claimKey: v.optional(v.string()) },
  handler: async (ctx, { code, bridgeId, claimKey }) => {
    const room = await roomByCode(ctx, code);
    if (room === null) return { prompt: null, reason: "no_room" as const };
    if (claimKey) {
      // A retried claim whose first try landed: hand back the same prompt instead of taking a second one.
      const running = await ctx.db
        .query("prompts")
        .withIndex("by_room_status", (q) => q.eq("roomId", room._id).eq("status", "running"))
        .take(32);
      const mine = running.find((p) => p.claimKey === claimKey);
      if (mine !== undefined) {
        return {
          prompt: { id: mine._id, text: mine.text, author: mine.author, claimSeq: mine.claimSeq,
                    attempt: mine.attempts, leaseMs: leaseMs() },
          reason: null,
        };
      }
    }
    const next = await ctx.db
      .query("prompts")
      .withIndex("by_room_status", (q) => q.eq("roomId", room._id).eq("status", "queued"))
      .order("asc")
      .first();
    if (next === null) return { prompt: null, reason: "empty" as const };

    const now = Date.now();
    const lease = leaseMs();
    const attempt = next.attempts + 1;
    const claimSeq = next.claimSeq + 1;
    await ctx.db.patch(next._id, {
      status: "running",
      attempts: attempt,
      claimSeq,
      leaseUntil: now + lease,
      bridgeId,
      claimKey: claimKey?.slice(0, 64),
      startedAt: now,
      firstTokenAt: null,
      answer: "",
      reasoning: "",
      liveTps: null,
      note: next.requeues > 0 ? `retry, attempt ${attempt} of ${MAX_ATTEMPTS}` : null,
    });
    await ctx.scheduler.runAfter(lease, internal.bridge.expireLease, { promptId: next._id, claimSeq });
    await logEvent(ctx, room._id, "claim",
      `Host claimed ${next.author}'s prompt${attempt > 1 ? ` (attempt ${attempt} of ${MAX_ATTEMPTS})` : ""}; ` +
        `lease ${Math.round(lease / 1000)} s`);
    return {
      prompt: { id: next._id, text: next.text, author: next.author, claimSeq, attempt, leaseMs: lease },
      reason: null,
    };
  },
});

export const append = internalMutation({
  args: {
    promptId: v.id("prompts"),
    claimSeq: v.number(),
    delta: v.string(),
    offset: v.optional(v.number()),
    reasoning: v.optional(v.string()),
    reasoningOffset: v.optional(v.number()),
    liveTps: v.optional(v.union(v.number(), v.null())),
  },
  handler: async (ctx, { promptId, claimSeq, delta, offset, reasoning, reasoningOffset, liveTps }) => {
    const prompt = await heldPrompt(ctx, promptId, claimSeq);
    if (prompt === null) return await refuseStale(ctx, promptId, claimSeq);
    const now = Date.now();
    const leaseUntil = now + leaseMs();
    const live = sane(liveTps);
    const text = delta || reasoning;
    await ctx.db.patch(promptId, {
      leaseUntil,
      ...(delta ? { answer: placeAt(prompt.answer, offset, delta) } : {}),
      ...(reasoning ? { reasoning: placeAt(prompt.reasoning, reasoningOffset, reasoning) } : {}),
      ...(text && prompt.firstTokenAt === null ? { firstTokenAt: now } : {}),
      ...(live !== null ? { liveTps: live } : {}),
    });
    return { ok: true as const, leaseUntil };
  },
});

export const finish = internalMutation({
  args: {
    promptId: v.id("prompts"),
    claimSeq: v.number(),
    tail: v.string(),
    offset: v.optional(v.number()),
    reasoningTail: v.optional(v.string()),
    reasoningOffset: v.optional(v.number()),
    tokens: v.union(v.number(), v.null()),
    promptTokens: v.union(v.number(), v.null()),
    decodeTps: v.union(v.number(), v.null()),
    prefillTps: v.union(v.number(), v.null()),
    finishReason: v.union(v.string(), v.null()),
  },
  handler: async (ctx, args) => {
    const prompt = await heldPrompt(ctx, args.promptId, args.claimSeq);
    if (prompt === null) return await refuseStale(ctx, args.promptId, args.claimSeq, ["done"]);
    const now = Date.now();
    const decodeTps = sane(args.decodeTps);
    const prefillTps = sane(args.prefillTps);
    await ctx.db.patch(args.promptId, {
      status: "done",
      answer: placeAt(prompt.answer, args.offset, args.tail),
      reasoning: placeAt(prompt.reasoning, args.reasoningOffset, args.reasoningTail ?? ""),
      firstTokenAt: prompt.firstTokenAt ?? (args.tail || args.reasoningTail ? now : null),
      tokens: args.tokens,
      promptTokens: args.promptTokens,
      decodeTps,
      prefillTps,
      liveTps: null,
      finishReason: args.finishReason?.slice(0, 32) ?? null,
      leaseUntil: null,
      finishedAt: now,
    });
    const room = await ctx.db.get(prompt.roomId);
    if (room !== null) {
      await ctx.db.patch(room._id, {
        servedCount: room.servedCount + 1,
        ...(decodeTps !== null ? { lastDecodeTps: decodeTps } : {}),
        ...(prefillTps !== null ? { lastPrefillTps: prefillTps } : {}),
      });
    }
    const took = ((now - (prompt.startedAt ?? now)) / 1000).toFixed(1);
    await logEvent(ctx, prompt.roomId, "done",
      `Answered ${prompt.author}: ${args.tokens ?? "?"} tokens in ${took} s, ${rate(decodeTps)} tok/s decode`);
    return { ok: true as const };
  },
});

/** Requeue, or fail once the prompt used its attempts. Shared by release (gateway_lost) and lease expiry. */
async function requeueOrFail(ctx: MutationCtx, prompt: Doc<"prompts">, why: string,
                             kind: EventKind): Promise<"queued" | "failed"> {
  const now = Date.now();
  if (prompt.attempts >= MAX_ATTEMPTS) {
    await logEvent(ctx, prompt.roomId, "failed", `${capital(why)} again; gave up on ${prompt.author}'s prompt`);
    await ctx.db.patch(prompt._id, {
      status: "failed",
      error: `${why}, ${prompt.attempts} times; giving up`,
      note: why,
      leaseUntil: null,
      liveTps: null,
      finishedAt: now,
    });
    return "failed";
  }
  await ctx.db.patch(prompt._id, {
    status: "queued",
    answer: "",
    reasoning: "",
    leaseUntil: null,
    liveTps: null,
    startedAt: null,
    firstTokenAt: null,
    requeues: prompt.requeues + 1,
    note: `${why}; requeued`,
  });
  await logEvent(ctx, prompt.roomId, kind, `${capital(why)}; requeued ${prompt.author}'s prompt`);
  return "queued";
}

function capital(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export const release = internalMutation({
  args: {
    promptId: v.id("prompts"),
    claimSeq: v.number(),
    reason: v.union(v.literal("busy"), v.literal("shutdown"), v.literal("gateway_lost")),
  },
  handler: async (ctx, { promptId, claimSeq, reason }) => {
    const prompt = await heldPrompt(ctx, promptId, claimSeq);
    if (prompt === null) return await refuseStale(ctx, promptId, claimSeq, ["queued", "failed"]);
    if (reason === "gateway_lost") {
      return { ok: true as const, status: await requeueOrFail(ctx, prompt, "the gateway connection dropped", "requeue") };
    }
    // The engine refused before any output (busy), or the bridge is stopping: this attempt does not count.
    await ctx.db.patch(promptId, {
      status: "queued",
      attempts: Math.max(0, prompt.attempts - 1),
      answer: "",
      reasoning: "",
      leaseUntil: null,
      liveTps: null,
      startedAt: null,
      firstTokenAt: null,
      note: reason === "busy" ? "the engine was busy; back in the queue" : "the host bridge restarted; back in the queue",
    });
    await logEvent(ctx, prompt.roomId, reason === "busy" ? "busy" : "requeue",
      reason === "busy"
        ? `Engine busy; ${prompt.author}'s prompt went back in the queue`
        : `Host bridge stopped; ${prompt.author}'s prompt went back in the queue`);
    return { ok: true as const, status: "queued" as const };
  },
});

export const fail = internalMutation({
  args: { promptId: v.id("prompts"), claimSeq: v.number(), error: v.string() },
  handler: async (ctx, { promptId, claimSeq, error }) => {
    const prompt = await heldPrompt(ctx, promptId, claimSeq);
    if (prompt === null) return await refuseStale(ctx, promptId, claimSeq, ["failed"]);
    await ctx.db.patch(promptId, {
      status: "failed",
      error: error.slice(0, 300),
      leaseUntil: null,
      liveTps: null,
      finishedAt: Date.now(),
    });
    await logEvent(ctx, prompt.roomId, "failed", `${prompt.author}'s prompt failed: ${error.slice(0, 120)}`);
    return { ok: true as const };
  },
});

/** Scheduled at claim time. A lease that was extended re-arms itself for its new end instead of expiring. */
export const expireLease = internalMutation({
  args: { promptId: v.id("prompts"), claimSeq: v.number() },
  handler: async (ctx, { promptId, claimSeq }) => {
    const prompt = await heldPrompt(ctx, promptId, claimSeq);
    if (prompt === null) return { action: "none" as const };
    const now = Date.now();
    if (prompt.leaseUntil !== null && prompt.leaseUntil > now) {
      await ctx.scheduler.runAt(prompt.leaseUntil, internal.bridge.expireLease, { promptId, claimSeq });
      return { action: "rearmed" as const };
    }
    return { action: await requeueOrFail(ctx, prompt, "the host went quiet and its lease expired", "lease") };
  },
});

const machineArg = v.object({
  nodeId: v.string(),
  name: v.string(),
  device: v.string(),
  layers: v.number(),
  layerRange: v.union(v.string(), v.null()),
  weightBytes: v.number(),
  holdsOutput: v.boolean(),
  path: linkPath,
  state: v.string(),
  os: v.union(v.string(), v.null()),
});

const MAX_MACHINES = 32;

/** The bridge's pool snapshot. Creates the room on first contact; rewrites only the machine rows that changed. */
export const syncPool = internalMutation({
  args: {
    code: v.string(),
    bridgeId: v.string(),
    title: v.string(),
    modelId: v.union(v.string(), v.null()),
    deploymentState: v.string(),
    tier: v.union(v.string(), v.null()),
    offloaded: v.union(v.string(), v.null()),
    slots: v.number(),
    machines: v.array(machineArg),
  },
  handler: async (ctx, args) => {
    if (!ROOM_CODE.test(args.code)) throw new ConvexError("room code must be 3-32 chars of a-z, 0-9 and -");
    const now = Date.now();
    const header = {
      title: args.title.slice(0, 64),
      modelId: args.modelId?.slice(0, 128) ?? null,
      deploymentState: args.deploymentState.slice(0, 32),
      tier: args.tier?.slice(0, 32) ?? null,
      offloaded: args.offloaded?.slice(0, 32) ?? null,
      slots: Math.min(Math.max(Math.floor(args.slots), 1), 16),
      bridgeId: args.bridgeId.slice(0, 64),
      bridgeSeenAt: now,
      updatedAt: now,
    };
    let room = await roomByCode(ctx, args.code);
    if (room === null) {
      const id = await ctx.db.insert("rooms", {
        code: args.code,
        open: true,
        lastDecodeTps: null,
        lastPrefillTps: null,
        servedCount: 0,
        ...header,
      });
      room = (await ctx.db.get(id))!;
      await logEvent(ctx, id, "host", `Room opened by host bridge ${header.bridgeId}`);
    } else {
      const back = now - room.bridgeSeenAt > BRIDGE_ONLINE_MS || room.bridgeId !== header.bridgeId;
      const stateChanged = room.deploymentState !== header.deploymentState || room.modelId !== header.modelId;
      await ctx.db.patch(room._id, header);
      if (back) await logEvent(ctx, room._id, "host", `Host bridge ${header.bridgeId} connected`);
      if (stateChanged) {
        await logEvent(ctx, room._id, "host", `Model ${header.modelId ?? "none"}: ${header.deploymentState}`);
      }
    }
    await syncMachines(ctx, room._id, args.machines.slice(0, MAX_MACHINES));
    return { roomId: room._id, open: room.open };
  },
});

type MachineIn = typeof machineArg.type;

async function syncMachines(ctx: MutationCtx, roomId: Id<"rooms">, incoming: MachineIn[]): Promise<void> {
  const existing = await ctx.db
    .query("machines")
    .withIndex("by_room", (q) => q.eq("roomId", roomId))
    .collect();
  const key = (m: { nodeId: string; device: string }) => `${m.nodeId}\u0000${m.device}`;
  const byKey = new Map(existing.map((m) => [key(m), m]));
  const seen = new Set<string>();
  for (const [order, raw] of incoming.entries()) {
    const m = {
      ...raw,
      name: raw.name.slice(0, 64),
      device: raw.device.slice(0, 32),
      state: raw.state.slice(0, 32),
      layerRange: raw.layerRange?.slice(0, 64) ?? null,
      os: raw.os?.slice(0, 32) ?? null,
      order,
    };
    const k = key(m);
    if (seen.has(k)) continue;
    seen.add(k);
    const row = byKey.get(k);
    if (row === undefined) {
      await ctx.db.insert("machines", { roomId, ...m });
    } else if (changed(row, m)) {
      await ctx.db.patch(row._id, m);
    }
  }
  for (const row of existing) {
    if (!seen.has(key(row))) await ctx.db.delete(row._id);
  }
}

function changed(row: Doc<"machines">, next: Record<string, unknown>): boolean {
  return Object.entries(next).some(([k, val]) => (row as Record<string, unknown>)[k] !== val);
}

/** The owner's switches: open or close the room to new prompts, or clear everything that is not running. */
export const admin = internalMutation({
  args: { code: v.string(), action: v.union(v.literal("open"), v.literal("close"), v.literal("clear")) },
  handler: async (ctx, { code, action }) => {
    const room = await roomByCode(ctx, code);
    if (room === null) return { ok: false as const, reason: "no_room" };
    if (action === "clear") {
      let removed = 0;
      for (const status of ["queued", "done", "failed"] as const) {
        const rows = await ctx.db
          .query("prompts")
          .withIndex("by_room_status", (q) => q.eq("roomId", room._id).eq("status", status))
          .take(500);
        for (const row of rows) await ctx.db.delete(row._id);
        removed += rows.length;
      }
      const log = await ctx.db
        .query("events")
        .withIndex("by_room", (q) => q.eq("roomId", room._id))
        .take(500);
      for (const row of log) await ctx.db.delete(row._id);
      await logEvent(ctx, room._id, "admin", "Host cleared the room");
      return { ok: true as const, removed };
    }
    await ctx.db.patch(room._id, { open: action === "open" });
    await logEvent(ctx, room._id, "admin", action === "open" ? "Host opened the room" : "Host closed the room");
    return { ok: true as const, open: action === "open" };
  },
});
