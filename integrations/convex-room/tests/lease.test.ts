import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { api, internal } from "../convex/_generated/api";
import type { Id } from "../convex/_generated/dataModel";
import { DEFAULT_LEASE_MS } from "../convex/lib/limits";
import { CODE, newRoom, openRoom, session, type T } from "./setup";

async function queue(t: T, text: string): Promise<Id<"prompts">> {
  const { id } = await t.mutation(api.prompts.submit, { code: CODE, author: "Ada", sessionId: session(), text });
  return id;
}

async function claim(t: T, bridgeId = "bridge-a") {
  const { prompt } = await t.mutation(internal.bridge.claim, { code: CODE, bridgeId });
  return prompt;
}

async function row(t: T, id: Id<"prompts">) {
  return await t.run(async (ctx) => (await ctx.db.get(id))!);
}

/** Moves the clock and lets every scheduled function that came due finish. */
async function advance(t: T, ms: number) {
  vi.advanceTimersByTime(ms);
  await t.finishInProgressScheduledFunctions();
}

describe("claim", () => {
  let t: T;
  beforeEach(async () => {
    vi.useFakeTimers();
    t = newRoom();
    await openRoom(t);
  });
  afterEach(() => vi.useRealTimers());

  test("takes the oldest queued prompt and leases it for 90 s", async () => {
    const first = await queue(t, "first");
    vi.advanceTimersByTime(10);
    await queue(t, "second");
    const now = Date.now();
    const got = await claim(t);
    expect(got).toMatchObject({ id: first, text: "first", claimSeq: 1, attempt: 1, leaseMs: DEFAULT_LEASE_MS });
    const r = await row(t, first);
    expect(r.status).toBe("running");
    expect(r.leaseUntil).toBe(now + DEFAULT_LEASE_MS);
    expect((await claim(t))!.text).toBe("second");
    expect(await claim(t)).toBeNull();
  });

  test("returns nothing for an unknown room", async () => {
    const { prompt, reason } = await t.mutation(internal.bridge.claim, { code: "nosuch", bridgeId: "b" });
    expect(prompt).toBeNull();
    expect(reason).toBe("no_room");
  });
});

describe("lease expiry and requeue", () => {
  let t: T;
  beforeEach(async () => {
    vi.useFakeTimers();
    t = newRoom();
    await openRoom(t);
  });
  afterEach(() => vi.useRealTimers());

  test("a silent host's prompt is requeued once, then marked failed", async () => {
    const id = await queue(t, "will the host die?");
    await claim(t);
    await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 1, delta: "Half an ans" });

    await advance(t, DEFAULT_LEASE_MS + 1);
    let r = await row(t, id);
    expect(r.status).toBe("queued");
    expect(r.requeues).toBe(1);
    expect(r.answer).toBe("");
    expect(r.note).toMatch(/lease expired; requeued/);

    const again = await claim(t, "bridge-b");
    expect(again).toMatchObject({ id, attempt: 2, claimSeq: 2 });
    expect((await row(t, id)).note).toBe("retry, attempt 2 of 2");

    await advance(t, DEFAULT_LEASE_MS + 1);
    r = await row(t, id);
    expect(r.status).toBe("failed");
    expect(r.error).toMatch(/2 times; giving up/);
  });

  test("appends extend the lease, and the scheduled check re-arms instead of expiring", async () => {
    const id = await queue(t, "slow prefill");
    await claim(t);
    await advance(t, 60_000);
    await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 1, delta: "" }); // keepalive

    await advance(t, 31_000); // past the first lease end: re-armed, still running
    expect((await row(t, id)).status).toBe("running");

    await advance(t, 60_000); // past the extended end with no word from the host
    expect((await row(t, id)).status).toBe("queued");
  });

  test("a bridge from an older claim cannot write over the new attempt", async () => {
    const id = await queue(t, "fencing");
    await claim(t, "old");
    await advance(t, DEFAULT_LEASE_MS + 1);
    await claim(t, "new");

    const stale = await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 1, delta: "zombie" });
    expect(stale).toEqual({ ok: false, reason: "stale" });
    const staleFinish = await t.mutation(internal.bridge.finish, {
      promptId: id, claimSeq: 1, tail: "", tokens: 1, promptTokens: 1, decodeTps: 1, prefillTps: 1, finishReason: "stop",
    });
    expect(staleFinish).toEqual({ ok: false, reason: "stale" });

    expect((await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 2, delta: "fresh" })).ok).toBe(true);
    const r = await row(t, id);
    expect(r.answer).toBe("fresh");
    expect(r.bridgeId).toBe("new");
  });

  test("a busy engine's prompt goes back without spending an attempt", async () => {
    const id = await queue(t, "busy");
    await claim(t);
    await t.mutation(internal.bridge.release, { promptId: id, claimSeq: 1, reason: "busy" });
    let r = await row(t, id);
    expect(r).toMatchObject({ status: "queued", attempts: 0, requeues: 0 });
    for (let i = 0; i < 3; i++) {
      const p = await claim(t);
      await t.mutation(internal.bridge.release, { promptId: id, claimSeq: p!.claimSeq, reason: "busy" });
    }
    r = await row(t, id);
    expect(r).toMatchObject({ status: "queued", attempts: 0, claimSeq: 4 });
  });

  test("a dropped gateway connection counts as an attempt", async () => {
    const id = await queue(t, "gateway dies");
    await claim(t);
    await t.mutation(internal.bridge.release, { promptId: id, claimSeq: 1, reason: "gateway_lost" });
    expect((await row(t, id)).status).toBe("queued");
    await claim(t);
    await t.mutation(internal.bridge.release, { promptId: id, claimSeq: 2, reason: "gateway_lost" });
    expect((await row(t, id)).status).toBe("failed");
  });

  test("finish records the engine's timings and the room's last rate", async () => {
    const id = await queue(t, "finish me");
    await claim(t);
    await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 1, delta: "Hello", liveTps: 1.7 });
    expect((await row(t, id)).liveTps).toBe(1.7);
    await t.mutation(internal.bridge.finish, {
      promptId: id, claimSeq: 1, tail: " world", tokens: 2, promptTokens: 14,
      decodeTps: 1.81, prefillTps: 3.3, finishReason: "length",
    });
    const r = await row(t, id);
    expect(r).toMatchObject({ status: "done", answer: "Hello world", tokens: 2, decodeTps: 1.81, liveTps: null });
    const room = await t.query(api.rooms.get, { code: CODE });
    expect(room).toMatchObject({ lastDecodeTps: 1.81, lastPrefillTps: 3.3, servedCount: 1 });

    await advance(t, DEFAULT_LEASE_MS + 1); // the armed lease check finds a finished prompt and does nothing
    expect((await row(t, id)).status).toBe("done");
  });

  test("answers are capped and nonsense rates are dropped", async () => {
    const id = await queue(t, "long");
    await claim(t);
    for (let i = 0; i < 5; i++) {
      await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 1, delta: "y".repeat(5000), liveTps: -4 });
    }
    const r = await row(t, id);
    expect(r.answer.length).toBe(6000);
    expect(r.liveTps).toBeNull();
  });
});
