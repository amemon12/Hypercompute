/** Lost replies and retries: every bridge write must be safe to send twice. */
import { ConvexError } from "convex/values";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { api, internal } from "../convex/_generated/api";
import type { Id } from "../convex/_generated/dataModel";
import { DEFAULT_LEASE_MS } from "../convex/lib/limits";
import { CODE, newRoom, openRoom, session, type T } from "./setup";

async function queue(t: T, text = "q"): Promise<Id<"prompts">> {
  return (await t.mutation(api.prompts.submit, { code: CODE, author: "Ada", sessionId: session(), text })).id;
}

const row = (t: T, id: Id<"prompts">) => t.run(async (ctx) => (await ctx.db.get(id))!);

describe("idempotent bridge writes", () => {
  let t: T;
  beforeEach(async () => {
    vi.useFakeTimers();
    t = newRoom();
    await openRoom(t);
  });
  afterEach(() => vi.useRealTimers());

  test("an append retried after a lost reply does not double the text", async () => {
    const id = await queue(t);
    await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b" });
    const hello = { promptId: id, claimSeq: 1, delta: "Hello", offset: 0 };
    await t.mutation(internal.bridge.append, hello);
    await t.mutation(internal.bridge.append, hello); // the retry
    await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 1, delta: " world", offset: 5 });
    await t.mutation(internal.bridge.finish, {
      promptId: id, claimSeq: 1, tail: "!", offset: 11, tokens: 3, promptTokens: 1, decodeTps: 1, prefillTps: 1,
      finishReason: "stop",
    });
    expect((await row(t, id)).answer).toBe("Hello world!");
  });

  test("a retried finish, release or fail whose first try landed answers ok", async () => {
    const a = await queue(t, "a");
    await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b" });
    const fin = { promptId: a, claimSeq: 1, tail: "x", tokens: 1, promptTokens: 1, decodeTps: 1, prefillTps: 1,
                  finishReason: "stop" };
    expect(await t.mutation(internal.bridge.finish, fin)).toEqual({ ok: true });
    expect(await t.mutation(internal.bridge.finish, fin)).toEqual({ ok: true, repeat: true });

    const b = await queue(t, "b");
    await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b" });
    const rel = { promptId: b, claimSeq: 1, reason: "busy" as const };
    expect((await t.mutation(internal.bridge.release, rel)).ok).toBe(true);
    expect(await t.mutation(internal.bridge.release, rel)).toEqual({ ok: true, repeat: true });
  });

  test("a retried claim gets the same prompt back instead of a second one", async () => {
    const first = await queue(t, "first");
    vi.advanceTimersByTime(5);
    await queue(t, "second");
    const one = await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b", claimKey: "k1" });
    const again = await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b", claimKey: "k1" });
    expect(again.prompt).toEqual(one.prompt);
    expect(one.prompt!.id).toBe(first);
    const other = await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b", claimKey: "k2" });
    expect(other.prompt!.text).toBe("second");
  });

  test("an append after the lease requeued the prompt is refused, even before anyone reclaims it", async () => {
    const id = await queue(t);
    await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "frozen" });
    vi.advanceTimersByTime(DEFAULT_LEASE_MS + 1);
    await t.finishInProgressScheduledFunctions();
    const late = await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 1, delta: "late", offset: 0 });
    expect(late).toEqual({ ok: false, reason: "stale" });
    expect((await row(t, id)).answer).toBe("");
    const log = await t.query(api.events.list, { code: CODE });
    expect(log.at(-1)?.text).toBe("Refused a late write for Ada's prompt: claim #1 had lost its lease");
  });
});

describe("presence abuse", () => {
  let t: T;
  beforeEach(async () => {
    vi.useFakeTimers();
    t = newRoom();
    await openRoom(t);
  });
  afterEach(() => vi.useRealTimers());

  const hb = (t: T, userId: string, interval = 10_000) =>
    t.mutation(api.presence.heartbeat, { roomId: CODE, userId, sessionId: session(), interval });

  test("a NaN interval is treated as 10 s, so the face still times out", async () => {
    const { roomToken } = await hb(t, "Ghost#abcd", Number.NaN);
    expect((await t.query(api.presence.list, { roomToken }))[0].online).toBe(true);
    vi.advanceTimersByTime(60_000);
    await t.finishInProgressScheduledFunctions();
    expect((await t.query(api.presence.list, { roomToken }))[0].online).toBe(false);
  });

  test("new faces per room are rate limited, returning faces are not", async () => {
    for (let i = 0; i < 20; i++) await hb(t, `p${i}#aaaa`);
    let err: unknown;
    try {
      await hb(t, "p20#aaaa");
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(ConvexError);
    expect((err as ConvexError<{ name: string }>).data.name).toBe("presenceJoin");
    await expect(hb(t, "p3#aaaa")).resolves.toBeTruthy(); // already in the room
  });

  test("names made only of invisible characters are refused", async () => {
    let err: unknown;
    try {
      await t.mutation(api.prompts.submit, { code: CODE, author: "​​", sessionId: session(), text: "q" });
    } catch (e) {
      err = e;
    }
    expect((err as ConvexError<{ code: string }>).data.code).toBe("bad_author");
    await expect(hb(t, "‮Ada#aaaa")).rejects.toBeInstanceOf(ConvexError);
  });
});
