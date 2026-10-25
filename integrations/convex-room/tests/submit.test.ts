import { ConvexError } from "convex/values";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { api, internal } from "../convex/_generated/api";
import { MAX_AUTHOR_CHARS, MAX_TEXT_CHARS, QUEUE_CAP } from "../convex/lib/limits";
import { CODE, newRoom, openRoom, session, type T } from "./setup";

async function refusal(p: Promise<unknown>): Promise<Record<string, unknown>> {
  try {
    await p;
  } catch (err) {
    expect(err).toBeInstanceOf(ConvexError);
    return (err as ConvexError<Record<string, unknown>>).data;
  }
  throw new Error("expected the mutation to be refused");
}

function submit(t: T, over: Partial<{ code: string; author: string; sessionId: string; text: string }> = {}) {
  return t.mutation(api.prompts.submit, {
    code: CODE,
    author: "Ada",
    sessionId: session(),
    text: "How many layers does the GX10 hold?",
    ...over,
  });
}

describe("prompts.submit input caps", () => {
  let t: T;
  beforeEach(async () => {
    t = newRoom();
    await openRoom(t);
  });

  test("accepts a normal prompt and queues it", async () => {
    const { position } = await submit(t);
    expect(position).toBe(1);
    const list = await t.query(api.prompts.list, { code: CODE });
    expect(list).toHaveLength(1);
    expect(list[0].status).toBe("queued");
    expect(list[0]).not.toHaveProperty("sessionId");
    expect(list[0].ownerKey).toMatch(/^[0-9a-f]{12}$/);
  });

  test(`refuses text over ${MAX_TEXT_CHARS} chars and empty text`, async () => {
    expect((await refusal(submit(t, { text: "x".repeat(MAX_TEXT_CHARS + 1) }))).code).toBe("bad_text");
    expect((await refusal(submit(t, { text: "  \n\t " }))).code).toBe("bad_text");
    await expect(submit(t, { text: "x".repeat(MAX_TEXT_CHARS) })).resolves.toBeTruthy();
  });

  test(`refuses author names over ${MAX_AUTHOR_CHARS} chars and blank names`, async () => {
    expect((await refusal(submit(t, { author: "n".repeat(MAX_AUTHOR_CHARS + 1) }))).code).toBe("bad_author");
    expect((await refusal(submit(t, { author: "   " }))).code).toBe("bad_author");
  });

  test("strips control characters before measuring", async () => {
    await submit(t, { text: "hi\u0000\u0007 there" });
    const [row] = await t.query(api.prompts.list, { code: CODE });
    expect(row.text).toBe("hi there");
  });

  test("refuses an unknown room, a closed room and a malformed session id", async () => {
    expect((await refusal(submit(t, { code: "nosuch" }))).code).toBe("bad_room");
    expect((await refusal(submit(t, { sessionId: "short" }))).code).toBe("bad_session");
    await t.mutation(internal.bridge.admin, { code: CODE, action: "close" });
    expect((await refusal(submit(t))).code).toBe("room_closed");
    await t.mutation(internal.bridge.admin, { code: CODE, action: "open" });
    await expect(submit(t)).resolves.toBeTruthy();
  });
});

describe("rate limits and the queue cap", () => {
  let t: T;
  beforeEach(async () => {
    vi.useFakeTimers();
    t = newRoom();
    await openRoom(t);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test("one session gets 3 prompts at once, then must wait", async () => {
    const sessionId = session();
    for (let i = 0; i < 3; i++) await submit(t, { sessionId });
    const data = await refusal(submit(t, { sessionId }));
    expect(data.kind).toBe("RateLimited");
    expect(data.name).toBe("promptSession");
    expect(Number(data.retryAfter)).toBeGreaterThan(0);

    vi.advanceTimersByTime(31_000);
    await expect(submit(t, { sessionId })).resolves.toBeTruthy();
  });

  test("a refused prompt spends no tokens", async () => {
    const sessionId = session();
    for (let i = 0; i < 5; i++) await refusal(submit(t, { sessionId, text: "" }));
    for (let i = 0; i < 3; i++) await submit(t, { sessionId });
  });

  test("the whole room is limited, so fresh session ids cannot flood the host", async () => {
    for (let i = 0; i < 6; i++) await submit(t);
    const data = await refusal(submit(t));
    expect(data.kind).toBe("RateLimited");
    expect(data.name).toBe("promptRoom");
  });

  test(`at most ${QUEUE_CAP} prompts wait or run at once`, async () => {
    for (let i = 0; i < 6; i++) await submit(t);
    vi.advanceTimersByTime(30_000);
    await submit(t);
    await submit(t);
    expect((await refusal(submit(t))).code).toBe("queue_full");

    // A finished prompt frees a place in the queue.
    const { prompt } = await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b" });
    await t.mutation(internal.bridge.finish, {
      promptId: prompt!.id,
      claimSeq: prompt!.claimSeq,
      tail: "ok",
      tokens: 1,
      promptTokens: 9,
      decodeTps: 2,
      prefillTps: 3,
      finishReason: "stop",
    });
    await expect(submit(t)).resolves.toBeTruthy();
  });
});

describe("presence", () => {
  test("heartbeats only for an existing room, with a bounded name", async () => {
    const t = newRoom();
    await openRoom(t);
    const hb = (userId: string, roomId = CODE) =>
      t.mutation(api.presence.heartbeat, { roomId, userId, sessionId: session(), interval: 10_000 });
    expect((await refusal(hb("Ada#ab12", "nosuch"))).code).toBe("bad_room");
    expect((await refusal(hb("no-tag"))).code).toBe("bad_author");
    expect((await refusal(hb(`${"n".repeat(MAX_AUTHOR_CHARS + 1)}#ab12`))).code).toBe("bad_author");
    const { roomToken } = await hb("Ada#ab12");
    const people = await t.query(api.presence.list, { roomToken });
    expect(people.map((p) => p.userId)).toEqual(["Ada#ab12"]);
  });
});
