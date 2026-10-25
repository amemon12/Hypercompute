import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { api, internal } from "../convex/_generated/api";
import { DEFAULT_LEASE_MS } from "../convex/lib/limits";
import { CODE, POOL, newRoom, openRoom, session, type T } from "./setup";

async function log(t: T): Promise<string[]> {
  return (await t.query(api.events.list, { code: CODE })).map((e) => `${e.kind}: ${e.text}`);
}

describe("the room's activity log", () => {
  let t: T;
  beforeEach(async () => {
    vi.useFakeTimers();
    t = newRoom();
    await openRoom(t);
  });
  afterEach(() => vi.useRealTimers());

  test("records the life of a prompt in order", async () => {
    const { id } = await t.mutation(api.prompts.submit, { code: CODE, author: "Ada", sessionId: session(), text: "hi" });
    await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b" });
    await t.mutation(internal.bridge.finish, {
      promptId: id, claimSeq: 1, tail: "hello", tokens: 3, promptTokens: 9, decodeTps: 1.8, prefillTps: 3,
      finishReason: "stop",
    });
    expect(await log(t)).toEqual([
      "host: Room opened by host bridge bridge-a",
      "submit: Ada asked: hi",
      "claim: Host claimed Ada's prompt; lease 90 s",
      "done: Answered Ada: 3 tokens in 0.0 s, 1.80 tok/s decode",
    ]);
  });

  test("shows the lease expiry, the retry and the refused stale write", async () => {
    const { id } = await t.mutation(api.prompts.submit, { code: CODE, author: "Ada", sessionId: session(), text: "q" });
    await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "frozen" });
    vi.advanceTimersByTime(DEFAULT_LEASE_MS + 1);
    await t.finishInProgressScheduledFunctions();
    await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "thawed" });
    await t.mutation(internal.bridge.append, { promptId: id, claimSeq: 1, delta: "zombie" });
    expect((await log(t)).slice(-3)).toEqual([
      "lease: The host went quiet and its lease expired; requeued Ada's prompt",
      "claim: Host claimed Ada's prompt (attempt 2 of 2); lease 90 s",
      "fence: Refused a stale write for Ada's prompt: claim #1 was superseded by #2",
    ]);
  });

  test("a busy engine retried in a loop adds one pair of lines, not one per retry", async () => {
    const { id } = await t.mutation(api.prompts.submit, { code: CODE, author: "Ada", sessionId: session(), text: "q" });
    for (let i = 0; i < 5; i++) {
      const { prompt } = await t.mutation(internal.bridge.claim, { code: CODE, bridgeId: "b" });
      await t.mutation(internal.bridge.release, { promptId: id, claimSeq: prompt!.claimSeq, reason: "busy" });
      vi.advanceTimersByTime(2_000);
    }
    const lines = await log(t);
    expect(lines.filter((l) => l.startsWith("busy:"))).toHaveLength(1);
    expect(lines.filter((l) => l.startsWith("claim:"))).toHaveLength(1);
  });

  test("notes a bridge coming back and a model state change", async () => {
    vi.advanceTimersByTime(45_000);
    await t.mutation(internal.bridge.syncPool, { ...POOL, bridgeId: "bridge-b", deploymentState: "loading" });
    expect((await log(t)).slice(-2)).toEqual([
      "host: Host bridge bridge-b connected",
      "host: Model kimi-k2.6: loading",
    ]);
  });

  test("clear empties the log down to one line", async () => {
    await t.mutation(api.prompts.submit, { code: CODE, author: "Ada", sessionId: session(), text: "q" });
    await t.mutation(internal.bridge.admin, { code: CODE, action: "clear" });
    expect(await log(t)).toEqual(["admin: Host cleared the room"]);
  });
});
