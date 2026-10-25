import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { api } from "../convex/_generated/api";
import { CODE, POOL, SECRET, newRoom, type T } from "./setup";

function post(t: T, path: string, body: unknown, auth: string | null = `Bearer ${SECRET}`) {
  return t.fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(auth ? { Authorization: auth } : {}) },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

describe("bridge HTTP routes", () => {
  let t: T;
  beforeEach(() => {
    vi.stubEnv("HC_ROOM_SECRET", SECRET);
    t = newRoom();
  });
  afterEach(() => vi.unstubAllEnvs());

  test("refuse a missing or wrong bearer token", async () => {
    expect((await post(t, "/bridge/pool", POOL, null)).status).toBe(401);
    expect((await post(t, "/bridge/pool", POOL, "Bearer nope")).status).toBe(401);
    expect((await post(t, "/bridge/pool", POOL, SECRET)).status).toBe(401);
    expect(await t.query(api.rooms.get, { code: CODE })).toBeNull();
  });

  test("refuse everything when the deployment has no secret set", async () => {
    vi.stubEnv("HC_ROOM_SECRET", "");
    expect((await post(t, "/bridge/pool", POOL)).status).toBe(401);
  });

  test("pool creates the room and syncs machines; a second snapshot replaces them", async () => {
    const r = await post(t, "/bridge/pool", POOL);
    expect(r.status).toBe(200);
    let machines = await t.query(api.machines.list, { code: CODE });
    expect(machines.map((m) => [m.name, m.layers, m.path])).toEqual([
      ["GX10", 40, "local"],
      ["G14", 22, "relay"],
    ]);

    const g14Direct = { ...POOL.machines[1], path: "direct", layers: 62, layerRange: "0-61" };
    await post(t, "/bridge/pool", { ...POOL, machines: [g14Direct] });
    machines = await t.query(api.machines.list, { code: CODE });
    expect(machines.map((m) => [m.name, m.layers, m.path])).toEqual([["G14", 62, "direct"]]);
    const room = await t.query(api.rooms.get, { code: CODE });
    expect(room).toMatchObject({ modelId: "kimi-k2.6", deploymentState: "ready", open: true });
  });

  test("bad JSON and bad arguments are a 400 that names the field", async () => {
    expect((await post(t, "/bridge/claim", "{not json")).status).toBe(400);
    expect((await post(t, "/bridge/claim", [1, 2])).status).toBe(400);
    const r = await post(t, "/bridge/claim", { code: CODE });
    expect(r.status).toBe(400);
    expect((await r.json()).error).toMatch(/bridgeId/);
    expect((await post(t, "/bridge/pool", { ...POOL, code: "Not A Code" })).status).toBe(400);
  });

  test("a full claim, append, finish round trip over HTTP", async () => {
    await post(t, "/bridge/pool", POOL);
    await t.mutation(api.prompts.submit, {
      code: CODE, author: "Ada", sessionId: "s".repeat(20), text: "hello pool",
    });
    const claimed = await (await post(t, "/bridge/claim", { code: CODE, bridgeId: "b1" })).json();
    expect(claimed.prompt.text).toBe("hello pool");
    const { id, claimSeq } = claimed.prompt;
    expect((await (await post(t, "/bridge/append", { promptId: id, claimSeq, delta: "Hi" })).json()).ok).toBe(true);
    const done = await post(t, "/bridge/finish", {
      promptId: id, claimSeq, tail: "!", tokens: 2, promptTokens: 5, decodeTps: 50, prefillTps: 200,
      finishReason: "stop",
    });
    expect(await done.json()).toEqual({ ok: true });
    const [p] = await t.query(api.prompts.list, { code: CODE });
    expect(p).toMatchObject({ status: "done", answer: "Hi!", decodeTps: 50 });
  });

  test("admin clear removes finished and queued prompts", async () => {
    await post(t, "/bridge/pool", POOL);
    await t.mutation(api.prompts.submit, { code: CODE, author: "A", sessionId: "q".repeat(20), text: "one" });
    const r = await (await post(t, "/bridge/admin", { code: CODE, action: "clear" })).json();
    expect(r).toEqual({ ok: true, removed: 1 });
    expect(await t.query(api.prompts.list, { code: CODE })).toEqual([]);
  });
});
