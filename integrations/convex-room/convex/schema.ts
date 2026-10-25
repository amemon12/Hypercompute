import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export const promptStatus = v.union(
  v.literal("queued"),
  v.literal("running"),
  v.literal("done"),
  v.literal("failed"),
);

export const linkPath = v.union(
  v.literal("local"),
  v.literal("direct"),
  v.literal("relay"),
  v.literal("connecting"),
  v.literal("lan"),
  v.literal("unknown"),
);

export default defineSchema({
  // One row per room code. Only the host bridge creates a room (POST /bridge/pool), so a phone can never make one.
  rooms: defineTable({
    code: v.string(),
    title: v.string(),
    open: v.boolean(),
    modelId: v.union(v.string(), v.null()),
    deploymentState: v.string(),
    tier: v.union(v.string(), v.null()),
    offloaded: v.union(v.string(), v.null()), // llama.cpp's own "25/25" line, when the gateway parsed it
    slots: v.number(),
    bridgeId: v.string(),
    bridgeSeenAt: v.number(),
    lastDecodeTps: v.union(v.number(), v.null()), // from engine timings of the last finished prompt
    lastPrefillTps: v.union(v.number(), v.null()),
    servedCount: v.number(),
    updatedAt: v.number(),
  }).index("by_code", ["code"]),

  // The machines the active deployment uses, plus joined machines it leaves out (layers 0).
  machines: defineTable({
    roomId: v.id("rooms"),
    nodeId: v.string(),
    name: v.string(),
    device: v.string(), // CUDA0, MTL0, CPU, ...
    layers: v.number(),
    layerRange: v.union(v.string(), v.null()),
    weightBytes: v.number(),
    holdsOutput: v.boolean(),
    path: linkPath,
    state: v.string(),
    os: v.union(v.string(), v.null()),
    order: v.number(),
  })
    .index("by_room", ["roomId"])
    .index("by_room_node_device", ["roomId", "nodeId", "device"]),

  // What Convex did for the room, newest last: claims, lease expiries, requeues, refused stale writes. Kept short.
  events: defineTable({
    roomId: v.id("rooms"),
    kind: v.string(),
    text: v.string(),
    at: v.number(),
  }).index("by_room", ["roomId"]),

  prompts: defineTable({
    roomId: v.id("rooms"),
    author: v.string(),
    sessionId: v.string(), // the rate-limit key; never returned by a public query
    ownerKey: v.string(), // a short hash of sessionId, so a phone can find its own prompts in the shared list
    text: v.string(),
    status: promptStatus,
    answer: v.string(),
    reasoning: v.string(), // reasoning_content deltas (gpt-oss, Kimi thinking), shown dimmed above the answer
    tokens: v.union(v.number(), v.null()),
    promptTokens: v.union(v.number(), v.null()),
    decodeTps: v.union(v.number(), v.null()),
    prefillTps: v.union(v.number(), v.null()),
    liveTps: v.union(v.number(), v.null()), // per-token engine timings while streaming, when the engine sends them
    finishReason: v.union(v.string(), v.null()),
    leaseUntil: v.union(v.number(), v.null()),
    attempts: v.number(), // claims that count toward the failure limit
    claimSeq: v.number(), // fencing token: bumps on every claim, so a writer from an older claim is refused
    claimKey: v.optional(v.string()), // the bridge's id for the claim call, so a retried claim gets the same prompt
    requeues: v.number(),
    bridgeId: v.union(v.string(), v.null()),
    note: v.union(v.string(), v.null()), // the last lease or requeue event, shown on the card
    error: v.union(v.string(), v.null()),
    createdAt: v.number(),
    startedAt: v.union(v.number(), v.null()),
    firstTokenAt: v.union(v.number(), v.null()),
    finishedAt: v.union(v.number(), v.null()),
  })
    .index("by_room_status", ["roomId", "status"])
    .index("by_room", ["roomId"]),
});
