/// <reference types="vite/client" />
import presenceTest from "@convex-dev/presence/test";
import rateLimiterTest from "@convex-dev/rate-limiter/test";
import { convexTest } from "convex-test";
import { internal } from "../convex/_generated/api";
import schema from "../convex/schema";

export const modules = import.meta.glob("../convex/**/*.ts");
export const SECRET = "test-secret-0123456789-abcdefghij";
export const CODE = "calhacks";

export function newRoom() {
  const t = convexTest(schema, modules);
  rateLimiterTest.register(t, "rateLimiter");
  presenceTest.register(t, "presence");
  return t;
}

export type T = ReturnType<typeof newRoom>;

export const POOL = {
  code: CODE,
  bridgeId: "bridge-a",
  title: "Cal Hacks pool",
  modelId: "kimi-k2.6",
  deploymentState: "ready",
  tier: "gpu",
  offloaded: "62/62",
  slots: 1,
  machines: [
    {
      nodeId: "local",
      name: "GX10",
      device: "CUDA0",
      layers: 40,
      layerRange: "0-39",
      weightBytes: 90e9,
      holdsOutput: false,
      path: "local" as const,
      state: "eligible",
      os: "linux",
    },
    {
      nodeId: "g14",
      name: "G14",
      device: "CUDA0",
      layers: 22,
      layerRange: "40-61",
      weightBytes: 14e9,
      holdsOutput: true,
      path: "relay" as const,
      state: "eligible",
      os: "windows",
    },
  ],
};

export async function openRoom(t: T) {
  return await t.mutation(internal.bridge.syncPool, POOL);
}

let sessionCounter = 0;
export function session(): string {
  sessionCounter += 1;
  return `session-${String(sessionCounter).padStart(12, "0")}`;
}
