import { MINUTE, RateLimiter } from "@convex-dev/rate-limiter";
import { components } from "./_generated/api";

export const rateLimiter = new RateLimiter(components.rateLimiter, {
  // Each phone: 3 prompts at once, then one every 30 s.
  promptSession: { kind: "token bucket", rate: 2, period: MINUTE, capacity: 3 },
  // The whole room, so fresh session ids cannot flood the host: 12 a minute, bursts of 6.
  promptRoom: { kind: "token bucket", rate: 12, period: MINUTE, capacity: 6 },
  // Presence heartbeats per session (the hook sends one every 10 s).
  presenceSession: { kind: "token bucket", rate: 30, period: MINUTE, capacity: 10 },
  // New faces per room: a table of judges arriving at once fits, a script minting names does not.
  presenceJoin: { kind: "token bucket", rate: 30, period: MINUTE, capacity: 20 },
});
