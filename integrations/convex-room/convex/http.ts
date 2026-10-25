/**
 * The only door the host uses: POST https://NAME.convex.site/bridge/<route>, each with
 * Authorization: Bearer <HC_ROOM_SECRET>. The bridge only ever calls out to these routes, so the host opens no port
 * and its gateway owner token never leaves the machine.
 */
import { httpRouter } from "convex/server";
import { ConvexError } from "convex/values";
import type { FunctionReference } from "convex/server";
import { internal } from "./_generated/api";
import { type ActionCtx, httpAction } from "./_generated/server";
import { bearerMatches } from "./lib/secret";

const MAX_BODY_BYTES = 64 * 1024;

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

type InternalMutation = FunctionReference<"mutation", "internal">;

function bridgeRoute(fn: InternalMutation) {
  return httpAction(async (ctx: ActionCtx, request: Request) => {
    if (!(await bearerMatches(request.headers.get("Authorization"), process.env.HC_ROOM_SECRET))) {
      return json(401, { error: "unauthorized" });
    }
    const text = await request.text();
    if (text.length > MAX_BODY_BYTES) return json(413, { error: "body too large" });
    let args: unknown;
    try {
      args = JSON.parse(text);
    } catch {
      return json(400, { error: "body must be JSON" });
    }
    if (typeof args !== "object" || args === null || Array.isArray(args)) {
      return json(400, { error: "body must be a JSON object" });
    }
    try {
      return json(200, await ctx.runMutation(fn, args as Record<string, unknown>));
    } catch (err) {
      // A bad argument is the caller's fault (400, final). Anything else, such as a write conflict that ran out of
      // retries or a timeout, is ours and may pass (500, which the bridge retries). Messages hold no secret.
      const message = err instanceof Error ? err.message.split("\n")[0].slice(0, 300) : "rejected";
      const callerFault = err instanceof ConvexError || /ArgumentValidationError|Validator error/.test(message);
      return json(callerFault ? 400 : 500, { error: callerFault ? message : "temporary failure; retry" });
    }
  });
}

const http = httpRouter();
const routes: Array<[string, InternalMutation]> = [
  ["/bridge/claim", internal.bridge.claim],
  ["/bridge/append", internal.bridge.append],
  ["/bridge/finish", internal.bridge.finish],
  ["/bridge/release", internal.bridge.release],
  ["/bridge/fail", internal.bridge.fail],
  ["/bridge/pool", internal.bridge.syncPool],
  ["/bridge/admin", internal.bridge.admin],
];
for (const [path, fn] of routes) {
  http.route({ path, method: "POST", handler: bridgeRoute(fn) });
}

export default http;
