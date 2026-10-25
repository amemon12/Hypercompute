/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type * as bridge from "../bridge.js";
import type * as events from "../events.js";
import type * as http from "../http.js";
import type * as lib_events from "../lib/events.js";
import type * as lib_limits from "../lib/limits.js";
import type * as lib_secret from "../lib/secret.js";
import type * as machines from "../machines.js";
import type * as presence from "../presence.js";
import type * as prompts from "../prompts.js";
import type * as rateLimits from "../rateLimits.js";
import type * as rooms from "../rooms.js";

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";

declare const fullApi: ApiFromModules<{
  bridge: typeof bridge;
  events: typeof events;
  http: typeof http;
  "lib/events": typeof lib_events;
  "lib/limits": typeof lib_limits;
  "lib/secret": typeof lib_secret;
  machines: typeof machines;
  presence: typeof presence;
  prompts: typeof prompts;
  rateLimits: typeof rateLimits;
  rooms: typeof rooms;
}>;

/**
 * A utility for referencing Convex functions in your app's public API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;

/**
 * A utility for referencing Convex functions in your app's internal API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = internal.myModule.myFunction;
 * ```
 */
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;

export declare const components: {
  presence: import("@convex-dev/presence/_generated/component.js").ComponentApi<"presence">;
  rateLimiter: import("@convex-dev/rate-limiter/_generated/component.js").ComponentApi<"rateLimiter">;
};
