import { Presence } from "@convex-dev/presence";
import { v } from "convex/values";
import { components } from "./_generated/api";
import { mutation, query } from "./_generated/server";
import { MAX_AUTHOR_CHARS, cleanAuthor, refuse } from "./lib/limits";
import { rateLimiter } from "./rateLimits";
import { roomByCode } from "./rooms";

export const presence = new Presence(components.presence);

// userId is "<name>#<tag>": the tag keeps two people called Sam apart, the name is what the face pile shows.
const USER_ID = new RegExp(`^[^#\\u0000-\\u001f]{1,${MAX_AUTHOR_CHARS}}#[a-z0-9]{4}$`);

const MAX_ONLINE = 60;

export const heartbeat = mutation({
  args: { roomId: v.string(), userId: v.string(), sessionId: v.string(), interval: v.number() },
  handler: async (ctx, { roomId, userId, sessionId, interval }) => {
    const room = await roomByCode(ctx, roomId);
    if (room === null) refuse("bad_room", "no such room; check the link");
    if (!USER_ID.test(userId) || cleanAuthor(userId) !== userId || sessionId.length > 256) {
      refuse("bad_author", "pick a shorter, plain name");
    }
    await rateLimiter.limit(ctx, "presenceSession", { key: sessionId, throws: true });
    // A face that is new to the room spends from a room-wide bucket and must fit under the cap, so rotating session
    // ids and names cannot flood the projector's face pile.
    const online = await presence.listRoom(ctx, roomId, true, MAX_ONLINE + 1);
    if (!online.some((p) => p.userId === userId)) {
      if (online.length >= MAX_ONLINE) refuse("bad_room", "the room is full right now");
      await rateLimiter.limit(ctx, "presenceJoin", { key: roomId, throws: true });
    }
    const bounded = Number.isFinite(interval) ? Math.min(Math.max(interval, 5_000), 60_000) : 10_000;
    return await presence.heartbeat(ctx, roomId, userId, sessionId, bounded);
  },
});

export const list = query({
  args: { roomToken: v.string() },
  handler: async (ctx, { roomToken }) => {
    return await presence.list(ctx, roomToken);
  },
});

export const disconnect = mutation({
  args: { sessionToken: v.string() },
  handler: async (ctx, { sessionToken }) => {
    // Called over HTTP by sendBeacon when a tab closes, so there is nothing to check here beyond the token.
    return await presence.disconnect(ctx, sessionToken);
  },
});
