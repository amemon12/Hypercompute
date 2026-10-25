import { ConvexError } from "convex/values";

export const MAX_TEXT_CHARS = 500;
export const MAX_AUTHOR_CHARS = 32;
export const QUEUE_CAP = 8; // queued + running prompts per room
export const MAX_ATTEMPTS = 2; // a prompt whose host goes quiet is requeued once, then marked failed
export const MAX_ANSWER_CHARS = 6000;
export const MAX_DELTA_CHARS = 2000;
export const RECENT_FINISHED = 12; // finished prompts the shared list keeps showing
export const BRIDGE_ONLINE_MS = 30_000;
export const DEFAULT_LEASE_MS = 90_000;
export const ROOM_CODE = /^[a-z0-9][a-z0-9-]{2,31}$/;
export const SESSION_ID = /^[A-Za-z0-9_-]{16,64}$/;

export type RefusalCode =
  | "bad_room"
  | "room_closed"
  | "bad_text"
  | "bad_author"
  | "bad_session"
  | "queue_full";

export function refuse(code: RefusalCode, message: string): never {
  throw new ConvexError({ code, message });
}

/** The lease length: HC_LEASE_MS on the deployment when it is a sane number, else 90 s. */
export function leaseMs(): number {
  const raw = Number(process.env.HC_LEASE_MS);
  return Number.isFinite(raw) && raw >= 5_000 && raw <= 600_000 ? raw : DEFAULT_LEASE_MS;
}

/** Collapses whitespace runs, strips control characters and trims. */
export function cleanText(raw: string): string {
  return raw
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, "")
    .replace(/[​-‏‪-‮⁠-⁩﻿]/g, "") // zero-width and bidi controls
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function cleanAuthor(raw: string): string {
  return cleanText(raw).replace(/\s+/g, " ");
}

export function checkPrompt(textRaw: string, authorRaw: string, sessionId: string): { text: string; author: string } {
  const text = cleanText(textRaw);
  const author = cleanAuthor(authorRaw);
  if (!SESSION_ID.test(sessionId)) refuse("bad_session", "reload the page to get a new session");
  if (text.length === 0) refuse("bad_text", "type a question first");
  if (text.length > MAX_TEXT_CHARS) refuse("bad_text", `keep it under ${MAX_TEXT_CHARS} characters`);
  if (author.length === 0) refuse("bad_author", "pick a name first");
  if (author.length > MAX_AUTHOR_CHARS) refuse("bad_author", `names are at most ${MAX_AUTHOR_CHARS} characters`);
  return { text, author };
}

/** A short public tag for a session id (SHA-256, first 12 hex chars). The phone computes the same tag locally. */
export async function ownerKeyOf(sessionId: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(sessionId));
  return Array.from(new Uint8Array(digest).slice(0, 6), (b) => b.toString(16).padStart(2, "0")).join("");
}
