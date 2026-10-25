/** Who this phone is, kept in localStorage when the browser allows it (a private tab may not). */

const KEY = "hc-room-identity-v1";
const MINE = "hc-room-mine-v1";

export interface Identity {
  name: string;
  tag: string; // 4 chars, keeps two people with the same name apart in the face pile
  sessionId: string; // the rate-limit key; random, never shown
}

function random(chars: number, alphabet: string): string {
  const bytes = crypto.getRandomValues(new Uint8Array(chars)); // works on plain-http LAN pages too
  return Array.from(bytes, (b) => alphabet[b % alphabet.length]).join("");
}

const B64URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

export function loadIdentity(): Identity | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<Identity>;
    if (typeof parsed.name === "string" && typeof parsed.tag === "string" && typeof parsed.sessionId === "string") {
      return { name: parsed.name, tag: parsed.tag, sessionId: parsed.sessionId };
    }
  } catch {
    // storage blocked or corrupt: ask for a name again
  }
  return null;
}

export function createIdentity(name: string, previous: Identity | null): Identity {
  const identity = {
    name,
    tag: previous?.tag ?? random(4, "abcdefghijklmnopqrstuvwxyz0123456789"),
    sessionId: previous?.sessionId ?? random(24, B64URL),
  };
  try {
    localStorage.setItem(KEY, JSON.stringify(identity));
  } catch {
    // works for this tab only
  }
  return identity;
}

export function presenceId(identity: Identity): string {
  return `${identity.name}#${identity.tag}`;
}

export function displayName(userId: string): string {
  const i = userId.lastIndexOf("#");
  return i > 0 ? userId.slice(0, i) : userId;
}

export function loadMine(): string[] {
  try {
    const raw = JSON.parse(localStorage.getItem(MINE) ?? "[]");
    return Array.isArray(raw) ? raw.filter((x): x is string => typeof x === "string").slice(-50) : [];
  } catch {
    return [];
  }
}

export function rememberMine(ids: string[]): void {
  try {
    localStorage.setItem(MINE, JSON.stringify(ids.slice(-50)));
  } catch {
    // fine: the highlight lasts for this tab only
  }
}
