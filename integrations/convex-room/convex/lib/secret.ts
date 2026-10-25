/**
 * Bearer check for the bridge routes. Both sides are hashed with SHA-256 first, so the comparison always runs over
 * 32 bytes and its time does not depend on where the first wrong character is or on the secret's length.
 */
export async function bearerMatches(header: string | null, secret: string | undefined): Promise<boolean> {
  if (!secret || secret.length < 24 || !header || !header.startsWith("Bearer ")) return false;
  const [given, expected] = await Promise.all([sha256(header.slice(7)), sha256(secret)]);
  let diff = 0;
  for (let i = 0; i < expected.length; i++) diff |= given[i] ^ expected[i];
  return diff === 0;
}

async function sha256(text: string): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}
