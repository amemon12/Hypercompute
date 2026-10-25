/**
 * One Idempotency-Key per attempt to start a plan. Retrying an attempt that failed (network error,
 * gateway restart, 409) reuses its key, so the gateway can never start two runs for one press. Once the
 * gateway has answered with a deployment the attempt is settled, and the next press of Run together on
 * the same plan gets a new key. Reusing the key forever would make the gateway hand back the old
 * stopped or failed deployment instead of starting a new run.
 */
export class AttemptKeys {
  private readonly keys = new Map<string, string>();

  constructor(private readonly makeKey: () => string) {}

  async run<T>(id: string, attempt: (key: string) => Promise<T>): Promise<T> {
    const key = this.keys.get(id) ?? this.makeKey();
    this.keys.set(id, key);
    const result = await attempt(key);
    this.keys.delete(id);
    return result;
  }
}
