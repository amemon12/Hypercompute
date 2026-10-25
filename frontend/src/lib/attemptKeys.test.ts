import { describe, expect, it } from 'vitest';
import { AttemptKeys } from './attemptKeys';

function counterKeys(): AttemptKeys {
  let n = 0;
  return new AttemptKeys(() => `key-${++n}`);
}

describe('AttemptKeys', () => {
  it('reuses the key when an attempt failed, so a retry stays idempotent', async () => {
    const keys = counterKeys();
    const seen: string[] = [];
    await expect(
      keys.run('plan-a', async (k) => {
        seen.push(k);
        throw new Error('network down');
      }),
    ).rejects.toThrow('network down');
    await keys.run('plan-a', async (k) => seen.push(k));
    expect(seen).toEqual(['key-1', 'key-1']);
  });

  it('uses a new key after the gateway answered, so the same plan can run again', async () => {
    const keys = counterKeys();
    const seen: string[] = [];
    await keys.run('plan-a', async (k) => seen.push(k));
    await keys.run('plan-a', async (k) => seen.push(k));
    expect(seen).toEqual(['key-1', 'key-2']);
  });

  it('keeps keys separate per plan', async () => {
    const keys = counterKeys();
    const seen: string[] = [];
    await expect(keys.run('plan-a', async () => Promise.reject(new Error('x')))).rejects.toThrow();
    await keys.run('plan-b', async (k) => seen.push(k));
    await keys.run('plan-a', async (k) => seen.push(k));
    expect(seen).toEqual(['key-2', 'key-1']);
  });
});
