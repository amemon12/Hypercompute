import { describe, expect, it } from 'vitest';
import { createEventDeduper, errorFromBody, newIdempotencyKey } from './client';
import type { HcEvent } from './types';

describe('errorFromBody', () => {
  it('keeps the gateway {detail, code, fix} shape', () => {
    const err = errorFromBody(409, { detail: 'Another deployment is active', code: 'deployment_active', fix: 'Stop it first' }, 'x');
    expect(err.message).toBe('Another deployment is active');
    expect(err.code).toBe('deployment_active');
    expect(err.fix).toBe('Stop it first');
    expect(err.status).toBe(409);
  });

  it('summarizes FastAPI validation errors', () => {
    const err = errorFromBody(422, { detail: [{ loc: ['body', 'ctx_size'], msg: 'Input should be >= 256' }] }, 'x');
    expect(err.message).toBe('Input should be >= 256');
    expect(err.code).toBe('validation_error');
  });

  it('reads llama-server error envelopes', () => {
    const err = errorFromBody(400, { error: { message: 'prompt too long', type: 'exceed_context_size_error', code: 400 } }, 'x');
    expect(err.message).toBe('prompt too long');
    expect(err.code).toBe('exceed_context_size_error');
  });

  it('falls back for empty or unknown bodies', () => {
    expect(errorFromBody(500, null, 'Request failed with HTTP 500').message).toBe('Request failed with HTTP 500');
    expect(errorFromBody(500, ['x'], 'fallback').code).toBeNull();
  });
});

describe('newIdempotencyKey', () => {
  it('returns distinct RFC 4122 v4 UUIDs', () => {
    const a = newIdempotencyKey();
    const b = newIdempotencyKey();
    expect(a).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(a).not.toBe(b);
  });
});

describe('createEventDeduper', () => {
  const ev = (seq: number, at: string): HcEvent => ({ seq, at, type: 'node.state', message: 'm', data: {} });

  it('drops an event replayed after a reconnect', () => {
    const isNew = createEventDeduper();
    expect(isNew(ev(1, '2026-10-20T12:00:00Z'))).toBe(true);
    expect(isNew(ev(2, '2026-10-20T12:00:01Z'))).toBe(true);
    expect(isNew(ev(1, '2026-10-20T12:00:00Z'))).toBe(false);
    expect(isNew(ev(2, '2026-10-20T12:00:01Z'))).toBe(false);
  });

  it('accepts events from a restarted gateway that numbers from 1 again', () => {
    const isNew = createEventDeduper();
    expect(isNew(ev(41, '2026-10-20T12:00:00Z'))).toBe(true);
    expect(isNew(ev(1, '2026-10-20T12:05:00Z'))).toBe(true);
    expect(isNew(ev(2, '2026-10-20T12:05:01Z'))).toBe(true);
  });

  it('stays bounded and forgets the oldest keys first', () => {
    const isNew = createEventDeduper(2);
    isNew(ev(1, 'a'));
    isNew(ev(2, 'b'));
    isNew(ev(3, 'c'));
    expect(isNew(ev(3, 'c'))).toBe(false);
    expect(isNew(ev(1, 'a'))).toBe(true);
  });
});
