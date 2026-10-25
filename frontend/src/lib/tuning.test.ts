import { describe, expect, it } from 'vitest';
import type { MeasuredProfile } from '../api/types';
import { candidate, plan, tuneResult } from '../testing/fixtures';
import { UNAVAILABLE } from './format';
import {
  currentTune,
  formatScore,
  formatSeconds,
  isTuneEvent,
  profileReuse,
  profileSummary,
  tuneRequestFrom,
  winnerIndex,
} from './tuning';

function profile(plan_fields: Record<string, unknown>): MeasuredProfile {
  return {
    id: 'p1',
    model_id: 'm',
    signature: 'm|ctx=4096|host=CUDA0',
    objective: 'interactive',
    plan_fields,
    ttft_ms: 120,
    decode_tokens_per_s: 40,
    prefill_tokens_per_s: 900,
    score: 1.7,
    tune_id: 't1',
    run_ids: [],
    created_at: '2026-10-20T12:00:00Z',
  };
}

describe('formatting', () => {
  it('formats score and seconds, never inventing 0', () => {
    expect(formatScore(1.234)).toBe('1.23 s');
    expect(formatScore(null)).toBe(UNAVAILABLE);
    expect(formatScore(Number.NaN)).toBe(UNAVAILABLE);
    expect(formatSeconds(12.34)).toBe('12.3 s');
    expect(formatSeconds(undefined)).toBe(UNAVAILABLE);
  });
});

describe('currentTune', () => {
  const old = tuneResult({ id: 'old', status: 'done', created_at: '2026-10-20T10:00:00Z' });
  const other = tuneResult({
    id: 'other',
    status: 'done',
    created_at: '2026-10-20T11:00:00Z',
    request: { ...old.request, model_id: 'other-model' },
  });

  it('prefers a running tune, then the newest for the model, then the newest', () => {
    const running = tuneResult({ id: 'run', status: 'running', created_at: '2026-10-20T09:00:00Z' });
    expect(currentTune([old, other, running], 'qwen2.5-7b-q4')?.id).toBe('run');
    expect(currentTune([old, other], 'qwen2.5-7b-q4')?.id).toBe('old');
    expect(currentTune([old, other], null)?.id).toBe('other');
    expect(currentTune([], null)).toBeNull();
    expect(currentTune(undefined, null)).toBeNull();
  });
});

describe('winnerIndex', () => {
  it('highlights the best candidate only once it was measured', () => {
    const t = tuneResult({ candidates: [candidate({ status: 'done' }), candidate({ status: 'running' })], best_index: 1 });
    expect(winnerIndex(t)).toBeNull();
    expect(winnerIndex({ ...t, best_index: 0 })).toBe(0);
    expect(winnerIndex({ ...t, best_index: 5 })).toBeNull();
    expect(winnerIndex({ ...t, best_index: null })).toBeNull();
  });
});

describe('tuneRequestFrom', () => {
  const base = { model_id: 'm', node_ids: ['a', 'b'], strategy: 'auto' as const, ctx_size: 8192, objective: 'throughput' as const };
  const opts = { maxCandidates: 3, tuneFlags: false };

  it('copies the selection and sends no friends for "this machine only"', () => {
    expect(tuneRequestFrom(base, opts)).toEqual({
      model_id: 'm',
      node_ids: ['a', 'b'],
      ctx_size: 8192,
      objective: 'throughput',
      max_candidates: 3,
      tune_flags: false,
      engine: 'llama_rpc',
      strategy: 'auto',
      allow_ram_tier: true,
      allow_capacity_mode: true,
    });
    const local = tuneRequestFrom({ ...base, strategy: 'local_only', node_ids: null, objective: undefined }, { maxCandidates: 4, tuneFlags: true });
    expect(local.node_ids).toEqual([]);
    expect(local.strategy).toBe('local_only');
    expect(local.objective).toBe('interactive');
  });

  it("keeps the owner's engine and plan limits, so the tune measures only plans they accept", () => {
    const sel = { ...base, strategy: 'all_selected' as const, engine: 'llama_rpc' as const, allow_ram_tier: false, allow_capacity_mode: false };
    expect(tuneRequestFrom(sel, opts)).toMatchObject({
      engine: 'llama_rpc',
      strategy: 'all_selected',
      allow_ram_tier: false,
      allow_capacity_mode: false,
    });
    expect(tuneRequestFrom({ ...base, engine: null }, opts).engine).toBe('llama_rpc');
  });
});

describe('profileReuse', () => {
  const llama = candidate({ status: 'done' });
  const vllm = candidate({ status: 'done', plan: plan({ engine: 'vllm', model_id: 'qwen2.5-7b-awq' }) });
  const done = tuneResult({ status: 'done', profile_id: 'p1', best_index: 0, candidates: [llama] });

  it('claims reuse only for a measured llama.cpp winner of the tuned model', () => {
    expect(profileReuse(done)).toEqual({
      reused: true,
      heading: 'Saved as a measured profile',
      text: 'later plans for this model and these machines reuse it',
    });
    const other = profileReuse({ ...done, candidates: [vllm] });
    expect(other?.reused).toBe(false);
    expect(other?.heading).toBe('Saved, but not used by later plans');
    expect(other?.text).toContain('measured with vLLM');
    const variant = profileReuse({ ...done, candidates: [candidate({ status: 'done', plan: plan({ model_id: 'other' }) })] });
    expect(variant).toMatchObject({ reused: false, text: expect.stringContaining('measured other') });
    expect(profileReuse({ ...done, best_index: null })?.reused).toBe(false);
  });

  it('neither claims nor denies reuse when the winner is missing', () => {
    // heading and detail agree: the profile was saved, whether plans replay it is unknown
    const unknown = profileReuse({ ...done, best_index: null });
    expect(unknown?.heading).toBe('Saved, reuse unconfirmed');
    expect(unknown?.heading).not.toContain('not used');
    expect(unknown?.text).toContain('the measured winner is unavailable');
  });

  it('says nothing until a profile was saved', () => {
    expect(profileReuse({ ...done, profile_id: null })).toBeNull();
    expect(profileReuse({ ...done, status: 'running' })).toBeNull();
  });
});

describe('isTuneEvent', () => {
  it('matches tune.* event types', () => {
    const e = { seq: 1, at: '2026-10-20T12:00:00Z', message: 'x', data: {} };
    expect(isTuneEvent({ ...e, type: 'tune.progress' })).toBe(true);
    expect(isTuneEvent({ ...e, type: 'deployment.ready' })).toBe(false);
  });
});

describe('profileSummary', () => {
  it('reads well-formed plan fields', () => {
    const s = profileSummary(
      profile({ engine: 'llama_rpc', tier: 'gpu_ram', device_list: ['RPC0', 'CUDA0'], cpu_layers: 4, engine_flags: { flash_attn: 'on', ubatch: 512 } }),
    );
    expect(s).toEqual({ engine: 'llama_rpc', tier: 'gpu_ram', devices: ['RPC0', 'CUDA0'], cpuLayers: 4, flags: ['flash_attn=on', 'ubatch=512'] });
  });

  it('treats malformed fields as missing', () => {
    const s = profileSummary(profile({ engine: 7, tier: null, device_list: 'CUDA0', cpu_layers: 'x', engine_flags: ['a'] }));
    expect(s).toEqual({ engine: null, tier: null, devices: [], cpuLayers: null, flags: [] });
    const mixed = profileSummary(profile({ device_list: ['CUDA0', 3], engine_flags: { a: 1, b: { c: 2 } } }));
    expect(mixed.devices).toEqual(['CUDA0']);
    expect(mixed.flags).toEqual(['a=1']);
  });
});
