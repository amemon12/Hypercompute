import { describe, expect, it } from 'vitest';
import { artifact, enginesStatus } from '../testing/fixtures';
import { DEFAULT_RUN_OPTIONS, advancedSummary, effectiveEngine, planRequestFor } from './runOptions';

describe('planRequestFor', () => {
  it('sends every advanced option with the gateway field names', () => {
    expect(planRequestFor('m', 'auto', ['a'], 4096, DEFAULT_RUN_OPTIONS)).toEqual({
      model_id: 'm',
      node_ids: ['a'],
      strategy: 'auto',
      ctx_size: 4096,
      engine: null,
      objective: 'interactive',
      use_profiles: true,
      allow_ram_tier: true,
      allow_capacity_mode: true,
    });
  });

  it('forces an engine and drops friends for "this machine only"', () => {
    const req = planRequestFor('m', 'local_only', ['a'], 2048, {
      ...DEFAULT_RUN_OPTIONS,
      engine: 'vllm',
      objective: 'throughput',
      allowCapacityMode: false,
    });
    expect(req.node_ids).toBeNull();
    expect(req.engine).toBe('vllm');
    expect(req.objective).toBe('throughput');
    expect(req.allow_capacity_mode).toBe(false);
  });
});

describe('effectiveEngine', () => {
  it('falls back to automatic when the forced engine cannot serve the model', () => {
    expect(effectiveEngine('vllm', enginesStatus({ available: ['llama_rpc', 'vllm'] }), artifact({ format: 'gguf' }))).toBe('auto');
    expect(effectiveEngine('vllm', enginesStatus({ available: ['llama_rpc', 'vllm'] }), artifact({ format: 'hf' }))).toBe('vllm');
    expect(effectiveEngine('auto', undefined, null)).toBe('auto');
  });
});

describe('advancedSummary', () => {
  it('lists only what differs from the defaults', () => {
    expect(advancedSummary(DEFAULT_RUN_OPTIONS)).toEqual([]);
    expect(
      advancedSummary({ engine: 'exo', objective: 'throughput', useProfiles: false, allowRamTier: false, allowCapacityMode: false }),
    ).toEqual(['engine EXO', 'many users', 'no measured profiles', 'no host RAM', 'no capacity mode']);
  });
});
