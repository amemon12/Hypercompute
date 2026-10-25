import { describe, expect, it } from 'vitest';
import { isEngineId } from '../api/types';
import { artifact, assignment, enginesStatus, engineInfo, extraEngine, modelView, plan } from '../testing/fixtures';
import {
  artifactEngines,
  availableEngines,
  capacityWarning,
  cpuLayersText,
  engineChoiceState,
  engineInstallCommand,
  engineManagedText,
  extraEngineStatus,
  isModelUsable,
  layersText,
  llamaEngineStatus,
  modelSizeBytes,
  overflowText,
  planSourceLabel,
} from './engines';

describe('isEngineId', () => {
  it('accepts the four engine ids only', () => {
    expect(['llama_rpc', 'vllm', 'exo', 'airllm'].every(isEngineId)).toBe(true);
    expect(isEngineId('exo_paged')).toBe(false);
    expect(isEngineId(3)).toBe(false);
    expect(isEngineId(null)).toBe(false);
  });
});

describe('artifactEngines', () => {
  it('uses the manifest list when set, else the default for the format', () => {
    expect(artifactEngines(artifact({ format: 'gguf' }))).toEqual(['llama_rpc']);
    expect(artifactEngines(artifact({ format: 'hf' }))).toEqual(['airllm', 'vllm']);
    expect(artifactEngines(artifact({ format: 'mlx' }))).toEqual(['exo']);
    expect(artifactEngines(artifact({ format: 'hf', engines: ['vllm'] }))).toEqual(['vllm']);
  });
});

describe('models', () => {
  it('lets engine-managed models be planned without a download', () => {
    expect(isModelUsable(modelView({ status: 'engine_managed' }))).toBe(true);
    expect(isModelUsable(modelView({ status: 'downloaded' }))).toBe(true);
    expect(isModelUsable(modelView({ status: 'downloading' }))).toBe(false);
  });

  it('falls back to the manifest size when there are no files', () => {
    const mlx = modelView({ status: 'engine_managed', artifact: artifact({ format: 'mlx', size_bytes: 4096 }) });
    expect(modelSizeBytes(mlx)).toBe(4096);
    expect(modelSizeBytes(modelView())).toBeNull();
    expect(modelSizeBytes(modelView({ total_bytes: 10 }))).toBe(10);
  });

  it('says EXO downloads MLX models on each Mac', () => {
    expect(engineManagedText(artifact({ format: 'mlx' }))).toBe('downloaded by EXO on each Mac');
  });
});

describe('engineInstallCommand', () => {
  it('matches the CLI', () => {
    expect(engineInstallCommand('llama_rpc')).toBe('uv run hypercompute engine install');
    expect(engineInstallCommand('vllm')).toBe('uv run hypercompute engine install --engine vllm');
  });
});

describe('engineChoiceState', () => {
  const status = enginesStatus();

  it('disables engines that cannot serve the model format', () => {
    const s = engineChoiceState('vllm', status, artifact({ format: 'gguf' }));
    expect(s.enabled).toBe(false);
    expect(s.reason).toBe('not for this model (GGUF files run on llama.cpp)');
  });

  it('disables engines the host has not loaded, with the host reason', () => {
    const hf = artifact({ format: 'hf' });
    const vllm = engineChoiceState('vllm', status, hf);
    expect(vllm).toEqual({ enabled: false, reason: 'not supported on this computer (vLLM needs Linux or WSL2)' });
    const air = engineChoiceState('airllm', status, hf);
    expect(air.reason).toBe('not installed on this computer. Install: uv run hypercompute engine install --engine airllm');
  });

  it('asks for a restart when the engine is installed but not loaded', () => {
    const s = enginesStatus({ host: [extraEngine({ engine: 'airllm', installed: true, status: 'installed' })] });
    expect(engineChoiceState('airllm', s, artifact({ format: 'hf' })).reason).toContain('Restart the gateway');
  });

  it('enables available engines and stays enabled while status is unknown', () => {
    expect(engineChoiceState('llama_rpc', status, artifact())).toEqual({ enabled: true, reason: null });
    expect(engineChoiceState('vllm', undefined, null).enabled).toBe(true);
  });

  it('filters unknown names out of available', () => {
    expect(availableEngines(enginesStatus({ available: ['llama_rpc', 'exo_paged', 'exo'] }))).toEqual(['llama_rpc', 'exo']);
    expect(availableEngines(undefined)).toEqual([]);
  });
});

describe('engine status lines', () => {
  it('offers the install command only when needed', () => {
    expect(extraEngineStatus(extraEngine({ engine: 'airllm' }), 'airllm')).toEqual({
      text: 'not installed',
      tone: 'neutral',
      install: 'uv run hypercompute engine install --engine airllm',
    });
    expect(extraEngineStatus(extraEngine({ engine: 'exo', status: 'running', installed: true }), 'exo').install).toBeNull();
    expect(extraEngineStatus(extraEngine({ engine: 'vllm', status: 'unsupported' }), 'vllm').install).toBeNull();
    expect(extraEngineStatus(undefined, 'vllm').text).toBe('not reported');
  });

  it('describes llama.cpp from its EngineInfo', () => {
    expect(llamaEngineStatus(engineInfo()).text).toBe('b11770 · cuda, smoke test passed');
    expect(llamaEngineStatus(engineInfo({ installed: false })).install).toBe('uv run hypercompute engine install');
    expect(llamaEngineStatus(engineInfo({ smoke: 'failed' })).tone).toBe('bad');
  });
});

describe('plan wording', () => {
  it('warns on disk-backed tiers only', () => {
    expect(capacityWarning(plan())).toBeNull();
    expect(capacityWarning(plan({ tier: 'gpu_ram' }))).toBeNull();
    expect(capacityWarning(plan({ tier: 'gpu_disk', capacity_mode: true, cpu_layers: 6 }))).toContain('seconds or more per token');
    expect(capacityWarning(plan({ tier: 'disk', engine: 'airllm', capacity_mode: true }))).toContain('Capacity mode');
  });

  it('tells llama.cpp disk paging apart from AirLLM reading every layer', () => {
    // planner.finish_plan sends every gpu_disk plan with capacity_mode true
    const paged = capacityWarning(plan({ tier: 'gpu_disk', capacity_mode: true, cpu_layers: 6 }));
    expect(paged).toContain("runs on this PC's CPU (6 layers)");
    expect(paged).toContain('paged from the GGUF file on disk');
    expect(paged).not.toContain('every layer');
    const air = capacityWarning(plan({ tier: 'disk', engine: 'airllm', capacity_mode: true, assignments: [] }));
    expect(air).toContain('every layer is read from disk for every token');
    expect(air).not.toContain('GGUF');
  });

  it('leaves the layer split to engines that decide it themselves', () => {
    const mac = assignment({ device: 'MLX', layers: [], holds_output: false, weight_bytes: 364904448 });
    expect(layersText(plan({ engine: 'exo', assignments: [mac, { ...mac, node_id: 'mac2' }] }), mac)).toBe('decided by EXO');
    const gpu = assignment({ layers: [] });
    expect(layersText(plan({ engine: 'vllm', assignments: [gpu] }), gpu)).toBe('all');
    expect(layersText(plan(), assignment())).toBe('4 (0–3)');
    expect(layersText(plan(), assignment({ layers: [], weight_bytes: 0 }))).toBe('0');
    const air = assignment({ layers: [], holds_output: false, weight_bytes: 1 << 30 });
    expect(layersText(plan({ engine: 'airllm', assignments: [air, { ...air, device: 'CUDA1' }] }), air)).toBe('decided by AirLLM');
  });

  it('counts layers itself for llama.cpp, even a device that holds only the output head', () => {
    // planner.layout_from_units on a measured-profile replay: CUDA1 gets 0 layers, holds_output, output-head bytes
    const head = assignment({ device: 'CUDA1', layers: [], holds_output: true, weight_bytes: 638582784 });
    const p = plan({ assignments: [assignment(), head] });
    expect(layersText(p, head)).toBe('0');
    expect(layersText(plan({ assignments: [head] }), head)).toBe('0');
  });

  it('labels measured profiles and estimates', () => {
    expect(planSourceLabel(plan())).toBe('Estimate');
    expect(planSourceLabel(plan({ heuristic: false, profile_id: 'p1' }))).toBe('Measured profile');
  });

  it('counts CPU layers', () => {
    expect(cpuLayersText(0)).toBeNull();
    expect(cpuLayersText(1)).toBe("1 layer on this PC's CPU");
    expect(cpuLayersText(6)).toBe("6 layers on this PC's CPU");
  });

  it('says where the overflow went', () => {
    expect(overflowText(plan())).toBe("none: every layer is on this machine's GPU");
    const ram = plan({ tier: 'gpu_ram', cpu_layers: 6, assignments: [assignment({ device: 'CPU' }), assignment()] });
    expect(overflowText(ram)).toBe("6 layers on this PC's CPU (from RAM)");
    const peers = plan({ assignments: [assignment({ node_id: 'peer1', device: 'RPC0', endpoint: '10.0.0.5:50052' }), assignment()] });
    expect(overflowText(peers)).toBe("friends' GPUs on 1 machine");
    const disk = plan({ tier: 'gpu_disk', capacity_mode: true, cpu_layers: 2 });
    expect(overflowText(disk)).toBe("2 layers on this PC's CPU (paged from disk)");
    expect(overflowText(plan({ engine: 'vllm' }))).toBeNull();
    expect(overflowText(plan({ assignments: [] }))).toBeNull();
  });
});
