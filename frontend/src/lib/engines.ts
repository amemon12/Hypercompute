// Engines, tiers and capacity mode: labels and the small decisions the dashboard makes from the contract
// (docs/ENGINE_TIERS.md). Nothing here guesses at a field the gateway does not send.

import {
  isEngineId,
  type DeviceAssignment,
  type EngineId,
  type EngineInfo,
  type EnginesStatus,
  type ExtraEngineInfo,
  type ModelArtifact,
  type ModelFormat,
  type ModelView,
  type Objective,
  type Plan,
  type Tier,
} from '../api/types';
import { formatLayers } from './format';

export const ENGINE_LABELS: Record<EngineId, string> = {
  llama_rpc: 'llama.cpp',
  vllm: 'vLLM',
  exo: 'EXO',
  airllm: 'AirLLM',
};

export const ENGINE_HINTS: Record<EngineId, string> = {
  llama_rpc: 'GGUF models on this PC, its RAM and friends over the network.',
  vllm: 'One NVIDIA GPU on Linux or WSL2. Best when many people use it at once.',
  exo: 'Macs only (MLX). Every chosen machine must be a Mac running EXO.',
  airllm: 'Capacity mode: reads every layer from disk for every token. Very slow.',
};

/** Engine picker order: llama.cpp, vLLM, EXO, AirLLM. */
export const ENGINE_CHOICES: readonly EngineId[] = ['llama_rpc', 'vllm', 'exo', 'airllm'];

export const TIER_LABELS: Record<Tier, string> = {
  gpu: 'All on GPUs',
  gpu_ram: "GPUs + this PC's RAM",
  gpu_disk: 'GPUs + disk paging',
  disk: 'Disk (capacity mode)',
};

export const OBJECTIVE_LABELS: Record<Objective, string> = {
  interactive: 'Single user',
  throughput: 'Many users',
};

export const OBJECTIVE_HINTS: Record<Objective, string> = {
  interactive: 'Fastest replies for one conversation at a time.',
  throughput: 'Most tokens per second in total when several people send requests at once.',
};

export const FORMAT_LABELS: Record<ModelFormat, string> = {
  gguf: 'GGUF',
  hf: 'safetensors',
  mlx: 'MLX',
};

const DEFAULT_ENGINES: Record<ModelFormat, EngineId[]> = {
  gguf: ['llama_rpc'],
  hf: ['airllm', 'vllm'],
  mlx: ['exo'],
};

/** Engines that can serve an artifact: the manifest's list, else the default for its format (adapters/base.py). */
export function artifactEngines(art: ModelArtifact): EngineId[] {
  return art.engines.length > 0 ? [...art.engines] : [...DEFAULT_ENGINES[art.format]];
}

/** A model can be planned once its files are here, or when its engine fetches them itself (EXO). */
export function isModelUsable(m: ModelView): boolean {
  return m.status === 'downloaded' || m.status === 'engine_managed';
}

/** Total bytes: download progress total, else the file list, else the manifest size (engine-managed models). */
export function modelSizeBytes(m: ModelView): number | null {
  if (m.total_bytes > 0) return m.total_bytes;
  const sum = m.artifact.files.reduce((acc, f) => acc + f.size_bytes, 0);
  if (sum > 0) return sum;
  return m.artifact.size_bytes ?? null;
}

/** What replaces the download button for a model the engine fetches on each machine. */
export function engineManagedText(art: ModelArtifact): string {
  return artifactEngines(art).includes('exo')
    ? 'downloaded by EXO on each Mac'
    : 'downloaded by the engine on each machine';
}

/** The CLI command that installs an engine on the machine it runs on (cli.py: engine install --engine). */
export function engineInstallCommand(engine: EngineId): string {
  return engine === 'llama_rpc' ? 'uv run hypercompute engine install' : `uv run hypercompute engine install --engine ${engine}`;
}

export interface EngineChoiceState {
  enabled: boolean;
  reason: string | null;
}

function hostStatusReason(engine: EngineId, status: EnginesStatus): string {
  const info = status.host.find((e) => e.engine === engine);
  const detail = info?.detail ? ` (${info.detail})` : '';
  const cmd = engineInstallCommand(engine);
  switch (info?.status) {
    case 'unsupported':
      return `not supported on this computer${detail}`;
    case 'installed':
    case 'running':
      // Adapters are built when the gateway starts, so a fresh install needs a restart.
      return `installed, but the gateway has not loaded it${detail}. Restart the gateway.`;
    case 'failed':
      return `failed its check on this computer${detail}. Reinstall: ${cmd} --force`;
    default:
      return `not installed on this computer${detail}. Install: ${cmd}`;
  }
}

/**
 * Whether the engine picker may offer `engine` for this model. Unknown availability (engines not loaded yet)
 * stays enabled: the gateway then answers with engine_unavailable and a fix.
 */
export function engineChoiceState(
  engine: EngineId,
  status: EnginesStatus | undefined,
  art: ModelArtifact | null,
): EngineChoiceState {
  if (art) {
    const serving = artifactEngines(art);
    if (!serving.includes(engine)) {
      const names = serving.map((e) => ENGINE_LABELS[e]).join(' or ');
      return { enabled: false, reason: `not for this model (${FORMAT_LABELS[art.format]} files run on ${names})` };
    }
  }
  if (status && !status.available.includes(engine)) return { enabled: false, reason: hostStatusReason(engine, status) };
  return { enabled: true, reason: null };
}

/** Engines the host reports as usable, ignoring names this dashboard does not know. */
export function availableEngines(status: EnginesStatus | undefined): EngineId[] {
  return (status?.available ?? []).filter(isEngineId);
}

export type Tone = 'ok' | 'warn' | 'bad' | 'neutral';

export interface StatusLine {
  text: string;
  tone: Tone;
  /** Shown only when the engine still has to be installed on that machine. */
  install: string | null;
}

const EXTRA_STATUS: Record<ExtraEngineInfo['status'], { text: string; tone: Tone }> = {
  running: { text: 'running', tone: 'ok' },
  installed: { text: 'installed', tone: 'ok' },
  not_installed: { text: 'not installed', tone: 'neutral' },
  failed: { text: 'failed', tone: 'bad' },
  unsupported: { text: 'not supported here', tone: 'neutral' },
};

export function extraEngineStatus(info: ExtraEngineInfo | undefined, engine: EngineId): StatusLine {
  if (!info) return { text: 'not reported', tone: 'neutral', install: null };
  const s = EXTRA_STATUS[info.status];
  const install = info.status === 'not_installed' || info.status === 'failed' ? engineInstallCommand(engine) : null;
  return { ...s, install };
}

export function llamaEngineStatus(info: EngineInfo): StatusLine {
  if (!info.installed) return { text: 'not installed', tone: 'bad', install: engineInstallCommand('llama_rpc') };
  const release = info.release ?? 'release unavailable';
  if (info.smoke === 'failed') {
    return { text: `${release}, smoke test failed`, tone: 'bad', install: engineInstallCommand('llama_rpc') };
  }
  return { text: `${release} · ${info.accelerator}${info.smoke === 'passed' ? ', smoke test passed' : ''}`, tone: 'ok', install: null };
}

// ---------------------------------------------------------------- plan wording

/**
 * Disk-backed plans get a banner. The planner sets capacity_mode on llama.cpp gpu_disk plans too, so the tier
 * decides the wording: gpu_disk pages only the CPU layers from the GGUF; disk (AirLLM) reads every layer.
 */
export function capacityWarning(plan: Plan): string | null {
  if (plan.tier === 'gpu_disk') {
    const n = plan.cpu_layers;
    const count = n > 0 ? ` (${n} layer${n === 1 ? '' : 's'})` : '';
    return `Part of the model runs on this PC's CPU${count} and does not fit in free RAM, so those layers are ` +
      'paged from the GGUF file on disk while it runs; the GPU layers run from VRAM. Expect seconds or more per token.';
  }
  if (plan.tier === 'disk' || plan.capacity_mode) {
    return 'Capacity mode: every layer is read from disk for every token. Expect seconds to minutes per token. ' +
      'Use it only when nothing faster fits.';
  }
  return null;
}

/** Engines that place layers themselves, so their assignments carry weights but no layer list. */
const SELF_SPLIT_ENGINES: readonly EngineId[] = ['exo', 'vllm', 'airllm'];

/**
 * The Layers cell of an assignment. EXO picks each Mac's layer span itself when it places the model, so its
 * assignments carry weights but no layers; a single engine-owned device with no layer list holds them all.
 * llama.cpp layers always come from the plan: a device with none may still hold the output head (PlanView
 * adds "+ output layer").
 */
export function layersText(plan: Plan, a: DeviceAssignment): string {
  if (a.layers.length > 0 || a.weight_bytes <= 0 || !SELF_SPLIT_ENGINES.includes(plan.engine)) return formatLayers(a.layers);
  if (plan.engine === 'exo') return 'decided by EXO';
  return plan.assignments.length === 1 ? 'all' : `decided by ${ENGINE_LABELS[plan.engine]}`;
}

/** "Measured profile" when the plan reuses a measured configuration, else "Estimate". */
export function planSourceLabel(plan: Plan): string {
  if (plan.profile_id) return 'Measured profile';
  return plan.heuristic ? 'Estimate' : 'Measured';
}

export function cpuLayersText(n: number): string | null {
  if (n <= 0) return null;
  return `${n} layer${n === 1 ? '' : 's'} on this PC's CPU`;
}

/**
 * Where the layers that do not fit this machine's GPU went, read from the plan itself: CPU layers (tier
 * gpu_ram / gpu_disk) and assignments on friends' machines (an RPC endpoint). Null for other engines.
 */
export function overflowText(plan: Plan): string | null {
  if (plan.engine !== 'llama_rpc' || plan.assignments.length === 0) return null;
  const peers = new Set(plan.assignments.filter((a) => a.endpoint !== null).map((a) => a.node_id)).size;
  const parts: string[] = [];
  const cpu = cpuLayersText(plan.cpu_layers);
  if (cpu) parts.push(`${cpu} (${plan.tier === 'gpu_disk' ? 'paged from disk' : 'from RAM'})`);
  if (peers > 0) parts.push(`friends' GPUs on ${peers} machine${peers === 1 ? '' : 's'}`);
  return parts.length === 0 ? "none: every layer is on this machine's GPU" : parts.join(' and ');
}
