// Tuner and measured-profile helpers. plan_fields arrives as a loose dict, so every read is narrowed here.

import type {
  HcEvent,
  MeasuredProfile,
  PlanRequest,
  TuneCandidateStatus,
  TuneRequest,
  TuneResult,
  TuneStatus,
} from '../api/types';
import { ENGINE_LABELS } from './engines';
import { UNAVAILABLE } from './format';

export const TUNE_STATUS_TEXT: Record<TuneStatus, string> = {
  running: 'Measuring',
  done: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export const CANDIDATE_STATUS_TEXT: Record<TuneCandidateStatus, string> = {
  pending: 'waiting',
  running: 'running',
  done: 'measured',
  failed: 'failed',
  skipped: 'skipped',
};

function finite(n: number | null | undefined): n is number {
  return typeof n === 'number' && Number.isFinite(n);
}

/** Score is the estimated seconds for a typical turn (TTFT + N_g / decode tok/s); lower is better. */
export function formatScore(score: number | null | undefined): string {
  return finite(score) ? `${score.toFixed(2)} s` : UNAVAILABLE;
}

export function formatSeconds(s: number | null | undefined): string {
  return finite(s) ? `${s.toFixed(1)} s` : UNAVAILABLE;
}

function newestFirst(a: TuneResult, b: TuneResult): number {
  return Date.parse(b.created_at) - Date.parse(a.created_at);
}

/** The tune to show: a running one first, else the newest for this model, else the newest of all. */
export function currentTune(tunes: TuneResult[] | undefined, modelId: string | null): TuneResult | null {
  if (!tunes || tunes.length === 0) return null;
  const sorted = [...tunes].sort(newestFirst);
  return (
    sorted.find((t) => t.status === 'running') ??
    sorted.find((t) => t.request.model_id === modelId) ??
    sorted[0] ??
    null
  );
}

/** The candidate the tuner picked, only once it was actually measured. */
export function winnerIndex(t: TuneResult): number | null {
  const i = t.best_index;
  if (i === null || i < 0 || i >= t.candidates.length) return null;
  return t.candidates[i]?.status === 'done' ? i : null;
}

/**
 * Tune what the owner selected in "Choose what to run", within the same limits their plans use, so the saved
 * profile is one those plans accept. "This machine only" means no friends at all. Automatic engine means
 * llama.cpp: tuning measures llama.cpp splits. Unset limits mean the PlanRequest defaults (allowed).
 */
export function tuneRequestFrom(
  sel: PlanRequest,
  opts: { maxCandidates: number; tuneFlags: boolean },
): TuneRequest {
  return {
    model_id: sel.model_id,
    node_ids: sel.strategy === 'local_only' ? [] : sel.node_ids,
    ctx_size: sel.ctx_size,
    objective: sel.objective ?? 'interactive',
    max_candidates: opts.maxCandidates,
    tune_flags: opts.tuneFlags,
    engine: sel.engine ?? 'llama_rpc',
    strategy: sel.strategy,
    allow_ram_tier: sel.allow_ram_tier ?? true,
    allow_capacity_mode: sel.allow_capacity_mode ?? true,
  };
}

export interface ProfileReuse {
  /** True only when the planner will replay the saved profile. */
  reused: boolean;
  heading: string;
  text: string;
}

const NOT_REUSED = 'Saved, but not used by later plans';

/**
 * Whether later plans can replay the profile a finished tune saved. The planner replays only llama.cpp profiles
 * of the same model (selector._with_profile), so a winner measured on another engine or variant is not reused.
 * Without a measured winner neither is claimed.
 */
export function profileReuse(t: TuneResult): ProfileReuse | null {
  if (t.status !== 'done' || !t.profile_id) return null;
  const i = winnerIndex(t);
  const winner = i === null ? null : t.candidates[i]?.plan;
  if (!winner) {
    return {
      reused: false,
      heading: 'Saved, reuse unconfirmed',
      text: 'the measured winner is unavailable, so whether later plans reuse it is unknown',
    };
  }
  if (winner.engine !== 'llama_rpc') {
    const text = `measured with ${ENGINE_LABELS[winner.engine]}; plans reuse only llama.cpp profiles`;
    return { reused: false, heading: NOT_REUSED, text };
  }
  if (winner.model_id !== t.request.model_id) {
    return {
      reused: false,
      heading: NOT_REUSED,
      text: `measured ${winner.model_id}, not ${t.request.model_id}; plans for this model keep estimates`,
    };
  }
  return { reused: true, heading: 'Saved as a measured profile', text: 'later plans for this model and these machines reuse it' };
}

/** tune.* events from the activity stream (tune.progress and friends). */
export function isTuneEvent(e: HcEvent): boolean {
  return e.type.startsWith('tune.');
}

// ---------------------------------------------------------------- measured profiles

function stringField(rec: Record<string, unknown>, key: string): string | null {
  const v = rec[key];
  return typeof v === 'string' ? v : null;
}

function numberField(rec: Record<string, unknown>, key: string): number | null {
  const v = rec[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(rec: Record<string, unknown>, key: string): string[] {
  const v = rec[key];
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

function flagList(rec: Record<string, unknown>, key: string): string[] {
  const v = rec[key];
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return [];
  return Object.entries(v as Record<string, unknown>)
    .filter(([, x]) => typeof x === 'string' || typeof x === 'number')
    .map(([k, x]) => `${k}=${String(x)}`);
}

export interface ProfileSummary {
  engine: string | null;
  tier: string | null;
  devices: string[];
  cpuLayers: number | null;
  flags: string[];
}

/** The parts of a profile's saved plan fields the table shows; anything malformed reads as missing. */
export function profileSummary(p: MeasuredProfile): ProfileSummary {
  const f = p.plan_fields !== null && typeof p.plan_fields === 'object' ? p.plan_fields : {};
  return {
    engine: stringField(f, 'engine'),
    tier: stringField(f, 'tier'),
    devices: stringList(f, 'device_list'),
    cpuLayers: numberField(f, 'cpu_layers'),
    flags: flagList(f, 'engine_flags'),
  };
}
