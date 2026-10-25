// The advanced run options in "Choose what to run" and the PlanRequest they produce.

import type { EngineId, EnginesStatus, ModelArtifact, Objective, PlanRequest, PlanStrategy } from '../api/types';
import { ENGINE_LABELS, OBJECTIVE_LABELS, engineChoiceState } from './engines';

export type EngineChoice = EngineId | 'auto';

export interface RunOptions {
  engine: EngineChoice;
  objective: Objective;
  useProfiles: boolean;
  allowRamTier: boolean;
  allowCapacityMode: boolean;
}

/** Same defaults as the gateway's PlanRequest. */
export const DEFAULT_RUN_OPTIONS: RunOptions = {
  engine: 'auto',
  objective: 'interactive',
  useProfiles: true,
  allowRamTier: true,
  allowCapacityMode: true,
};

/** A forced engine that this model or host cannot use falls back to automatic instead of failing later. */
export function effectiveEngine(
  choice: EngineChoice,
  status: EnginesStatus | undefined,
  art: ModelArtifact | null,
): EngineChoice {
  if (choice === 'auto') return 'auto';
  return engineChoiceState(choice, status, art).enabled ? choice : 'auto';
}

export function planRequestFor(
  modelId: string,
  strategy: PlanStrategy,
  nodeIds: string[],
  ctxSize: number,
  opts: RunOptions,
): PlanRequest {
  return {
    model_id: modelId,
    node_ids: strategy === 'local_only' ? null : nodeIds,
    strategy,
    ctx_size: ctxSize,
    engine: opts.engine === 'auto' ? null : opts.engine,
    objective: opts.objective,
    use_profiles: opts.useProfiles,
    allow_ram_tier: opts.allowRamTier,
    allow_capacity_mode: opts.allowCapacityMode,
  };
}

/** What differs from the defaults, for the collapsed "Advanced" line. Empty when nothing changed. */
export function advancedSummary(opts: RunOptions): string[] {
  const out: string[] = [];
  if (opts.engine !== 'auto') out.push(`engine ${ENGINE_LABELS[opts.engine]}`);
  if (opts.objective !== DEFAULT_RUN_OPTIONS.objective) out.push(OBJECTIVE_LABELS[opts.objective].toLowerCase());
  if (!opts.useProfiles) out.push('no measured profiles');
  if (!opts.allowRamTier) out.push('no host RAM');
  if (!opts.allowCapacityMode) out.push('no capacity mode');
  return out;
}
