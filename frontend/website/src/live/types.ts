import type { ContributionLimits, HostInventory, Plan } from '../../../src/api/types';
export type { NodeView, ModelView, Deployment, SessionView, BenchmarkResult, MeasuredProfile, PlanRequest, HostInventory } from '../../../src/api/types';
export interface DecodeEstimate { ms_per_token: number; tokens_per_second: number; basis: 'uncalibrated'; assumptions: string[] }
export type LivePlan = Plan & { decode_estimate?: DecodeEstimate | null };
export interface ConnectorStatus {
  stage: 'idle' | 'starting' | 'verifying' | 'ready' | 'paused' | 'stopping' | 'error';
  state: string | null; detail: string | null; error: string | null; fix: string | null;
  node_id: string | null; inventory: HostInventory | null; contribution: ContributionLimits | null;
}
export interface ConnectorBootstrap { control_token: string; inventory: HostInventory | null; status: ConnectorStatus }
