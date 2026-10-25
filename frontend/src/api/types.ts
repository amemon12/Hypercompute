// Mirrors backend/hypercompute/schemas.py (the contract) for the fields the dashboard reads.
// Sizes are bytes, times are ISO-8601 UTC strings, and an unavailable measurement is null (never 0).

export type Accelerator = 'cuda' | 'metal' | 'vulkan' | 'cpu' | 'rocm' | 'unknown';

/** One model instance has exactly one engine owner; engines never share a model's layers. */
export type EngineId = 'llama_rpc' | 'exo' | 'airllm' | 'vllm';

/**
 * gpu: every layer on accelerators; gpu_ram: the first layers run on the host CPU from RAM;
 * gpu_disk: like gpu_ram but paged from the GGUF on disk; disk: AirLLM reads every layer from disk per token.
 */
export type Tier = 'gpu' | 'gpu_ram' | 'gpu_disk' | 'disk';

/** The order the dashboard lists engines in. */
export const ENGINE_IDS: readonly EngineId[] = ['llama_rpc', 'vllm', 'exo', 'airllm'];

export function isEngineId(value: unknown): value is EngineId {
  return typeof value === 'string' && (ENGINE_IDS as readonly string[]).includes(value);
}

// ---------------------------------------------------------------- inventory

export interface Diagnostic {
  level: 'info' | 'warning' | 'error';
  code: string;
  message: string;
  fix: string | null;
}

export interface OSInfo {
  system: 'windows' | 'macos' | 'linux' | 'other';
  release: string;
  version: string;
  machine: string;
}

export interface CpuInfo {
  model: string | null;
  physical_cores: number | null;
  logical_cores: number | null;
}

export interface MemoryPool {
  id: string;
  kind: 'system_ram' | 'unified' | 'discrete_vram';
  total_bytes: number | null;
  free_bytes: number | null;
  measured_at: string | null;
  source: string;
  note: string | null;
}

export interface Device {
  id: string;
  kind: 'cpu' | 'gpu' | 'igpu' | 'accel';
  name: string;
  vendor: string | null;
  pool_id: string;
  backend: Accelerator;
  total_bytes: number | null;
  free_bytes: number | null;
  driver_version: string | null;
  compute_capability: string | null;
  source: string;
}

export interface EngineDevice {
  name: string;
  description: string;
  total_bytes: number | null;
  free_bytes: number | null;
}

export interface EngineInfo {
  engine: 'llama_rpc';
  installed: boolean;
  release: string | null;
  build_number: number | null;
  commit: string | null;
  rpc_proto: string | null;
  variant: string | null;
  accelerator: Accelerator;
  bin_dir: string | null;
  smoke: 'passed' | 'failed' | 'not_run';
  smoke_detail: string | null;
  devices: EngineDevice[];
}

/** An optional engine installed in its own environment (AirLLM, vLLM, EXO), never in the gateway's. */
export interface ExtraEngineInfo {
  engine: EngineId;
  installed: boolean;
  status: 'not_installed' | 'installed' | 'running' | 'failed' | 'unsupported';
  version: string | null;
  endpoint: string | null;
  detail: string | null;
}

export interface NetworkInterface {
  name: string;
  ipv4: string[];
  is_up: boolean;
  speed_mbps: number | null;
  kind: 'wifi' | 'ethernet' | 'virtual' | 'loopback' | 'other';
}

export interface HostInventory {
  hostname: string;
  os: OSInfo;
  cpu: CpuInfo;
  memory_pools: MemoryPool[];
  devices: Device[];
  disk_free_bytes: number | null;
  network: NetworkInterface[];
  engine: EngineInfo;
  collected_at: string;
  diagnostics: Diagnostic[];
  extra_engines: ExtraEngineInfo[];
}

// ---------------------------------------------------------------- nodes

export interface ContributionLimits {
  devices: string[];
  max_bytes: number;
  reserve_bytes: number;
  enforcement: 'soft_admission';
  cache_enabled: boolean;
}

export interface WorkerEndpoint {
  host: string;
  port: number;
}

export type WorkerState = 'starting' | 'running' | 'exited' | 'stopped' | 'paused';

export type NodeState = 'verifying' | 'eligible' | 'ineligible' | 'stale' | 'left' | 'removed';

export interface VerificationResult {
  checked_at: string;
  reachable: boolean;
  engine_match: boolean;
  rpc_devices: EngineDevice[];
  detail: string;
  fix: string | null;
}

export interface NodeView {
  id: string;
  display_name: string;
  is_local: boolean;
  session_id: string | null;
  state: NodeState;
  state_reason: string | null;
  joined_at: string;
  last_heartbeat_at: string | null;
  heartbeat_age_s: number | null;
  inventory: HostInventory;
  contribution: ContributionLimits;
  worker: WorkerEndpoint | null;
  worker_state: WorkerState | null;
  worker_cpu_seconds: number | null;
  verification: VerificationResult | null;
  deployments: string[];
}

// ---------------------------------------------------------------- sessions

export interface InviteView {
  code: string;
  expires_at: string;
  join_address: string;
  join_command: string;
}

export interface SessionView {
  id: string;
  name: string;
  status: 'active' | 'closed';
  created_at: string;
  lan_addresses: string[];
  gateway_port: number;
  invite: InviteView | null;
}

export interface CreateSessionRequest {
  name?: string;
}

// ---------------------------------------------------------------- owner / health

export interface OwnerBootstrap {
  owner_token: string;
  version: string;
  gateway_port: number;
  lan_addresses: string[];
}

/** All that GET /api/health tells a caller that is not the owner on this machine. */
export interface PublicEngineInfo {
  release: string | null;
  rpc_proto: string | null;
}

export interface Health {
  status: 'ok';
  version: string;
  /** Full EngineInfo for the owner (loopback); any other caller gets only {release, rpc_proto}. */
  engine: EngineInfo | PublicEngineInfo;
  time?: string;
}

export function isFullEngineInfo(engine: EngineInfo | PublicEngineInfo): engine is EngineInfo {
  return 'installed' in engine && Array.isArray(engine.devices);
}

// ---------------------------------------------------------------- models

export interface ModelFile {
  name: string;
  url: string;
  sha256: string;
  size_bytes: number;
  /** Relative path inside the artifact dir (HF repos keep subdirectories). */
  path: string | null;
}

export interface GgufSummary {
  architecture: string | null;
  name: string | null;
  file_type: string | null;
  n_layer: number | null;
  n_embd: number | null;
  n_head: number | null;
  n_head_kv: number | null;
  context_length: number | null;
  vocab_size: number | null;
  tensor_count: number;
  total_tensor_bytes: number;
  layer_bytes: number[];
  token_embd_bytes: number;
  output_bytes: number;
  output_tied: boolean;
  other_bytes: number;
  chat_template_present: boolean;
  key_length: number | null;
  value_length: number | null;
}

/** gguf: llama.cpp; hf: safetensors directory (AirLLM, vLLM); mlx: EXO. */
export type ModelFormat = 'gguf' | 'hf' | 'mlx';

export interface ModelArtifact {
  id: string;
  display_name: string;
  family: string;
  params_b: number;
  quantization: string;
  format: ModelFormat;
  /** Empty means the default engines for the format (artifactEngines in lib/engines.ts). */
  engines: EngineId[];
  /** Groups the GGUF / HF / MLX variants of one model. */
  logical_id: string | null;
  /** For artifacts an engine downloads itself (EXO/MLX), where files is empty. */
  size_bytes: number | null;
  repo: string;
  revision: string;
  license: string;
  files: ModelFile[];
  default_ctx: number;
  max_output_tokens: number;
  role: 'plumbing' | 'capacity_candidate' | 'standard';
  notes: string | null;
}

/** engine_managed: the engine fetches the weights itself on every node (EXO/MLX); nothing to download here. */
export type ModelStatus = 'not_downloaded' | 'downloading' | 'downloaded' | 'corrupt' | 'error' | 'engine_managed';

export interface ModelView {
  artifact: ModelArtifact;
  status: ModelStatus;
  local_path: string | null;
  downloaded_bytes: number;
  total_bytes: number;
  verified_sha256: boolean;
  gguf: GgufSummary | null;
  error: string | null;
}

// ---------------------------------------------------------------- planning

export type PlanStrategy = 'auto' | 'all_selected' | 'fastest_fit' | 'local_only';

/** interactive: one user, lowest latency; throughput: many concurrent requests. */
export type Objective = 'interactive' | 'throughput';

export interface PlanRequest {
  model_id: string;
  node_ids: string[] | null;
  strategy: PlanStrategy;
  ctx_size: number;
  allow_ram_tier?: boolean;
  allow_capacity_mode?: boolean;
  /** Force an engine; null lets the selector choose. */
  engine?: EngineId | null;
  objective?: Objective;
  use_profiles?: boolean;
}

export interface DeviceAssignment {
  node_id: string;
  device: string;
  endpoint: string | null;
  pool_id: string;
  layers: number[];
  holds_output: boolean;
  weight_bytes: number;
  kv_bytes: number;
  compute_bytes_est: number;
  overhead_bytes_est: number;
  total_bytes_est: number;
  budget_bytes: number;
  fits: boolean;
}

export interface CandidateSummary {
  label: string;
  devices: string[];
  feasible: boolean;
  reason: string;
}

export type FitLabel = 'estimated_fit' | 'does_not_fit' | 'fit_unknown';

export interface Plan {
  id: string;
  model_id: string;
  strategy: PlanStrategy;
  ctx_size: number;
  feasible: boolean;
  fit_label: FitLabel;
  heuristic: boolean;
  assignments: DeviceAssignment[];
  device_list: string[];
  tensor_split: number[];
  rpc_endpoints: string[];
  cpu_resident_bytes: number;
  reasons: string[];
  warnings: string[];
  candidates: CandidateSummary[];
  created_at: string;
  engine: EngineId;
  tier: Tier;
  /** Layers 0..cpu_layers-1 run on the host CPU. */
  cpu_layers: number;
  /** -ngl value when cpu_layers > 0 (offloaded layers + output head). */
  n_gpu_units: number | null;
  /** llama.cpp: flash_attn, ubatch, batch, threads. */
  engine_flags: Record<string, string | number>;
  /** Engine-specific plan data (vLLM args, EXO placement preview, AirLLM shard dir). */
  engine_options: Record<string, unknown>;
  /** Measured profile this plan reuses (then heuristic is false). */
  profile_id: string | null;
  /** Disk-backed tier: expect seconds or more per token. */
  capacity_mode: boolean;
}

// ---------------------------------------------------------------- deployments

export type DeploymentState =
  | 'planned'
  | 'reserving'
  | 'preparing'
  | 'loading'
  | 'warming'
  | 'ready'
  | 'draining'
  | 'stopped'
  | 'failed';

export const TERMINAL_STATES: ReadonlySet<DeploymentState> = new Set<DeploymentState>([
  'stopped',
  'failed',
]);

export interface StateTransition {
  state: DeploymentState;
  at: string;
  detail: string | null;
}

export interface ObservedDevice {
  name: string;
  model_bytes: number;
  kv_bytes: number;
  compute_bytes: number;
  layers_est: number | null;
}

export interface ObservedPlacement {
  offloaded: string | null;
  devices: ObservedDevice[];
  evidence_lines: string[];
  parsed_at: string;
}

export interface Deployment {
  id: string;
  plan: Plan;
  state: DeploymentState;
  history: StateTransition[];
  error: string | null;
  cleanup: string | null;
  progress: string | null;
  engine_args: string[];
  upstream_port: number | null;
  pid: number | null;
  placement: ObservedPlacement | null;
  participation_verified: boolean;
  participation_detail: string | null;
  created_at: string;
  idempotency_key: string | null;
  purpose: DeploymentPurpose;
}

/** "tune": a tuner trial, never offered to chat or benchmarks. */
export type DeploymentPurpose = 'run' | 'tune';

export interface CreateDeploymentRequest {
  plan_id?: string;
  plan?: PlanRequest;
  purpose?: DeploymentPurpose;
}

// ---------------------------------------------------------------- runs / benchmarks

export interface EngineTimings {
  prompt_n: number | null;
  prompt_ms: number | null;
  predicted_n: number | null;
  predicted_ms: number | null;
  predicted_per_second: number | null;
  prompt_per_second: number | null;
  cache_n: number | null;
}

export interface RunRecord {
  id: string;
  attempt_id: string;
  deployment_id: string | null;
  kind: 'chat' | 'benchmark' | 'warmup';
  status: 'ok' | 'error' | 'cancelled' | 'timeout';
  simulated: boolean;
  model: Record<string, unknown>;
  engine: Record<string, unknown>;
  placement: Record<string, unknown>[];
  transport: string;
  ctx_size: number | null;
  stream: boolean;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  ttft_ms: number | null;
  total_ms: number | null;
  timings: EngineTimings | null;
  finish_reason: string | null;
  sampling: Record<string, unknown>;
  cache_state: 'warm' | 'cold' | 'unknown';
  output_text: string | null;
  error: string | null;
  started_at: string;
  benchmark_id: string | null;
  prompt_label: string | null;
}

export interface BenchmarkRequest {
  deployment_id?: string | null;
  prompt_set?: 'short' | 'long' | 'both';
  repetitions?: number;
  warmup?: number;
  max_tokens?: number;
  label?: string | null;
}

export interface StatSummary {
  n: number;
  median: number | null;
  min: number | null;
  max: number | null;
}

export interface BenchmarkSummary {
  prompt_label: string;
  ttft_ms: StatSummary;
  decode_tokens_per_s: StatSummary;
  prompt_tokens_per_s: StatSummary;
  completion_tokens: StatSummary;
  failures: number;
}

export interface BenchmarkResult {
  id: string;
  deployment_id: string;
  label: string | null;
  status: 'running' | 'done' | 'failed';
  request: Required<BenchmarkRequest>;
  run_ids: string[];
  summaries: BenchmarkSummary[];
  placement_label: string;
  created_at: string;
  error: string | null;
}

// ---------------------------------------------------------------- engines

/** GET /api/engines. `available` lists the engines this host has a working adapter for. */
export interface EnginesStatus {
  available: string[];
  llama_rpc: EngineInfo;
  host: ExtraEngineInfo[];
  nodes: Record<string, ExtraEngineInfo[]>;
}

// ---------------------------------------------------------------- tuning / measured profiles

export interface TuneRequest {
  model_id: string;
  node_ids?: string[] | null;
  ctx_size?: number;
  objective?: Objective;
  max_candidates?: number;
  /** Also try flash attention and micro-batch size, not only the layer split. */
  tune_flags?: boolean;
  decode_tokens?: number;
  // The owner's own plan limits: the tune measures only configurations their plans would accept.
  /** Tuning measures llama.cpp splits; null means llama_rpc. */
  engine?: EngineId | null;
  strategy?: PlanStrategy;
  allow_ram_tier?: boolean;
  allow_capacity_mode?: boolean;
}

export type TuneCandidateStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped';

export interface TuneCandidate {
  label: string;
  plan: Plan;
  status: TuneCandidateStatus;
  deployment_id: string | null;
  load_s: number | null;
  /** Short prompt, prompt cache off. */
  ttft_ms: number | null;
  decode_tokens_per_s: number | null;
  /** Long prompt, prompt cache off. */
  prefill_tokens_per_s: number | null;
  /** Lower is better: estimated seconds for a typical turn. */
  score: number | null;
  run_ids: string[];
  error: string | null;
}

export type TuneStatus = 'running' | 'done' | 'failed' | 'cancelled';

export interface TuneResult {
  id: string;
  request: Required<TuneRequest>;
  status: TuneStatus;
  candidates: TuneCandidate[];
  best_index: number | null;
  profile_id: string | null;
  created_at: string;
  error: string | null;
  /** Why the winner was not saved as a measured profile, when it was not. */
  note: string | null;
}

/** A configuration that was actually run and measured, reused by the planner for the same setup. */
export interface MeasuredProfile {
  id: string;
  model_id: string;
  /** model + ctx + sorted (node, device) set the plan used. */
  signature: string;
  objective: Objective;
  /** device_list, tensor_split, cpu_layers, n_gpu_units, engine_flags, rpc_endpoints, tier, engine. */
  plan_fields: Record<string, unknown>;
  ttft_ms: number | null;
  decode_tokens_per_s: number | null;
  prefill_tokens_per_s: number | null;
  score: number;
  tune_id: string | null;
  run_ids: string[];
  created_at: string;
}

// ---------------------------------------------------------------- events

export interface HcEvent {
  seq: number;
  at: string;
  type: string;
  message: string;
  data: Record<string, unknown>;
}

// ---------------------------------------------------------------- errors / chat

export interface ApiErrorBody {
  detail: string;
  code?: string;
  fix?: string | null;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

/** One `chat.completion.chunk` from llama-server, passed through the gateway unchanged. */
export interface ChatChunk {
  id?: string;
  model?: string;
  object?: string;
  choices?: Array<{
    index?: number;
    finish_reason?: string | null;
    delta?: { role?: string; content?: string | null; reasoning_content?: string | null };
  }>;
  usage?: ChatUsage;
  timings?: Partial<EngineTimings>;
  error?: { message?: string; type?: string; code?: number | string };
}
