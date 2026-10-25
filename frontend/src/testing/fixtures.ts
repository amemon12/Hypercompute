// Test builders that follow backend/hypercompute/schemas.py defaults. Used by *.test.ts(x) only.

import type {
  Deployment,
  DeviceAssignment,
  EngineInfo,
  EnginesStatus,
  ExtraEngineInfo,
  HostInventory,
  ModelArtifact,
  ModelView,
  NodeView,
  Plan,
  TuneCandidate,
  TuneResult,
} from '../api/types';

export function engineInfo(patch: Partial<EngineInfo> = {}): EngineInfo {
  return {
    engine: 'llama_rpc',
    installed: true,
    release: 'b11770',
    build_number: 11770,
    commit: 'abc1234',
    rpc_proto: '8.0.0',
    variant: 'windows-x64-cuda',
    accelerator: 'cuda',
    bin_dir: null,
    smoke: 'passed',
    smoke_detail: null,
    devices: [],
    ...patch,
  };
}

export function extraEngine(patch: Partial<ExtraEngineInfo> & Pick<ExtraEngineInfo, 'engine'>): ExtraEngineInfo {
  return { installed: false, status: 'not_installed', version: null, endpoint: null, detail: null, ...patch };
}

export function artifact(patch: Partial<ModelArtifact> = {}): ModelArtifact {
  return {
    id: 'qwen2.5-7b-q4',
    display_name: 'Qwen2.5 7B',
    family: 'qwen2.5',
    params_b: 7,
    quantization: 'Q4_K_M',
    format: 'gguf',
    engines: [],
    logical_id: null,
    size_bytes: null,
    repo: 'Qwen/Qwen2.5-7B-Instruct-GGUF',
    revision: 'main',
    license: 'apache-2.0',
    files: [],
    default_ctx: 4096,
    max_output_tokens: 1024,
    role: 'standard',
    notes: null,
    ...patch,
  };
}

export function modelView(patch: Partial<ModelView> = {}): ModelView {
  return {
    artifact: artifact(),
    status: 'downloaded',
    local_path: null,
    downloaded_bytes: 0,
    total_bytes: 0,
    verified_sha256: true,
    gguf: null,
    error: null,
    ...patch,
  };
}

export function assignment(patch: Partial<DeviceAssignment> = {}): DeviceAssignment {
  return {
    node_id: 'host',
    device: 'CUDA0',
    endpoint: null,
    pool_id: 'vram:CUDA0',
    layers: [0, 1, 2, 3],
    holds_output: true,
    weight_bytes: 1,
    kv_bytes: 1,
    compute_bytes_est: 1,
    overhead_bytes_est: 1,
    total_bytes_est: 4,
    budget_bytes: 8,
    fits: true,
    ...patch,
  };
}

export function plan(patch: Partial<Plan> = {}): Plan {
  return {
    id: 'plan1',
    model_id: 'qwen2.5-7b-q4',
    strategy: 'auto',
    ctx_size: 4096,
    feasible: true,
    fit_label: 'estimated_fit',
    heuristic: true,
    assignments: [assignment()],
    device_list: ['CUDA0'],
    tensor_split: [5],
    rpc_endpoints: [],
    cpu_resident_bytes: 0,
    reasons: [],
    warnings: [],
    candidates: [],
    created_at: '2026-10-20T12:00:00Z',
    engine: 'llama_rpc',
    tier: 'gpu',
    cpu_layers: 0,
    n_gpu_units: null,
    engine_flags: {},
    engine_options: {},
    profile_id: null,
    capacity_mode: false,
    ...patch,
  };
}

export function candidate(patch: Partial<TuneCandidate> = {}): TuneCandidate {
  return {
    label: 'GPU only',
    plan: plan(),
    status: 'pending',
    deployment_id: null,
    load_s: null,
    ttft_ms: null,
    decode_tokens_per_s: null,
    prefill_tokens_per_s: null,
    score: null,
    run_ids: [],
    error: null,
    ...patch,
  };
}

export function tuneResult(patch: Partial<TuneResult> = {}): TuneResult {
  return {
    id: 'tune1',
    request: {
      model_id: 'qwen2.5-7b-q4',
      node_ids: null,
      ctx_size: 4096,
      objective: 'interactive',
      max_candidates: 4,
      tune_flags: true,
      decode_tokens: 64,
      engine: null,
      strategy: 'auto',
      allow_ram_tier: true,
      allow_capacity_mode: false,
    },
    status: 'running',
    candidates: [],
    best_index: null,
    profile_id: null,
    created_at: '2026-10-20T12:00:00Z',
    error: null,
    note: null,
    ...patch,
  };
}

export function deployment(patch: Partial<Deployment> = {}): Deployment {
  return {
    id: 'dep1',
    plan: plan(),
    state: 'ready',
    history: [],
    error: null,
    cleanup: null,
    progress: null,
    engine_args: [],
    upstream_port: null,
    pid: null,
    placement: null,
    participation_verified: false,
    participation_detail: null,
    created_at: '2026-10-20T12:00:00Z',
    idempotency_key: null,
    purpose: 'run',
    ...patch,
  };
}

function inventory(engine: EngineInfo, extras: ExtraEngineInfo[]): HostInventory {
  return {
    hostname: 'h',
    os: { system: 'windows', release: '11', version: '10.0', machine: 'x86_64' },
    cpu: { model: null, physical_cores: null, logical_cores: null },
    memory_pools: [],
    devices: [],
    disk_free_bytes: null,
    network: [],
    engine,
    collected_at: '2026-10-20T12:00:00Z',
    diagnostics: [],
    extra_engines: extras,
  };
}

export function node(patch: Partial<NodeView> & { extras?: ExtraEngineInfo[]; llama?: EngineInfo } = {}): NodeView {
  const { extras = [], llama = engineInfo(), ...rest } = patch;
  return {
    id: 'peer1',
    display_name: 'Friend Mac',
    is_local: false,
    session_id: 's1',
    state: 'eligible',
    state_reason: null,
    joined_at: '2026-10-20T12:00:00Z',
    last_heartbeat_at: null,
    heartbeat_age_s: 1,
    inventory: inventory(llama, extras),
    contribution: { devices: [], max_bytes: 0, reserve_bytes: 0, enforcement: 'soft_admission', cache_enabled: false },
    worker: null,
    worker_state: null,
    worker_cpu_seconds: null,
    verification: null,
    deployments: [],
    ...rest,
  };
}

export function enginesStatus(patch: Partial<EnginesStatus> = {}): EnginesStatus {
  return {
    available: ['llama_rpc'],
    llama_rpc: engineInfo(),
    host: [
      extraEngine({ engine: 'airllm' }),
      extraEngine({ engine: 'vllm', status: 'unsupported', detail: 'vLLM needs Linux or WSL2' }),
      extraEngine({ engine: 'exo', status: 'unsupported', detail: 'EXO runs on macOS only' }),
    ],
    nodes: {},
    ...patch,
  };
}
