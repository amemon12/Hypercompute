// Gateway client. Owner GETs need only loopback + Host; every non-GET sends X-Hypercompute-Owner,
// whose value comes from GET /api/owner/bootstrap (fetched once, refetched if the gateway restarts).

import {
  EMPTY_STREAM_STATE,
  applyStreamItem,
  completionProblem,
  hasOutput,
  parseStreamData,
  type ChatStreamState,
} from './chatStream';
import { readSseStream } from './sse';
import type {
  BenchmarkRequest,
  BenchmarkResult,
  ChatMessage,
  Deployment,
  EnginesStatus,
  Health,
  HcEvent,
  MeasuredProfile,
  ModelView,
  NodeView,
  OwnerBootstrap,
  Plan,
  PlanRequest,
  RunRecord,
  SessionView,
  TuneRequest,
  TuneResult,
} from './types';

const OWNER_HEADER = 'X-Hypercompute-Owner';

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly fix: string | null;

  constructor(status: number, detail: string, code: string | null = null, fix: string | null = null) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.fix = fix;
  }
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  if (err instanceof Error) return new ApiError(0, err.message, 'client_error');
  return new ApiError(0, 'Unexpected error', 'client_error');
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Normalize {detail, code, fix}, FastAPI validation errors and llama-server {error:{...}}. */
export function errorFromBody(status: number, body: unknown, fallback: string): ApiError {
  if (isRecord(body)) {
    const { detail, code, fix, error } = body;
    if (typeof detail === 'string') {
      return new ApiError(
        status,
        detail,
        typeof code === 'string' ? code : null,
        typeof fix === 'string' ? fix : null,
      );
    }
    if (Array.isArray(detail)) {
      const parts = detail
        .map((d) => (isRecord(d) && typeof d.msg === 'string' ? d.msg : null))
        .filter((m): m is string => m !== null);
      return new ApiError(status, parts.join('; ') || fallback, 'validation_error');
    }
    if (isRecord(error) && typeof error.message === 'string') {
      const type = typeof error.type === 'string' ? error.type : null;
      return new ApiError(status, error.message, type);
    }
  }
  return new ApiError(status, fallback);
}

async function errorFromResponse(res: Response): Promise<ApiError> {
  const fallback = `Request failed with HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ''}`;
  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  return errorFromBody(res.status, body, fallback);
}

function unreachable(err: unknown): ApiError {
  const reason = err instanceof Error ? ` (${err.message})` : '';
  return new ApiError(
    0,
    `Cannot reach the Hypercompute gateway${reason}`,
    'gateway_unreachable',
    'Check that the gateway is running on this computer, then reload this page.',
  );
}

// ---------------------------------------------------------------- owner token

let bootstrapPromise: Promise<OwnerBootstrap> | null = null;

export function getBootstrap(): Promise<OwnerBootstrap> {
  if (!bootstrapPromise) {
    bootstrapPromise = request<OwnerBootstrap>('GET', '/api/owner/bootstrap').catch((err: unknown) => {
      bootstrapPromise = null;
      throw err;
    });
  }
  return bootstrapPromise;
}

interface RequestOptions {
  body?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

async function send(method: string, path: string, opts: RequestOptions): Promise<Response> {
  const headers: Record<string, string> = { Accept: 'application/json', ...opts.headers };
  if (opts.body !== undefined) headers['Content-Type'] = 'application/json';
  if (method !== 'GET') headers[OWNER_HEADER] = (await getBootstrap()).owner_token;
  try {
    return await fetch(path, {
      method,
      headers,
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
      signal: opts.signal,
      cache: 'no-store',
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err;
    throw unreachable(err);
  }
}

/** Send with the owner token; a 401/403 on a mutation refreshes the token once (gateway restart). */
async function sendAuthed(method: string, path: string, opts: RequestOptions): Promise<Response> {
  const res = await send(method, path, opts);
  if (method !== 'GET' && (res.status === 401 || res.status === 403)) {
    bootstrapPromise = null;
    return send(method, path, opts);
  }
  return res;
}

async function request<T>(method: string, path: string, opts: RequestOptions = {}): Promise<T> {
  const res = await sendAuthed(method, path, opts);
  if (!res.ok) throw await errorFromResponse(res);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  if (!text) return null as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(res.status, `The gateway returned a non-JSON response for ${path}`, 'bad_response');
  }
}

// ---------------------------------------------------------------- endpoints

export const api = {
  health: (): Promise<Health> => request('GET', '/api/health'),
  currentSession: (): Promise<SessionView | null> => request('GET', '/api/sessions/current'),
  createSession: (name?: string): Promise<SessionView> =>
    request('POST', '/api/sessions', { body: name ? { name } : {} }),
  newInvite: (sessionId: string): Promise<SessionView> =>
    request('POST', `/api/sessions/${encodeURIComponent(sessionId)}/invites`),
  closeSession: (sessionId: string): Promise<SessionView> =>
    request('DELETE', `/api/sessions/${encodeURIComponent(sessionId)}`),
  nodes: (): Promise<NodeView[]> => request('GET', '/api/nodes'),
  verifyNode: (nodeId: string): Promise<NodeView> =>
    request('POST', `/api/nodes/${encodeURIComponent(nodeId)}/verify`),
  removeNode: (nodeId: string): Promise<NodeView> =>
    request('POST', `/api/nodes/${encodeURIComponent(nodeId)}/leave`),
  models: (): Promise<ModelView[]> => request('GET', '/api/models'),
  downloadModel: (modelId: string): Promise<ModelView> =>
    request('POST', `/api/models/${encodeURIComponent(modelId)}/download`),
  createPlan: (req: PlanRequest): Promise<Plan> => request('POST', '/api/plans', { body: req }),
  deployments: (): Promise<Deployment[]> => request('GET', '/api/deployments'),
  createDeployment: (planId: string, idempotencyKey: string): Promise<Deployment> =>
    request('POST', '/api/deployments', {
      body: { plan_id: planId },
      headers: { 'Idempotency-Key': idempotencyKey },
    }),
  stopDeployment: (deploymentId: string): Promise<Deployment> =>
    request('DELETE', `/api/deployments/${encodeURIComponent(deploymentId)}`),
  runs: (limit = 50): Promise<RunRecord[]> => request('GET', `/api/runs?limit=${limit}`),
  benchmarks: (): Promise<BenchmarkResult[]> => request('GET', '/api/benchmarks'),
  startBenchmark: (req: BenchmarkRequest): Promise<BenchmarkResult> =>
    request('POST', '/api/benchmarks', { body: req }),
  engines: (): Promise<EnginesStatus> => request('GET', '/api/engines'),
  tunes: (): Promise<TuneResult[]> => request('GET', '/api/tune'),
  tune: (tuneId: string): Promise<TuneResult> => request('GET', `/api/tune/${encodeURIComponent(tuneId)}`),
  startTune: (req: TuneRequest): Promise<TuneResult> => request('POST', '/api/tune', { body: req }),
  cancelTune: (tuneId: string): Promise<TuneResult> => request('DELETE', `/api/tune/${encodeURIComponent(tuneId)}`),
  profiles: (modelId?: string): Promise<MeasuredProfile[]> =>
    request('GET', modelId ? `/api/profiles?model_id=${encodeURIComponent(modelId)}` : '/api/profiles'),
};

export function newIdempotencyKey(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = ((b[6] ?? 0) & 0x0f) | 0x40;
  b[8] = ((b[8] ?? 0) & 0x3f) | 0x80;
  const hex = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

// ---------------------------------------------------------------- event stream

export type EventStreamStatus = 'connecting' | 'live' | 'reconnecting';

/** Stable identity of an event. `seq` alone is not unique: the gateway restarts numbering at 1. */
export function eventKey(event: HcEvent): string {
  return `${event.seq}|${event.at}`;
}

/**
 * Returns a filter that accepts each event once. Keyed by seq and timestamp rather than "seq <= last
 * seen", because a restarted gateway numbers its events from 1 again and a seq check would then
 * drop every new event.
 */
export function createEventDeduper(capacity = 1000): (event: HcEvent) => boolean {
  const seen = new Set<string>();
  return (event) => {
    const key = eventKey(event);
    if (seen.has(key)) return false;
    seen.add(key);
    if (seen.size > capacity) {
      const oldest = seen.values().next().value;
      if (oldest !== undefined) seen.delete(oldest);
    }
    return true;
  };
}

/**
 * Subscribe to GET /api/events. On an error it reconnects (backoff up to 10 s) without `?since`, so
 * the gateway replays its buffer; already-shown events are dropped by `createEventDeduper`. Using
 * `?since=<last seq>` would miss everything after a gateway restart. Returns an unsubscribe function.
 */
export function subscribeEvents(
  onEvent: (event: HcEvent) => void,
  onStatus: (status: EventStreamStatus, detail?: string) => void,
): () => void {
  let source: EventSource | null = null;
  const isNew = createEventDeduper();
  let retryTimer: ReturnType<typeof setTimeout> | undefined;
  let backoffMs = 1000;
  let closed = false;

  const handle = (msg: MessageEvent<string>): void => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(msg.data);
    } catch {
      onStatus('live', 'Received an unreadable activity event');
      return;
    }
    if (!isRecord(parsed) || typeof parsed.seq !== 'number' || typeof parsed.type !== 'string') {
      onStatus('live', 'Received an activity event in an unexpected shape');
      return;
    }
    const event = parsed as unknown as HcEvent;
    if (!isNew(event)) return;
    onEvent(event);
  };

  const connect = (): void => {
    if (closed) return;
    const es = new EventSource('/api/events');
    source = es;
    es.addEventListener('hc', handle as EventListener);
    es.onmessage = handle;
    es.onopen = () => {
      backoffMs = 1000;
      onStatus('live');
    };
    es.onerror = () => {
      es.close();
      if (closed) return;
      onStatus('reconnecting', `Activity stream interrupted; retrying in ${Math.round(backoffMs / 1000)} s`);
      retryTimer = setTimeout(connect, backoffMs);
      backoffMs = Math.min(backoffMs * 2, 10_000);
    };
  };

  onStatus('connecting');
  connect();
  return () => {
    closed = true;
    clearTimeout(retryTimer);
    source?.close();
  };
}

// ---------------------------------------------------------------- streaming chat

export interface ChatOutcome {
  state: ChatStreamState;
  /** Request sent -> first content/reasoning delta, measured in this browser. */
  ttftMs: number | null;
  totalMs: number;
  status: 'completed' | 'stopped' | 'error';
  error: ApiError | null;
}

export interface ChatRequestOptions {
  model: string;
  messages: ChatMessage[];
  maxTokens?: number;
  signal: AbortSignal;
  onUpdate: (state: ChatStreamState, ttftMs: number | null) => void;
}

function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

export async function streamChat(opts: ChatRequestOptions): Promise<ChatOutcome> {
  const started = performance.now();
  let state = EMPTY_STREAM_STATE;
  let ttftMs: number | null = null;
  const finish = (status: ChatOutcome['status'], error: ApiError | null): ChatOutcome => ({
    state,
    ttftMs,
    totalMs: performance.now() - started,
    status,
    error,
  });

  try {
    const res = await sendAuthed('POST', '/v1/chat/completions', {
      body: {
        model: opts.model,
        messages: opts.messages,
        stream: true,
        stream_options: { include_usage: true },
        ...(opts.maxTokens ? { max_tokens: opts.maxTokens } : {}),
      },
      headers: { Accept: 'text/event-stream' },
      signal: opts.signal,
    });
    if (!res.ok) return finish('error', await errorFromResponse(res));
    if (!res.body) return finish('error', new ApiError(res.status, 'The response had no body to stream'));

    for await (const sse of readSseStream(res.body)) {
      const item = parseStreamData(sse.data);
      const hadOutput = hasOutput(state);
      state = applyStreamItem(state, item);
      if (!hadOutput && hasOutput(state)) ttftMs = performance.now() - started;
      opts.onUpdate(state, ttftMs);
      if (item.kind === 'done' || state.streamError) break;
    }
    const problem = completionProblem(state);
    return problem ? finish('error', new ApiError(200, problem, 'stream_incomplete')) : finish('completed', null);
  } catch (err) {
    if (isAbort(err) || opts.signal.aborted) return finish('stopped', null);
    return finish('error', toApiError(err));
  }
}
