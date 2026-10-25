// Pure accumulation of llama-server `chat.completion.chunk` events (passed through by the gateway).
// See docs/research/llama_server.md: content and reasoning arrive as deltas, `finish_reason` only on
// the dedicated final chunk, `usage` on a trailing chunk with empty `choices`, `timings` on the last
// chunk before `data: [DONE]`, and a mid-stream failure is `data: {"error": {...}}` with no [DONE].

import type { ChatChunk, ChatUsage, EngineTimings } from './types';

export interface ChatStreamState {
  content: string;
  reasoning: string;
  finishReason: string | null;
  timings: Partial<EngineTimings> | null;
  usage: ChatUsage | null;
  sawDone: boolean;
  streamError: string | null;
}

export const EMPTY_STREAM_STATE: ChatStreamState = {
  content: '',
  reasoning: '',
  finishReason: null,
  timings: null,
  usage: null,
  sawDone: false,
  streamError: null,
};

export type StreamItem = { kind: 'done' } | { kind: 'chunk'; chunk: ChatChunk };

/** Interpret one SSE `data` payload. Throws on anything that is not `[DONE]` or a JSON object. */
export function parseStreamData(data: string): StreamItem {
  const trimmed = data.trim();
  if (trimmed === '[DONE]') return { kind: 'done' };
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    throw new Error(`Unreadable stream chunk (possibly cut off): ${trimmed.slice(0, 120)}`);
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Stream chunk is not a JSON object');
  }
  return { kind: 'chunk', chunk: parsed as ChatChunk };
}

export function applyStreamItem(state: ChatStreamState, item: StreamItem): ChatStreamState {
  if (item.kind === 'done') return { ...state, sawDone: true };
  const { chunk } = item;
  if (chunk.error) {
    const message = chunk.error.message ?? 'The engine reported an error mid-stream';
    return { ...state, streamError: message };
  }
  let content = state.content;
  let reasoning = state.reasoning;
  let finishReason = state.finishReason;
  for (const choice of chunk.choices ?? []) {
    const delta = choice.delta;
    if (delta?.content) content += delta.content;
    if (delta?.reasoning_content) reasoning += delta.reasoning_content;
    if (choice.finish_reason) finishReason = choice.finish_reason;
  }
  return {
    ...state,
    content,
    reasoning,
    finishReason,
    usage: chunk.usage ?? state.usage,
    timings: chunk.timings ?? state.timings,
  };
}

/** True once the model produced real output (content or reasoning), the TTFT moment. */
export function hasOutput(state: ChatStreamState): boolean {
  return state.content.length > 0 || state.reasoning.length > 0;
}

/** A stream is complete only with `[DONE]` or a finish chunk, and no mid-stream error. */
export function completionProblem(state: ChatStreamState): string | null {
  if (state.streamError) return state.streamError;
  if (!state.sawDone && state.finishReason === null) {
    return 'The stream ended before the model finished (no finish chunk or [DONE]).';
  }
  return null;
}
