import { describe, expect, it } from 'vitest';
import {
  EMPTY_STREAM_STATE,
  applyStreamItem,
  completionProblem,
  hasOutput,
  parseStreamData,
  type ChatStreamState,
} from './chatStream';
import { parseSseText } from './sse';

function run(payload: string): ChatStreamState {
  return parseSseText(payload)
    .map((e) => parseStreamData(e.data))
    .reduce(applyStreamItem, EMPTY_STREAM_STATE);
}

describe('chat stream accumulation', () => {
  it('accumulates content, reasoning, finish reason, usage and timings', () => {
    const state = run(
      [
        'data: {"choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":null}}]}',
        'data: {"choices":[{"index":0,"finish_reason":null,"delta":{"reasoning_content":"think"}}]}',
        'data: {"choices":[{"index":0,"finish_reason":null,"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"index":0,"finish_reason":"length","delta":{}}]}',
        'data: {"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":5},"timings":{"predicted_n":2,"predicted_per_second":12.5}}',
        'data: [DONE]',
      ].join('\n\n') + '\n\n',
    );
    expect(state.content).toBe('Hi');
    expect(state.reasoning).toBe('think');
    expect(state.finishReason).toBe('length');
    expect(state.usage).toEqual({ completion_tokens: 2, prompt_tokens: 5 });
    expect(state.timings).toEqual({ predicted_n: 2, predicted_per_second: 12.5 });
    expect(state.sawDone).toBe(true);
    expect(completionProblem(state)).toBeNull();
  });

  it('does not count the role-only first chunk as output', () => {
    const state = run('data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}\n\n');
    expect(hasOutput(state)).toBe(false);
  });

  it('reports a mid-stream engine error', () => {
    const state = run(
      'data: {"choices":[{"index":0,"delta":{"content":"pa"}}]}\n\ndata: {"error":{"message":"rpc device lost","type":"server_error"}}\n\n',
    );
    expect(state.content).toBe('pa');
    expect(completionProblem(state)).toBe('rpc device lost');
  });

  it('flags a stream that ends without [DONE] or a finish chunk', () => {
    const state = run('data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n');
    expect(completionProblem(state)).toMatch(/ended before the model finished/);
  });

  it('accepts a finish chunk without [DONE]', () => {
    const state = run('data: {"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}\n\n');
    expect(completionProblem(state)).toBeNull();
  });

  it('rejects malformed or truncated payloads', () => {
    expect(() => parseStreamData('{"choices":[{"delta":')).toThrow(/Unreadable stream chunk/);
    expect(() => parseStreamData('[1,2]')).toThrow(/not a JSON object/);
    expect(parseStreamData(' [DONE] ')).toEqual({ kind: 'done' });
  });
});
