import { describe, expect, it } from 'vitest';
import { SseParser, parseSseText, readSseStream, type SseEvent } from './sse';

function feedChunks(chunks: string[]): SseEvent[] {
  const events: SseEvent[] = [];
  const parser = new SseParser((e) => events.push(e));
  for (const chunk of chunks) parser.feed(chunk);
  parser.flush();
  return events;
}

function streamOf(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(c);
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>): Promise<SseEvent[]> {
  const out: SseEvent[] = [];
  for await (const e of readSseStream(stream)) out.push(e);
  return out;
}

// A realistic llama-server stream: role chunk, content, finish chunk, usage+timings, keep-alive, [DONE].
const LLAMA_STREAM = [
  'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}]}',
  '',
  'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"Hél"}}]}',
  '',
  ':',
  '',
  'data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"lo 世界"}}]}',
  '',
  'data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}]}',
  '',
  'data: {"choices":[],"usage":{"completion_tokens":3},"timings":{"predicted_n":3,"predicted_per_second":21.5}}',
  '',
  'data: [DONE]',
  '',
  '',
].join('\n');

describe('SseParser basics', () => {
  it('parses single and multiple events', () => {
    expect(parseSseText('data: a\n\ndata: b\n\n')).toEqual([
      { event: 'message', data: 'a', id: null },
      { event: 'message', data: 'b', id: null },
    ]);
  });

  it('joins multi-line data with newlines', () => {
    expect(parseSseText('data: line1\ndata: line2\ndata:\ndata: line4\n\n')).toEqual([
      { event: 'message', data: 'line1\nline2\n\nline4', id: null },
    ]);
  });

  it('strips exactly one leading space after the colon', () => {
    expect(parseSseText('data:  two spaces\n\ndata:none\n\n').map((e) => e.data)).toEqual([
      ' two spaces',
      'none',
    ]);
  });

  it('handles CRLF line endings', () => {
    expect(parseSseText('data: a\r\ndata: b\r\n\r\ndata: [DONE]\r\n\r\n').map((e) => e.data)).toEqual([
      'a\nb',
      '[DONE]',
    ]);
  });

  it('handles lone CR line endings', () => {
    expect(parseSseText('data: a\rdata: b\r\rdata: c\r\r').map((e) => e.data)).toEqual(['a\nb', 'c']);
  });

  it('ignores comments and keep-alive pings', () => {
    expect(parseSseText(':\n\n: ping\ndata: x\n\n:\n\n')).toEqual([{ event: 'message', data: 'x', id: null }]);
  });

  it('reads event and id fields and keeps the last id', () => {
    const events = parseSseText('event: hc\nid: 7\ndata: {"seq":7}\n\ndata: next\n\n');
    expect(events).toEqual([
      { event: 'hc', data: '{"seq":7}', id: '7' },
      { event: 'message', data: 'next', id: '7' },
    ]);
  });

  it('does not dispatch an event without data', () => {
    expect(parseSseText('event: hc\n\nid: 3\n\n')).toEqual([]);
  });

  it('dispatches an empty data field', () => {
    expect(parseSseText('data:\n\n')).toEqual([{ event: 'message', data: '', id: null }]);
  });

  it('treats a field with no colon as a field name with empty value', () => {
    expect(parseSseText('data\n\n')).toEqual([{ event: 'message', data: '', id: null }]);
  });

  it('strips a leading byte-order mark', () => {
    expect(parseSseText('\uFEFFdata: x\n\n').map((e) => e.data)).toEqual(['x']);
  });

  it('dispatches a final event missing its trailing blank line on flush', () => {
    expect(parseSseText('data: a\n\ndata: [DONE]').map((e) => e.data)).toEqual(['a', '[DONE]']);
  });
});

describe('SseParser fragmentation', () => {
  const whole = parseSseText(LLAMA_STREAM);

  it('parses the reference stream', () => {
    expect(whole.map((e) => e.data).at(-1)).toBe('[DONE]');
    expect(whole).toHaveLength(6);
  });

  it('gives identical events for every two-way split point', () => {
    for (let i = 0; i <= LLAMA_STREAM.length; i++) {
      expect(feedChunks([LLAMA_STREAM.slice(0, i), LLAMA_STREAM.slice(i)])).toEqual(whole);
    }
  });

  it('gives identical events when fed one character at a time', () => {
    expect(feedChunks(Array.from(LLAMA_STREAM))).toEqual(whole);
  });

  it('gives identical events for every split point with CRLF endings', () => {
    const crlf = LLAMA_STREAM.replace(/\n/g, '\r\n');
    for (let i = 0; i <= crlf.length; i++) {
      expect(feedChunks([crlf.slice(0, i), crlf.slice(i)])).toEqual(whole);
    }
  });

  it('does not treat a CRLF split across chunks as two line endings', () => {
    expect(feedChunks(['data: a\r', '\ndata: b\r', '\n\r', '\n'])).toEqual([
      { event: 'message', data: 'a\nb', id: null },
    ]);
  });

  it('survives empty chunks between a CR and its LF', () => {
    expect(feedChunks(['data: a\r', '', '\n\r\n'])).toEqual([{ event: 'message', data: 'a', id: null }]);
  });

  it('handles several events arriving in one chunk', () => {
    expect(feedChunks(['data: 1\n\ndata: 2\n\ndata: 3\n\n']).map((e) => e.data)).toEqual(['1', '2', '3']);
  });
});

describe('readSseStream', () => {
  const encoder = new TextEncoder();

  it('reassembles UTF-8 characters split across byte chunks', async () => {
    const bytes = encoder.encode(LLAMA_STREAM);
    const expected = parseSseText(LLAMA_STREAM);
    for (const size of [1, 2, 3, 5, 7, 64]) {
      const chunks: Uint8Array[] = [];
      for (let i = 0; i < bytes.length; i += size) chunks.push(bytes.slice(i, i + size));
      expect(await collect(streamOf(chunks))).toEqual(expected);
    }
  });

  it('cancels the stream when the consumer stops early', async () => {
    let cancelled = false;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: [DONE]\n\ndata: never\n\n'));
      },
      cancel() {
        cancelled = true;
      },
    });
    for await (const e of readSseStream(stream)) {
      expect(e.data).toBe('[DONE]');
      break;
    }
    expect(cancelled).toBe(true);
  });

  it('propagates a stream error to the consumer', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: a\n\n'));
        controller.error(new Error('connection reset'));
      },
    });
    await expect(collect(stream)).rejects.toThrow('connection reset');
  });
});
