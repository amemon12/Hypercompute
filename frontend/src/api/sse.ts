// Incremental Server-Sent Events parser (WHATWG "event stream interpretation"), used for the
// streamed /v1/chat/completions response, which EventSource cannot consume (it needs POST + headers).
//
// Handles: events split across arbitrary chunk boundaries, CRLF / LF / lone CR line endings
// (including a CRLF split between two chunks), multi-line `data:` fields, comments and
// keep-alive pings (`:`), `event:` / `id:` fields, and a leading byte-order mark.

export interface SseEvent {
  /** `event:` field, or "message" when absent. */
  event: string;
  /** All `data:` lines of the event joined with "\n". */
  data: string;
  /** Last seen `id:` field (persists across events per the spec), or null. */
  id: string | null;
}

export type SseListener = (event: SseEvent) => void;

export class SseParser {
  private buffer = '';
  private pendingCR = false;
  private started = false;
  private dataLines: string[] = [];
  private eventType = '';
  private lastEventId: string | null = null;

  constructor(private readonly listener: SseListener) {}

  /** Feed already-decoded text. Call as often as chunks arrive. */
  feed(text: string): void {
    let chunk = text;
    if (!this.started && chunk.length > 0) {
      this.started = true;
      if (chunk.charCodeAt(0) === 0xfeff) chunk = chunk.slice(1);
    }
    if (this.pendingCR) {
      // The previous chunk ended in "\r"; a "\n" here belongs to the same CRLF line ending.
      if (chunk.length === 0) return;
      this.pendingCR = false;
      if (chunk.charCodeAt(0) === 0x0a) chunk = chunk.slice(1);
    }
    const buf = this.buffer + chunk;
    let lineStart = 0;
    for (let i = 0; i < buf.length; i++) {
      const code = buf.charCodeAt(i);
      if (code !== 0x0a && code !== 0x0d) continue;
      const line = buf.slice(lineStart, i);
      if (code === 0x0d) {
        if (i + 1 < buf.length) {
          if (buf.charCodeAt(i + 1) === 0x0a) i++;
        } else {
          this.pendingCR = true;
        }
      }
      this.processLine(line);
      lineStart = i + 1;
    }
    this.buffer = buf.slice(lineStart);
  }

  /**
   * End of stream. The spec discards an event that was not terminated by a blank line; we are
   * lenient and dispatch it so a server that omits the final blank line after `data: [DONE]`
   * still terminates cleanly. Consumers must still validate the payload (it may be truncated).
   */
  flush(): void {
    if (this.buffer.length > 0) {
      const rest = this.buffer;
      this.buffer = '';
      this.processLine(rest);
    }
    this.pendingCR = false;
    this.dispatch();
  }

  private processLine(line: string): void {
    if (line === '') {
      this.dispatch();
      return;
    }
    if (line.charCodeAt(0) === 0x3a) return; // ":" comment / keep-alive ping
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.charCodeAt(0) === 0x20) value = value.slice(1);
    switch (field) {
      case 'data':
        this.dataLines.push(value);
        break;
      case 'event':
        this.eventType = value;
        break;
      case 'id':
        if (!value.includes('\0')) this.lastEventId = value;
        break;
      default:
        // `retry` and unknown fields are ignored.
        break;
    }
  }

  private dispatch(): void {
    if (this.dataLines.length === 0) {
      this.eventType = '';
      return;
    }
    const event: SseEvent = {
      event: this.eventType === '' ? 'message' : this.eventType,
      data: this.dataLines.join('\n'),
      id: this.lastEventId,
    };
    this.dataLines = [];
    this.eventType = '';
    this.listener(event);
  }
}

/** Parse a whole SSE payload at once (convenience for tests and small bodies). */
export function parseSseText(text: string): SseEvent[] {
  const events: SseEvent[] = [];
  const parser = new SseParser((e) => events.push(e));
  parser.feed(text);
  parser.flush();
  return events;
}

/**
 * Read an SSE byte stream, decoding UTF-8 incrementally (a multi-byte character split across
 * network chunks is reassembled by TextDecoder in streaming mode) and yielding events in order.
 * Breaking out of the loop cancels the underlying stream.
 */
export async function* readSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseEvent, void, undefined> {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  const queue: SseEvent[] = [];
  const parser = new SseParser((e) => queue.push(e));
  let finished = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.feed(decoder.decode(value, { stream: true }));
      for (const event of queue.splice(0)) yield event;
    }
    finished = true;
    parser.feed(decoder.decode());
    parser.flush();
    for (const event of queue.splice(0)) yield event;
  } finally {
    if (!finished) {
      // Consumer stopped early ([DONE], abort or error): release the connection.
      try {
        await reader.cancel();
      } catch {
        // cancel() rejects when the stream already errored or was aborted; nothing left to release.
      }
    }
    reader.releaseLock();
  }
}
