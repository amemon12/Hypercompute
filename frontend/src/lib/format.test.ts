import { describe, expect, it } from 'vitest';
import {
  UNAVAILABLE,
  formatAge,
  formatBytes,
  formatCountdown,
  formatLayers,
  formatMs,
  formatRate,
  formatStat,
  isStale,
} from './format';

describe('formatBytes', () => {
  it('uses explicit binary units', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1536)).toBe('1.50 KiB');
    expect(formatBytes(25 * 1024 ** 3 + 0.6 * 1024 ** 3)).toBe('25.6 GiB');
    expect(formatBytes(128 * 1024 ** 3)).toBe('128 GiB');
  });

  it('renders missing measurements as unavailable, never 0', () => {
    expect(formatBytes(null)).toBe(UNAVAILABLE);
    expect(formatBytes(undefined)).toBe(UNAVAILABLE);
    expect(formatBytes(Number.NaN)).toBe(UNAVAILABLE);
  });
});

describe('timing formatters', () => {
  it('formats durations and rates', () => {
    expect(formatMs(312.4)).toBe('312 ms');
    expect(formatMs(2500)).toBe('2.50 s');
    expect(formatMs(null)).toBe(UNAVAILABLE);
    expect(formatRate(18.25)).toBe('18.3 tok/s');
    expect(formatRate(null)).toBe(UNAVAILABLE);
  });

  it('formats heartbeat ages and staleness at 15 s', () => {
    expect(formatAge(null)).toBe(UNAVAILABLE);
    expect(formatAge(0.2)).toBe('just now');
    expect(formatAge(4.6)).toBe('5 s ago');
    expect(isStale(15)).toBe(false);
    expect(isStale(15.1)).toBe(true);
    expect(isStale(null)).toBe(false);
  });

  it('counts down to expiry', () => {
    const now = Date.parse('2026-10-20T12:00:00Z');
    expect(formatCountdown('2026-10-20T12:04:05Z', now)).toBe('4:05');
    expect(formatCountdown('2026-10-20T13:00:09Z', now)).toBe('1:00:09');
    expect(formatCountdown('2026-10-20T11:59:59Z', now)).toBe('expired');
    expect(formatCountdown('not a date', now)).toBe(UNAVAILABLE);
  });
});

describe('summaries', () => {
  it('formats median with range', () => {
    expect(formatStat({ n: 3, median: 20, min: 18, max: 22.5 }, formatRate)).toBe(
      '20.0 tok/s (18.0 tok/s – 22.5 tok/s)',
    );
    expect(formatStat({ n: 0, median: null, min: null, max: null }, formatRate)).toBe(UNAVAILABLE);
  });

  it('formats layer lists', () => {
    expect(formatLayers([])).toBe('0');
    expect(formatLayers([3])).toBe('1 (3)');
    expect(formatLayers([2, 0, 1])).toBe('3 (0–2)');
    expect(formatLayers([0, 4, 9])).toBe('3 (0, 4, 9)');
  });
});
