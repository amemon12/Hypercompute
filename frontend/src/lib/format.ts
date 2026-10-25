// Display helpers. A missing measurement (null/undefined) renders as "unavailable", never as 0.

import type { FitLabel, HostInventory, OSInfo, PlanStrategy, StatSummary } from '../api/types';

export const UNAVAILABLE = 'unavailable';
export const STALE_AFTER_S = 15;

function missing(n: number | null | undefined): n is null | undefined {
  return n === null || n === undefined || Number.isNaN(n);
}

const BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'] as const;

/** Binary (IEC) units with the unit always spelled out, e.g. "23.9 GiB". */
export function formatBytes(bytes: number | null | undefined): string {
  if (missing(bytes)) return UNAVAILABLE;
  const sign = bytes < 0 ? '-' : '';
  let value = Math.abs(bytes);
  let unit = 0;
  while (value >= 1024 && unit < BYTE_UNITS.length - 1) {
    value /= 1024;
    unit++;
  }
  if (unit === 0) return `${sign}${value} B`;
  const digits = value < 10 ? 2 : value < 100 ? 1 : 0;
  return `${sign}${value.toFixed(digits)} ${BYTE_UNITS[unit]}`;
}

export function formatMs(ms: number | null | undefined): string {
  if (missing(ms)) return UNAVAILABLE;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatRate(tokensPerSecond: number | null | undefined): string {
  if (missing(tokensPerSecond)) return UNAVAILABLE;
  return `${tokensPerSecond.toFixed(1)} tok/s`;
}

export function formatCount(n: number | null | undefined): string {
  if (missing(n)) return UNAVAILABLE;
  return String(n);
}

export function formatAge(seconds: number | null | undefined): string {
  if (missing(seconds)) return UNAVAILABLE;
  if (seconds < 1) return 'just now';
  if (seconds < 90) return `${Math.round(seconds)} s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min ago`;
  return `${(seconds / 3600).toFixed(1)} h ago`;
}

export function isStale(heartbeatAgeS: number | null | undefined): boolean {
  return !missing(heartbeatAgeS) && heartbeatAgeS > STALE_AFTER_S;
}

/** Remaining time until an ISO timestamp as m:ss (h:mm:ss above an hour), or "expired". */
export function formatCountdown(expiresAtIso: string, nowMs: number): string {
  const end = Date.parse(expiresAtIso);
  if (Number.isNaN(end)) return UNAVAILABLE;
  const total = Math.floor((end - nowMs) / 1000);
  if (total <= 0) return 'expired';
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = String(total % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return UNAVAILABLE;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString();
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return UNAVAILABLE;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

const OS_NAMES: Record<OSInfo['system'], string> = {
  windows: 'Windows',
  macos: 'macOS',
  linux: 'Linux',
  other: 'Other OS',
};

export function osLabel(os: OSInfo): string {
  return [OS_NAMES[os.system], os.release, os.machine].filter(Boolean).join(' · ');
}

/** The device people would name the machine by: first GPU/accelerator, else the CPU model. */
export function mainDeviceName(inv: HostInventory): string {
  const accel = inv.devices.find((d) => d.kind === 'gpu' || d.kind === 'igpu' || d.kind === 'accel');
  if (accel) return accel.name;
  return inv.cpu.model ?? inv.devices[0]?.name ?? UNAVAILABLE;
}

/** "median (min – max)" or "unavailable" when there were no samples. */
export function formatStat(stat: StatSummary, fmt: (n: number | null) => string): string {
  if (stat.median === null) return UNAVAILABLE;
  const range = stat.min !== null && stat.max !== null ? ` (${fmt(stat.min)} – ${fmt(stat.max)})` : '';
  return `${fmt(stat.median)}${range}`;
}

/** "12 (0–11)" for contiguous layers, "3 (0, 4, 9)" otherwise. */
export function formatLayers(layers: number[]): string {
  if (layers.length === 0) return '0';
  const sorted = [...layers].sort((a, b) => a - b);
  const first = sorted[0] ?? 0;
  const last = sorted[sorted.length - 1] ?? 0;
  if (last - first + 1 === sorted.length) {
    return sorted.length === 1 ? `1 (${first})` : `${sorted.length} (${first}–${last})`;
  }
  const shown = sorted.slice(0, 6).join(', ');
  return `${sorted.length} (${shown}${sorted.length > 6 ? ', …' : ''})`;
}

export const STRATEGY_LABELS: Record<PlanStrategy, string> = {
  auto: 'Fastest that fits',
  all_selected: 'Run together (every selected friend gets layers)',
  fastest_fit: 'Fastest fit',
  local_only: 'This machine only',
};

export const STRATEGY_HINTS: Record<PlanStrategy, string> = {
  auto:
    "Tries this machine's GPU first, then friends' GPUs or this PC's RAM for the rest, and capacity mode from disk " +
    'only when nothing faster fits.',
  all_selected: 'Every friend you tick below holds part of the model.',
  fastest_fit: 'Uses the fewest devices that fit, starting with this machine.',
  local_only: 'Runs only on this computer; friends stay idle.',
};

export const FIT_LABELS: Record<FitLabel, string> = {
  estimated_fit: 'Estimated to fit',
  does_not_fit: 'Does not fit',
  fit_unknown: 'Fit unknown',
};

export function quoteArg(arg: string): string {
  return /[\s"']/.test(arg) ? `"${arg.replace(/"/g, '\\"')}"` : arg;
}
