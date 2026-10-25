import { useEffect, useState } from "react";

export const MACHINE_COLORS = ["#f2b84b", "#5ec2b7", "#ef7d5a", "#a99cf0", "#b9d76a", "#6fb3f2", "#f28cb8", "#d9c38a"];

export function machineColor(index: number): string {
  return MACHINE_COLORS[index % MACHINE_COLORS.length];
}

export function gib(bytes: number): string {
  const v = bytes / 2 ** 30;
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}

export function tps(rate: number | null | undefined): string {
  if (rate === null || rate === undefined) return "n/a";
  return rate >= 100 ? rate.toFixed(0) : rate >= 10 ? rate.toFixed(1) : rate.toFixed(2);
}

export function seconds(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

/** Re-renders every `everyMs`, for countdowns and "online" checks computed against server timestamps. */
export function useNow(everyMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(id);
  }, [everyMs]);
  return now;
}

export function hue(text: string): number {
  let h = 0;
  for (const ch of text) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return h;
}

export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  const letters = parts.length > 1 ? parts[0][0] + parts[1][0] : name.slice(0, 2);
  return letters.toUpperCase();
}
