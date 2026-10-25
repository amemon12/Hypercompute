import { useEffect, useRef, useState } from 'react';
import { useLiveRoom } from './context';
import type { LivePlan, MeasuredProfile } from './types';
export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [status, setStatus] = useState(''); const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => () => clearTimeout(timer.current), []);
  return <button type="button" className="hc-live-button hc-live-button-quiet" onClick={async () => {
    try { await navigator.clipboard.writeText(text); setStatus('Copied'); } catch { setStatus('Select and copy the text'); }
    clearTimeout(timer.current); timer.current = setTimeout(() => setStatus(''), 3000);
  }}><span aria-live="polite">{status || label}</span></button>;
}
export function LiveStatus() {
  const room = useLiveRoom();
  return <><div className="hc-live-status" role="status"><span className="hc-status-dot" data-online={room.online} />
    <span>{room.online ? 'Connected to this computer’s host' : room.connection === 'connecting' ? 'Looking for your local host…' : 'Local host offline'}{!room.online && room.updatedAt && ' · showing last received data'}</span>
    <button className="hc-live-text-button" type="button" onClick={() => void room.refresh()}>Refresh</button></div>{room.online && room.error && <p className="hc-live-error" role="status">{room.error}</p>}</>;
}
export function ActionError() { const room = useLiveRoom(); return room.actionError ? <div className="hc-live-error" role="alert"><p>{room.actionError}</p><button type="button" className="hc-live-text-button" onClick={room.clearError}>Dismiss</button></div> : null; }
export function rate(value: number | null | undefined): string { return value == null || !Number.isFinite(value) || value <= 0 ? 'Not measured' : `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} tokens/s`; }
export function SpeedEstimate({ plan, profiles }: { plan: LivePlan; profiles: MeasuredProfile[] }) {
  const measured = plan.profile_id ? profiles.find(p => p.id === plan.profile_id && p.model_id === plan.model_id) : undefined;
  if (measured?.decode_tokens_per_s != null) return <div className="hc-live-speed"><span className="hc-live-label">Matching measured profile</span><strong>{rate(measured.decode_tokens_per_s)}</strong><small>Measured for this model and device configuration. Today’s conditions may differ.</small></div>;
  const estimate = plan.decode_estimate;
  if (!estimate || !plan.feasible || plan.capacity_mode) return <p className="hc-live-fine"><strong>No speed estimate available.</strong> Run a benchmark after loading the model.</p>;
  return <div className="hc-live-speed"><span className="hc-live-label">Uncalibrated planner estimate</span><strong>≈ {rate(estimate.tokens_per_second)}</strong><small>Estimated decode speed for one conversation. This setup has not been benchmarked.</small><details><summary>Estimate assumptions</summary><ul>{estimate.assumptions.map(a => <li key={a}>{a}</li>)}</ul></details></div>;
}
