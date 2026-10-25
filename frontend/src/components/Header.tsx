import type { ApiError } from '../api/client';
import { isFullEngineInfo, type Health, type OwnerBootstrap } from '../api/types';
import { StatusText } from './common';

interface HeaderProps {
  bootstrap: OwnerBootstrap | null;
  health: Health | undefined;
  healthError: ApiError | null;
}

export function engineSummary(health: Health | undefined): string {
  if (!health) return 'engine: checking…';
  const e = health.engine;
  if (!isFullEngineInfo(e)) {
    return `llama.cpp ${e.release ?? 'release unavailable'} · accelerator shown only on the host computer`;
  }
  if (!e.installed) return 'llama.cpp engine not installed';
  const release = e.release ?? 'release unavailable';
  const smoke = e.smoke === 'passed' ? 'smoke test passed' : e.smoke === 'failed' ? 'smoke test FAILED' : 'smoke test not run';
  return `llama.cpp ${release} · ${e.accelerator} · ${smoke}`;
}

export function Header({ bootstrap, health, healthError }: HeaderProps) {
  const addresses = bootstrap
    ? bootstrap.lan_addresses.map((a) => `${a}:${bootstrap.gateway_port}`)
    : [];
  return (
    <header className="app-header">
      <div className="brand">
        <h1>Hypercompute</h1>
        <span className="muted small">{bootstrap ? `v${bootstrap.version}` : ''}</span>
      </div>
      <dl className="header-facts">
        <div>
          <dt>Gateway on your network</dt>
          <dd className="mono">{addresses.length > 0 ? addresses.join(', ') : 'unavailable'}</dd>
        </div>
        <div>
          <dt>Engine</dt>
          <dd>
            {healthError ? (
              <StatusText tone="bad">gateway unreachable</StatusText>
            ) : (
              <span className="mono">{engineSummary(health)}</span>
            )}
          </dd>
        </div>
      </dl>
    </header>
  );
}
