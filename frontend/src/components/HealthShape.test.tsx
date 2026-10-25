import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { EngineInfo, Health } from '../api/types';
import { engineSummary } from './Header';
import { TechnicalDetails } from './TechnicalDetails';

// GET /api/health from a caller that is not the owner on this machine (for example the page opened
// through the LAN address) carries only {release, rpc_proto} in `engine` (backend/hypercompute/app.py).
const lanHealth: Health = { status: 'ok', version: '0.1.0', engine: { release: 'b1234', rpc_proto: '3.0.0' } };

const ownerEngine: EngineInfo = {
  engine: 'llama_rpc',
  installed: true,
  release: 'b1234',
  build_number: 1234,
  commit: 'abc1234',
  rpc_proto: '3.0.0',
  variant: 'windows-x64-cuda',
  accelerator: 'cuda',
  bin_dir: null,
  smoke: 'passed',
  smoke_detail: null,
  devices: [],
};

describe('health with the non-owner engine shape', () => {
  it('summarizes the release without claiming the engine is missing', () => {
    expect(engineSummary(lanHealth)).toBe('llama.cpp b1234 · accelerator shown only on the host computer');
  });

  it('renders technical details without crashing', () => {
    const html = renderToStaticMarkup(
      <TechnicalDetails deployment={null} plan={null} nodes={undefined} models={undefined} health={lanHealth} />,
    );
    expect(html).toContain('Technical details');
    expect(html).not.toContain('Engine on this machine');
  });

  it('shows release and accelerator for the owner shape', () => {
    const health: Health = { status: 'ok', version: '0.1.0', engine: ownerEngine };
    expect(engineSummary(health)).toBe('llama.cpp b1234 · cuda · smoke test passed');
    const html = renderToStaticMarkup(
      <TechnicalDetails deployment={null} plan={null} nodes={undefined} models={undefined} health={health} />,
    );
    expect(html).toContain('Engine on this machine');
  });
});
