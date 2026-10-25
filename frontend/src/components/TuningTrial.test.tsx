import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { deployment } from '../testing/fixtures';
import { ChatPanel } from './ChatPanel';
import { DeploymentPanel } from './DeploymentPanel';
import { RunsPanel } from './RunsPanel';

const noop = (): void => undefined;
const trial = deployment({ id: 'trial1', purpose: 'tune' });
const run = deployment({ id: 'run1' });

function benchmarkButton(html: string): string {
  return html.match(/<button[^>]*>Benchmark<\/button>/)?.[0] ?? '';
}

function runsPanel(ready: typeof run | null): string {
  return renderToStaticMarkup(
    <RunsPanel runs={[]} runsError={null} benchmarks={[]} benchmarksError={null} readyDeployment={ready} onChanged={noop} />,
  );
}

describe('tuner trial deployments (purpose "tune")', () => {
  it('keeps chat off for a ready tuning trial and says why', () => {
    const html = renderToStaticMarkup(<ChatPanel deployment={trial} maxTokens={undefined} />);
    expect(html).not.toContain('Talking to');
    expect(html).toContain('tuning trial');
    expect(html).toMatch(/<textarea[^>]*disabled=""/);
    expect(renderToStaticMarkup(<ChatPanel deployment={run} maxTokens={undefined} />)).toContain('Talking to qwen2.5-7b-q4');
  });

  it("offers benchmarks only on the owner's own ready run", () => {
    const onTrial = runsPanel(trial);
    expect(benchmarkButton(onTrial)).toContain('disabled=""');
    expect(benchmarkButton(onTrial)).toContain('tuning trial');
    expect(benchmarkButton(runsPanel(run))).not.toContain('disabled');
    expect(benchmarkButton(runsPanel(null))).toContain('disabled=""');
  });

  it("shows a trial as a tuning trial, not as the owner's run", () => {
    const html = renderToStaticMarkup(<DeploymentPanel deployment={trial} onChanged={noop} />);
    expect(html).toContain('Tuning trial');
    expect(html).toContain('Chat and benchmarks stay off');
    expect(renderToStaticMarkup(<DeploymentPanel deployment={run} onChanged={noop} />)).not.toContain('Tuning trial');
  });
});
