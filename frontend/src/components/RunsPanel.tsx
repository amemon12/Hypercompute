import { api, type ApiError } from '../api/client';
import type { BenchmarkResult, Deployment, RunRecord } from '../api/types';
import { useAction } from '../hooks/useAction';
import { isTuningTrial, readyRun } from '../lib/deployments';
import { formatCount, formatDateTime, formatMs, formatRate, formatStat, formatTime } from '../lib/format';
import { Card, ErrorNote } from './common';

function BenchmarkCard({ result }: { result: BenchmarkResult }) {
  return (
    <div className="bench">
      <p>
        <strong>{result.label ?? result.placement_label}</strong>{' '}
        <span className="muted small">
          {result.status} · {formatDateTime(result.created_at)} · {result.request.repetitions} repetitions,{' '}
          {result.request.max_tokens} max tokens
          {result.label ? ` · ${result.placement_label}` : ''}
        </span>
      </p>
      {result.error && <p className="small"><strong>Error:</strong> {result.error}</p>}
      {result.summaries.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Prompt</th>
                <th>TTFT median (range)</th>
                <th>Decode median (range)</th>
                <th>Prompt processing median (range)</th>
                <th>Tokens (median)</th>
                <th>Samples</th>
                <th>Failures</th>
              </tr>
            </thead>
            <tbody>
              {result.summaries.map((s) => (
                <tr key={s.prompt_label}>
                  <td>{s.prompt_label}</td>
                  <td>{formatStat(s.ttft_ms, formatMs)}</td>
                  <td>{formatStat(s.decode_tokens_per_s, formatRate)}</td>
                  <td>{formatStat(s.prompt_tokens_per_s, formatRate)}</td>
                  <td>{formatCount(s.completion_tokens.median)}</td>
                  <td>{s.decode_tokens_per_s.n}</td>
                  <td>{s.failures}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

interface RunsPanelProps {
  runs: RunRecord[] | undefined;
  runsError: ApiError | null;
  benchmarks: BenchmarkResult[] | undefined;
  benchmarksError: ApiError | null;
  readyDeployment: Deployment | null;
  onChanged: () => void;
}

function benchmarkTitle(target: Deployment | null, ready: Deployment | null): string {
  if (target) return 'Short and long prompts, 3 repetitions each';
  return isTuningTrial(ready) ? 'Off during a tuning trial: benchmarks need your own ready run' : 'Needs a ready run';
}

export function RunsPanel({ runs, runsError, benchmarks, benchmarksError, readyDeployment, onChanged }: RunsPanelProps) {
  // Benchmarks pin the owner's run by id, so the gateway refuses one that is no longer the ready deployment.
  const target = readyRun(readyDeployment);
  const start = useAction(
    (id: string) => api.startBenchmark({ deployment_id: id, prompt_set: 'both', repetitions: 3 }),
    onChanged,
  );
  const running = (benchmarks ?? []).some((b) => b.status === 'running');

  return (
    <Card
      title="Runs and benchmarks"
      subtitle="Every request is recorded, including failed and cancelled ones."
      actions={
        <button
          type="button"
          disabled={!target || start.busy || running}
          onClick={() => {
            if (target) void start.run(target.id);
          }}
          title={benchmarkTitle(target, readyDeployment)}
        >
          {running ? 'Benchmark running…' : start.busy ? 'Starting…' : 'Benchmark'}
        </button>
      }
    >
      <ErrorNote error={start.error} onDismiss={start.clearError} />
      <ErrorNote error={benchmarksError} />
      {(benchmarks ?? []).slice(0, 5).map((b) => (
        <BenchmarkCard key={b.id} result={b} />
      ))}

      <ErrorNote error={runsError} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Started</th>
              <th>Kind</th>
              <th>Status</th>
              <th>TTFT</th>
              <th>Tokens</th>
              <th>Decode</th>
            </tr>
          </thead>
          <tbody>
            {(runs ?? []).map((r) => (
              <tr key={r.id} title={r.error ?? undefined}>
                <td className="mono small">{formatTime(r.started_at)}</td>
                <td>
                  {r.kind}
                  {r.prompt_label && <span className="muted small"> · {r.prompt_label}</span>}
                  {r.simulated && <strong> (simulated)</strong>}
                </td>
                <td>
                  {r.status}
                  {r.error && <span className="muted small block">{r.error}</span>}
                </td>
                <td>{formatMs(r.ttft_ms)}</td>
                <td>{formatCount(r.completion_tokens)}</td>
                <td>{formatRate(r.timings?.predicted_per_second)}</td>
              </tr>
            ))}
            {runs && runs.length === 0 && (
              <tr>
                <td colSpan={6} className="muted">
                  No runs yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
