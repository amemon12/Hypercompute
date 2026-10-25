import { useLiveRoom } from './context';
import { ActionError, LiveStatus, rate } from './common';

export function BenchmarksPanel() {
  const room = useLiveRoom();
  const deployment = room.activeDeployment;
  const running = room.benchmarks.some(b => b.status === 'running');
  const current = [...room.benchmarks].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const ready = room.online && deployment?.state === 'ready' && deployment.purpose !== 'tune';
  return <div className="hc-live-panel hc-benchmarks">
    <div className="hc-benchmark-top">
      <div className="hc-benchmark-history">
        <h3>Windows-and-Mac results</h3>
        <p>ASUS G14 with an RTX 5070 Ti Laptop GPU (12 GiB) and an M4 MacBook Air (16 GB), connected over campus Wi-Fi.</p>
        <div className="hc-live-table-wrap" tabIndex={0} role="region" aria-label="Historical benchmark results, horizontally scrollable">
          <table className="hc-live-table hc-history-table">
            <caption>Recorded chat decode</caption>
            <thead><tr><th scope="col">Model</th><th scope="col">Median decode</th><th scope="col">Range / samples</th></tr></thead>
            <tbody>
              <tr><th scope="row">Qwen2.5 14B<br /><small>Q6_K</small></th><td><strong className="hc-benchmark-rate">12.0</strong><span className="hc-benchmark-unit">tokens/s</span></td><td>11.7–12.3 tokens/s<br /><small>2 samples</small></td></tr>
              <tr><th scope="row">Mistral Small 3.2 24B<br /><small>Q4_K_M</small></th><td><strong className="hc-benchmark-rate">9.1</strong><span className="hc-benchmark-unit">tokens/s</span></td><td>3.5–10.6 tokens/s<br /><small>12 samples</small></td></tr>
            </tbody>
          </table>
        </div>
        <p className="hc-live-fine">These measurements describe the setup above. Your results will depend on your hardware and network.</p>
        <details className="hc-benchmark-conditions"><summary>Conditions and source</summary><p className="hc-live-fine">Successful chats with at least 8 generated tokens; 4,096-token context, earlier local-first split, Mac output head, runtime b11770 / RPC 8.0.0. Prefix cache was reused and power mode was not recorded. This was not a standardized fresh-cache benchmark.</p><a className="hc-live-text-link" href="https://github.com/amemon12/Hypercompute/blob/main/results/pair/20261021-campus-wifi-g14-m4air.json" target="_blank" rel="noreferrer">Inspect the recorded results <span aria-hidden="true">↗</span></a></details>
      </div>
      <div className="hc-benchmark-live">
        <h3>Benchmark your session</h3>
        <LiveStatus />
        <p>Run fixed short and long prompts three times each to measure decode speed and time to first token.</p>
        <p className="hc-live-fine">Chat pauses while the benchmark runs.</p>
        <button className="hc-live-button" type="button" disabled={!ready || !!room.busy || running} onClick={() => void room.benchmark()}>{running ? 'Benchmark in progress…' : room.busy === 'Starting benchmark' ? 'Starting benchmark…' : 'Run benchmark'} <span aria-hidden="true">↗</span></button>
        {!ready && <p className="hc-live-fine hc-benchmark-ready-note">Start a run in the session above to enable the benchmark.</p>}
        <details className="hc-benchmark-conditions"><summary>Measurement protocol</summary><p className="hc-live-fine">One warmup, three measured repetitions per prompt, up to 128 generated tokens and one request at a time. Prompt-cache reuse is disabled. Median and range reflect successful samples; failures are reported separately.</p></details>
        <ActionError />
        {!current.length && <p className="hc-live-fine">No live measurements yet.</p>}
      </div>
    </div>
    <div className="hc-benchmark-results" aria-live="polite">{current.slice(0, 5).map(result => <article className="hc-benchmark-result" key={result.id}>
      <div className="hc-live-result-heading"><strong>{result.deployment_id === deployment?.id ? 'Current deployment' : 'Previous deployment'}</strong><span>{result.status}</span></div>
      <p className="hc-live-fine">{room.deployments.find(d => d.id === result.deployment_id)?.plan.model_id ?? 'Model metadata unavailable'} · {room.deployments.find(d => d.id === result.deployment_id)?.plan.ctx_size.toLocaleString() ?? 'Unknown'} context tokens · {result.placement_label || result.deployment_id} · {result.request.repetitions} repetitions · {result.request.max_tokens} maximum output tokens</p>
      {result.status === 'running' && <p role="status">Running the fixed prompt set. Results will appear when measurement finishes…</p>}
      {result.error && <p className="hc-live-error">{result.error}</p>}
      {result.summaries.length > 0 && <div className="hc-live-table-wrap" tabIndex={0} role="region" aria-label="Live benchmark results, horizontally scrollable"><table className="hc-live-table"><caption>Live measured results</caption><thead><tr><th scope="col">Prompt</th><th scope="col">Median decode</th><th scope="col">Observed range</th><th scope="col">Median first token</th><th scope="col">Samples / failures</th></tr></thead><tbody>{result.summaries.map(summary => <tr key={summary.prompt_label}><th scope="row">{summary.prompt_label}</th><td>{rate(summary.decode_tokens_per_s.median)}</td><td>{summary.decode_tokens_per_s.min == null || summary.decode_tokens_per_s.max == null ? 'Not measured' : `${summary.decode_tokens_per_s.min.toFixed(1)}–${summary.decode_tokens_per_s.max.toFixed(1)} tokens/s`}</td><td>{summary.ttft_ms.median == null ? 'Not measured' : `${(summary.ttft_ms.median / 1000).toFixed(2)} s`}</td><td>{summary.decode_tokens_per_s.n} / {summary.failures}</td></tr>)}</tbody></table></div>}
    </article>)}</div>
  </div>;
}
