import { useState } from 'react';
import { benchmarkSources, largeModelRuns, poolingComparisons, summarizeModel } from './benchmark-evidence';
import './large-model-results.css';

function duration(seconds: number) {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const rounded = Math.round(seconds);
  return `${Math.floor(rounded / 60)}m ${String(rounded % 60).padStart(2, '0')}s`;
}

export function LargeModelResults() {
  return (
    <section className="hc-large-results" id="large-models" aria-labelledby="large-model-heading">
      <div className="hc-benchmark-section-heading">
        <h2 id="large-model-heading">Large-model runs</h2>
        <p>Measured</p>
      </div>
      <p className="hc-large-intro">235B across four laptops. 671B on one.</p>
      <p className="hc-large-description">Quantized mixture-of-experts models, using GPU memory, RAM and SSD-backed weights. The table shows medians from the recorded requests, with loading timed separately.</p>
      <div className="hc-large-table-scroll" role="region" aria-label="Large-model runtime measurements" tabIndex={0}>
        <table className="hc-large-table">
          <caption>Request duration includes prompt processing and generation. It excludes model loading, downloads and warmups.</caption>
          <thead><tr><th scope="col">Model / setup</th><th scope="col">Decode</th><th scope="col">First token</th><th scope="col">Request duration</th><th scope="col">Model load</th></tr></thead>
          <tbody>{largeModelRuns.map(run => {
            const summary = summarizeModel(run);
            return <tr key={run.id}>
              <th scope="row">
                <span className="hc-large-model">{run.model}{run.id === 'deepseek-671b' && ' · 671B'}</span>
                <span className="hc-large-quantization">{run.quantization}</span>
                <span className="hc-large-configuration">{run.peerCount ? `G14 + ${run.peerCount} M4 MacBook Airs` : 'G14 alone'}</span>
              </th>
              <td><strong>{summary.decodeTokensPerSecond.median.toFixed(2)}</strong><span>tokens/s</span></td>
              <td><strong>{duration(summary.timeToFirstTokenSeconds.median)}</strong><span>{duration(summary.timeToFirstTokenSeconds.min)}–{duration(summary.timeToFirstTokenSeconds.max)}</span></td>
              <td><strong>{duration(summary.requestSeconds.median)}</strong><span>{run.outputTokens} output tokens · {summary.sampleCount} requests</span></td>
              <td><strong>{duration(summary.loadSeconds.median)}</strong><span>{run.batches.length} {run.batches.length === 1 ? 'load' : 'loads'}</span></td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <div className="hc-large-notes">
        <p>235B uses the nine steady-state requests. Its caches were populated; the initial cold load took 38m 09s over Wi-Fi.</p>
        <p>671B generated 32 reasoning tokens per request. Multi-Mac layouts with the G14 as host were planned, but no successful distributed 671B run was recorded on it.</p>
      </div>
      <details className="hc-large-method">
        <summary>Models, machines and measurement method</summary>
        <div className="hc-large-method-body">
          <p>The host was a Zephyrus G14 with an RTX 5070 Ti Laptop GPU (12 GiB), 31.1 GiB of RAM and an NVMe SSD. Each M4 MacBook Air had 16 GiB of unified memory and contributed up to 7.5 GiB. Peer traffic used Iroh over building Wi-Fi.</p>
          <ul>{largeModelRuns.map(run => {
            const stats = summarizeModel(run);
            return <li key={run.id}><a href={run.modelCardUrl} target="_blank" rel="noreferrer">{run.model}</a>: {run.parametersBillions}B total / {run.activeParametersBillions}B active parameters; {(run.fileBytes / 1024 ** 3).toFixed(2)} GiB checkpoint. Recorded prompts were {stats.promptTokens.min}–{stats.promptTokens.max} tokens, with {run.outputTokens} output tokens.</li>;
          })}</ul>
          <p>These runs used runtime b11770, temperature 0, seed 7 and no prompt cache. Warmups and failed attempts are excluded. The 120B and 671B samples contained reasoning output, not completed final answers. Output quality was not evaluated. Some 120B runs shared the host with a background code workflow.</p>
          <div className="hc-benchmark-source-links"><a href={benchmarkSources.paper} target="_blank" rel="noreferrer">Experiment notes</a>{largeModelRuns.map(run => <a key={run.id} href={run.batches[0].sourceUrl} target="_blank" rel="noreferrer">{run.id === 'qwen-235b' ? '235B' : run.id === 'deepseek-671b' ? '671B' : '120B'} raw run</a>)}</div>
        </div>
      </details>
    </section>
  );
}

const metrics = {
  prefill: { label: 'Prompt processing', unit: 'tokens/s', max: 12, precision: 2, direction: 'Higher is faster' },
  decode: { label: 'Token generation', unit: 'tokens/s', max: 5, precision: 2, direction: 'Higher is faster' },
  ssdMiBPerToken: { label: 'Host SSD reads', unit: 'MiB / decoded token', max: 700, precision: 0, direction: 'Lower means fewer host reads' },
} as const;

export function PoolingResults() {
  const [metric, setMetric] = useState<keyof typeof metrics>('prefill');
  const display = metrics[metric];
  return (
    <section className="hc-pooling-results" id="pooling" aria-labelledby="pooling-heading">
      <div className="hc-pooling-heading">
        <div><h2 id="pooling-heading">What three Macs changed</h2><p>Same model. G14 alone, then G14 with three M4 MacBook Airs.</p></div>
        <label className="hc-pooling-select">Compare<select value={metric} onChange={event => setMetric(event.target.value as keyof typeof metrics)}>{Object.entries(metrics).map(([key, value]) => <option value={key} key={key}>{value.label}</option>)}</select></label>
      </div>
      <div className="hc-pooling-legend"><span><i />G14 alone</span><span><i />G14 + 3 Macs</span><span>{display.unit} · {display.direction}</span></div>
      <div className="hc-pooling-chart" aria-live="polite">
        {poolingComparisons.map(comparison => {
          const values = comparison[metric];
          return <div className="hc-pooling-row" key={comparison.id}>
            <div className="hc-pooling-model"><h3>{comparison.model}</h3><p>9 measured requests per setup</p></div>
            <div className="hc-pooling-bars">
              <div className="hc-pooling-bar"><span className="hc-pooling-bar-label">G14 alone</span><div><i style={{ width: `${values.localMean / display.max * 100}%` }} /><b>{values.localMean.toFixed(display.precision)}</b></div></div>
              <div className="hc-pooling-bar hc-pooling-bar-shared"><span className="hc-pooling-bar-label">G14 + 3 Macs</span><div><i style={{ width: `${values.distributedMean / display.max * 100}%` }} /><b>{values.distributedMean.toFixed(display.precision)}</b></div></div>
            </div>
            <div className="hc-pooling-ratio"><strong>{values.ratio.toFixed(2)}×</strong><span>95% interval {values.confidence95[0].toFixed(2)}–{values.confidence95[1].toFixed(2)}</span></div>
          </div>;
        })}
      </div>
      <div className="hc-pooling-findings">
        <div><strong>54% fewer host SSD reads</strong><p>With 235B, the Macs held 22 layers and reduced how much the host read from its SSD per token.</p></div>
        <div><strong>85% fewer host SSD reads</strong><p>With 120B, more of the remaining expert weights fit in the host’s RAM. Prefill was 2.99× faster.</p></div>
      </div>
      <p className="hc-pooling-note">Values are pooled means; intervals use the repo’s bootstrap analysis. The 235B decode interval includes no improvement. SSD reads are measured on the host, not across the whole system, and are not an energy measurement. <a href={benchmarkSources.paper} target="_blank" rel="noreferrer">Read the comparison</a></p>
    </section>
  );
}
