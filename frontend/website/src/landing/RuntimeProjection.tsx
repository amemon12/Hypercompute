import { useEffect, useId, useRef, useState } from 'react';
import { largeModelRuns, projectOutputRuntime, summarizeModel, type LargeModelRun } from './benchmark-evidence';
import './runtime-projection.css';

const MAX_OUTPUT_TOKENS = 1024;

export function formatRuntime(seconds: number) {
  const rounded = Math.round(seconds);
  const minutes = Math.floor(rounded / 60);
  return minutes ? `${minutes}m ${String(rounded % 60).padStart(2, '0')}s` : `${rounded}s`;
}

function axisMaximum(seconds: number) {
  const step = 10 ** Math.floor(Math.log10(seconds / 4));
  return Math.ceil(seconds / (step * 4)) * step * 4;
}

function RuntimeChart({ model, outputTokens, loadSeconds }: {
  model: LargeModelRun;
  outputTokens: number;
  loadSeconds: number;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(960);
  const titleId = useId();
  const descriptionId = useId();

  useEffect(() => {
    const node = container.current;
    if (!node) return;
    const observer = new ResizeObserver(entries => {
      setWidth(Math.max(240, Math.round(entries[0].contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const compact = width < 600;
  const height = compact ? 270 : 330;
  const plot = { left: 50, right: width - 16, top: 32, bottom: height - 53 };
  const endpoint = projectOutputRuntime(model, MAX_OUTPUT_TOKENS);
  const maxSeconds = axisMaximum(endpoint.max + loadSeconds);
  const x = (tokens: number) => plot.left + tokens / MAX_OUTPUT_TOKENS * (plot.right - plot.left);
  const y = (seconds: number) => plot.bottom - seconds / maxSeconds * (plot.bottom - plot.top);
  const prediction = (tokens: number) => projectOutputRuntime(model, Math.max(1, tokens));
  const allTokens = [1, ...Array.from({ length: 64 }, (_, index) => (index + 1) * 16)];
  const observedLengths = model.measuredSamples.map(sample => sample.outputTokens);
  const observedMin = Math.min(...observedLengths);
  const observedMax = Math.max(...observedLengths);
  const point = prediction(outputTokens);
  const path = (tokens: number[], metric: 'min' | 'median' | 'max') => tokens.map((value, index) =>
    `${index ? 'L' : 'M'}${x(value).toFixed(2)} ${y(prediction(value)[metric] + loadSeconds).toFixed(2)}`,
  ).join(' ');
  const envelope = `${path(allTokens, 'max')} ${path([...allTokens].reverse(), 'min').replace(/^M/, 'L')} Z`;
  const minutes = maxSeconds >= 120;
  const color = model.mode === 'local' ? '#b87951' : '#728c68';

  return (
    <div className="hc-runtime-chart" ref={container}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${titleId} ${descriptionId}`}>
        <title id={titleId}>Estimated response time by output length for {model.model}</title>
        <desc id={descriptionId}>
          At {outputTokens} output tokens, the median estimate is {formatRuntime(point.median + loadSeconds)}.
          The range across recorded rates is {formatRuntime(point.min + loadSeconds)} to {formatRuntime(point.max + loadSeconds)}.
          The recorded responses contained {observedMin === observedMax ? observedMax : `${observedMin} to ${observedMax}`} tokens.
          Longer responses are extrapolations, not measured results.
        </desc>
        <text x={plot.left} y="13" className="hc-runtime-axis-title">{minutes ? 'Minutes' : 'Seconds'}</text>
        {[0, 1, 2, 3, 4].map(index => {
          const seconds = maxSeconds * index / 4;
          return <g key={index}>
            <path d={`M${plot.left} ${y(seconds)}H${plot.right}`} className="hc-runtime-grid" />
            <text x={plot.left - 10} y={y(seconds) + 4} textAnchor="end">
              {minutes ? Number((seconds / 60).toFixed(1)) : seconds}
            </text>
          </g>;
        })}
        {(compact ? [0, 512, 1024] : [0, 256, 512, 768, 1024]).map(tokens => (
          <g key={tokens}>
            <path d={`M${x(tokens)} ${plot.bottom}v5`} className="hc-runtime-axis" />
            <text x={x(tokens)} y={plot.bottom + 23} textAnchor={tokens === 1024 ? 'end' : 'middle'}>{tokens.toLocaleString()}</text>
          </g>
        ))}
        <text x={plot.right} y={height - 2} textAnchor="end" className="hc-runtime-axis-title">Output tokens</text>
        <path d={envelope} fill={color} fillOpacity=".11" />
        <path d={path(allTokens, 'median')} fill="none" stroke={color} strokeWidth="2.5" strokeDasharray="7 5" />
        {model.measuredSamples.map(sample => (
          <circle key={`${sample.run}-${sample.rep}`} cx={x(sample.outputTokens)} cy={y(sample.requestSeconds + loadSeconds)}
            r="3" fill="var(--hc-paper)" stroke="currentColor" strokeWidth="1.2" />
        ))}
        <path d={`M${x(observedMin)} ${plot.bottom + 7}H${x(observedMax)}`} stroke="currentColor" strokeWidth="3" />
        <path d={`M${x(observedMin)} ${plot.bottom + 3}v8M${x(observedMax)} ${plot.bottom + 3}v8`} stroke="currentColor" />
        <path d={`M${x(outputTokens)} ${plot.bottom}V${y(point.median + loadSeconds)}`} className="hc-runtime-selection" />
        <circle cx={x(outputTokens)} cy={y(point.median + loadSeconds)} r="5" fill="currentColor" />
      </svg>
    </div>
  );
}

export function RuntimeProjection() {
  const [modelId, setModelId] = useState('qwen-235b');
  const [outputTokens, setOutputTokens] = useState(256);
  const [includeLoad, setIncludeLoad] = useState(false);
  const controlsId = useId();
  const model = largeModelRuns.find(run => run.id === modelId) ?? largeModelRuns[0];
  const summary = summarizeModel(model);
  const prediction = projectOutputRuntime(model, outputTokens);
  const loadSeconds = includeLoad ? summary.loadSeconds.median : 0;
  const observedLengths = model.measuredSamples.map(sample => sample.outputTokens);
  const observedMin = Math.min(...observedLengths);
  const observedMax = Math.max(...observedLengths);
  const observedLength = observedMin === observedMax ? `${observedMax}` : `${observedMin}–${observedMax}`;
  const promptRange = summary.promptTokens.min === summary.promptTokens.max
    ? `${summary.promptTokens.min}` : `${summary.promptTokens.min}–${summary.promptTokens.max}`;
  const plotColor = model.mode === 'local' ? '#b87951' : '#728c68';

  return (
    <section className="hc-runtime-projection" id="runtime-estimates" aria-labelledby="runtime-estimates-heading">
      <header className="hc-runtime-heading">
        <h2 id="runtime-estimates-heading">Runtime estimates</h2>
        <p>Time for a longer response at the recorded rate.</p>
      </header>

      <div className="hc-runtime-controls">
        <div className="hc-runtime-model-control">
          <label htmlFor={`${controlsId}-model`}>Model and setup</label>
          <select id={`${controlsId}-model`} value={modelId} onChange={event => setModelId(event.target.value)}>
            {largeModelRuns.map(run => <option key={run.id} value={run.id}>
              {run.model} · {run.parametersBillions}B · {run.mode === 'local' ? 'G14 alone' : 'G14 + 3 Macs'}
            </option>)}
          </select>
        </div>
        <div className="hc-runtime-length-control">
          <label htmlFor={`${controlsId}-length`}>Output length <span>{outputTokens.toLocaleString()} tokens</span></label>
          <input id={`${controlsId}-length`} type="range" min="32" max={MAX_OUTPUT_TOKENS} step="32" value={outputTokens}
            aria-valuetext={`${outputTokens} output tokens`} onChange={event => setOutputTokens(Number(event.target.value))} />
          <div className="hc-runtime-range-ends" aria-hidden="true"><span>32</span><span>1,024</span></div>
        </div>
      </div>

      <div className="hc-runtime-result-row">
        <div className="hc-runtime-result">
          <p className="hc-runtime-result-label">Estimated response time</p>
          <output aria-live="polite" aria-atomic="true">{formatRuntime(prediction.median + loadSeconds)}</output>
          <p className="hc-runtime-sample-range">{formatRuntime(prediction.min + loadSeconds)}–{formatRuntime(prediction.max + loadSeconds)} across the recorded rates</p>
        </div>
        <label className="hc-runtime-load-control">
          <input type="checkbox" checked={includeLoad} onChange={event => setIncludeLoad(event.target.checked)} />
          <span>Add model load <small>+{formatRuntime(summary.loadSeconds.median)}, recorded median</small></span>
        </label>
      </div>

      <RuntimeChart model={model} outputTokens={outputTokens} loadSeconds={loadSeconds} />

      <div className="hc-runtime-legend" aria-hidden="true">
        <span><i className="hc-runtime-recorded-mark" />Recorded requests{includeLoad ? ' + median load' : ''}: {observedLength} tokens</span>
        <span><i className="hc-runtime-dashed-mark" style={{ borderTopColor: plotColor }} />Constant-rate estimate</span>
        <span><i className="hc-runtime-range-mark" style={{ backgroundColor: plotColor }} />Range across samples</span>
      </div>
      <p className="hc-runtime-method">
        Based on {summary.sampleCount} requests with {promptRange} prompt tokens, keeping the prompt and setup fixed.
        Estimate = recorded request time + (new output tokens − recorded output tokens) ÷ decode rate.
        Circles show recorded request durations{includeLoad ? ' with the median model load added' : ''}; the dashed line extends each request at its recorded rate.
        The shaded range follows the samples; it is not a confidence interval. Longer responses have not been tested under these conditions.
        {includeLoad ? ' Model load is added once; downloads, peer seeding and warmups are excluded.' : ' Model load, downloads, peer seeding and warmups are excluded.'}
      </p>
    </section>
  );
}
