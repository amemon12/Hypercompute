import { useEffect, useId, useRef, useState } from 'react';
import { calculateCostEnergy } from './costEnergyMath';
import './cost-energy.css';

export type CostEnergyModel = {
  id: string;
  label: string;
  decodeTps: number;
  localWatts: number;
  cloudHourlyPrice: number;
  cloudConfiguration: string;
};

const money = (value: number) => new Intl.NumberFormat('en-US', {
  style: 'currency', currency: 'USD', minimumFractionDigits: 2,
  maximumFractionDigits: value > 0 && value < .01 ? 4 : 3,
}).format(value);
const quantity = (value: number) => new Intl.NumberFormat('en-US', {
  maximumFractionDigits: value > 0 && value < .01 ? Math.min(8, 1 - Math.floor(Math.log10(value))) : 3,
}).format(value);
const approximateQuantity = (value: number) => new Intl.NumberFormat('en-US', {
  maximumFractionDigits: value > 0 && value < .1 ? Math.min(8, 1 - Math.floor(Math.log10(value))) : 1,
}).format(value);
const readNumber = (value: string) => value.trim() ? Number(value) : NaN;

function NumberField({ label, value, onChange, min, max, step = 'any', hint }: {
  label: string; value: string; onChange: (value: string) => void;
  min: number; max: number; step?: string; hint?: string;
}) {
  const id = useId();
  const number = readNumber(value);
  const invalid = !Number.isFinite(number) || number < min || number > max || (step === '1' && !Number.isInteger(number));
  return <label className="hc-cost-field" htmlFor={id}>
    <span>{label}</span>
    <input id={id} type="number" value={value} min={min} max={max} step={step}
      onChange={event => onChange(event.target.value)} aria-invalid={invalid}
      aria-describedby={hint ? `${id}-hint` : undefined} />
    {hint && <small id={`${id}-hint`}>{hint}</small>}
  </label>;
}

function CostSpeedPlot({ outputTokens, localCost, cloudHourlyPrice, selectedSpeed }: {
  outputTokens: number; localCost: number; cloudHourlyPrice: number; selectedSpeed: number;
}) {
  const id = useId();
  const figure = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(720);
  useEffect(() => {
    const node = figure.current;
    if (!node) return;
    const observer = new ResizeObserver(entries => {
      setWidth(Math.max(240, Math.round(entries[0].contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  const compact = width < 600;
  const height = compact ? 270 : 294;
  const left = 90;
  const right = width - 16;
  const top = 26;
  const bottom = height - 50;
  const cloudCost = (speed: number) => (outputTokens - 1) / speed / 3600 * cloudHourlyPrice;
  const rawMaximum = Math.max(cloudCost(5), localCost, .001) * 1.12;
  const axisStep = 10 ** Math.floor(Math.log10(rawMaximum));
  const maximum = Math.ceil(rawMaximum / axisStep) * axisStep;
  const axisMoney = (value: number) => new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD', minimumFractionDigits: 2,
    maximumFractionDigits: Math.max(2, 1 - Math.floor(Math.log10(maximum))),
  }).format(value);
  const x = (speed: number) => left + (speed - 5) / 195 * (right - left);
  const y = (cost: number) => bottom - cost / maximum * (bottom - top);
  const points = Array.from({ length: 196 }, (_, index) => `${x(index + 5)},${y(cloudCost(index + 5))}`).join(' ');
  const markerX = x(selectedSpeed);
  const markerY = y(cloudCost(selectedSpeed));

  return <figure className="hc-cost-figure" ref={figure}>
    <div className="hc-cost-plot-heading">
      <h3>What if the cloud runs faster?</h3>
      <div className="hc-cost-legend" aria-hidden="true">
        <span><i className="hc-cost-line-local" />Local electricity</span>
        <span><i className="hc-cost-line-cloud" />Cloud rental</span>
      </div>
    </div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
      <title id={`${id}-title`}>Estimated cost for {outputTokens.toLocaleString()} output tokens</title>
      <desc id={`${id}-desc`}>Cloud rental cost falls as assumed throughput rises from 5 to 200 tokens per second. Local electricity cost stays at {money(localCost)}. At the selected {selectedSpeed} tokens per second, cloud rental costs {money(cloudCost(selectedSpeed))}. These are projections, not cloud benchmark results.</desc>
      {[0, .5, 1].map(fraction => {
        const value = maximum * fraction;
        return <g key={fraction}>
          <path d={`M${left} ${y(value)}H${right}`} className="hc-cost-grid" />
          <text x={left - 12} y={y(value) + 4} textAnchor="end">{axisMoney(value)}</text>
        </g>;
      })}
      <path d={`M${left} ${top}V${bottom}H${right}`} className="hc-cost-axis" />
      <path d={`M${left} ${y(localCost)}H${right}`} className="hc-cost-local-stroke" />
      <polyline points={points} className="hc-cost-cloud-stroke" />
      <path d={`M${markerX} ${markerY}V${bottom}`} className="hc-cost-marker-guide" />
      <circle cx={markerX} cy={markerY} r="5" fill="#171715" />
      {(compact ? [5, 100, 200] : [5, 50, 100, 150, 200]).map(speed => <g key={speed}>
        <path d={`M${x(speed)} ${bottom}v5`} className="hc-cost-axis" />
        <text x={x(speed)} y={bottom + 22} textAnchor={speed === 200 ? 'end' : 'middle'}>{speed}</text>
      </g>)}
      <text x={right} y={height - 2} textAnchor="end">{compact ? 'Cloud speed · tokens/s' : 'Assumed cloud throughput · tokens/s'}</text>
    </svg>
    <figcaption>The dot is your selected cloud speed. The local line uses the recorded decode rate and the power assumption below.</figcaption>
  </figure>;
}

export function CostEnergyEstimate({ models }: { models: CostEnergyModel[] }) {
  const id = useId();
  const [modelId, setModelId] = useState(models[0]?.id ?? '');
  const model = models.find(item => item.id === modelId) ?? models[0];
  const [tokens, setTokens] = useState('1000');
  const [cloudSpeed, setCloudSpeed] = useState('20');
  const [watts, setWatts] = useState(String(model?.localWatts ?? 250));
  const [electricity, setElectricity] = useState('0.15');
  const [gridFactor, setGridFactor] = useState('0.288152');
  const [rental, setRental] = useState(String(model?.cloudHourlyPrice ?? 3.58));

  if (!model) return null;

  const values = {
    outputTokens: readNumber(tokens), localDecodeTps: model.decodeTps,
    cloudDecodeTps: readNumber(cloudSpeed), localWatts: readNumber(watts),
    electricityPerKwh: readNumber(electricity), gridKgPerKwh: readNumber(gridFactor),
    cloudHourlyPrice: readNumber(rental),
  };
  const withinLimits = values.outputTokens <= 1_000_000 && values.cloudDecodeTps >= 5 && values.cloudDecodeTps <= 200
    && values.localWatts <= 5000 && values.electricityPerKwh <= 10 && values.gridKgPerKwh <= 2 && values.cloudHourlyPrice <= 1000;
  const estimate = withinLimits ? calculateCostEnergy(values) : null;

  function chooseModel(nextId: string) {
    const next = models.find(item => item.id === nextId);
    if (!next) return;
    setModelId(nextId);
    setWatts(String(next.localWatts));
    setRental(String(next.cloudHourlyPrice));
  }

  return <section className="hc-cost-energy" id="cost-energy" aria-labelledby={`${id}-heading`}>
    <div className="hc-benchmark-section-heading">
      <h2 id={`${id}-heading`}>Cost and energy estimates</h2>
      <p>Same output, different hardware</p>
    </div>
    <p className="hc-cost-intro">Compare local electricity with a rented GPU configuration. No power or billing was recorded for these runs.</p>

    <div className="hc-cost-primary-controls">
      <label className="hc-cost-field" htmlFor={`${id}-model`}>
        <span>Recorded model</span>
        <select id={`${id}-model`} value={model.id} onChange={event => chooseModel(event.target.value)}>
          {models.map(item => <option value={item.id} key={item.id}>{item.label}</option>)}
        </select>
        <small>{model.decodeTps.toFixed(3)} tokens/s recorded median</small>
      </label>
      <NumberField label="Output tokens" value={tokens} onChange={setTokens} min={2} max={1_000_000} step="1" hint="2–1,000,000 tokens" />
      <NumberField label="Assumed cloud speed · tokens/s" value={cloudSpeed} onChange={setCloudSpeed} min={5} max={200} hint="Illustrative; not measured" />
    </div>

    {estimate ? <>
      <dl className="hc-cost-results">
        <div><dt>Local electricity</dt><dd>{money(estimate.localCost)}</dd></div>
        <div><dt>Cloud rental</dt><dd>{money(estimate.cloudCost)}</dd></div>
        <div><dt>Estimated cost difference</dt><dd>{estimate.costDifference < 0 ? '−' : ''}{money(Math.abs(estimate.costDifference))}</dd>
          <p>{estimate.costDifference >= 0 ? 'Less spent locally in this scenario.' : 'More spent locally in this scenario.'}</p>
        </div>
      </dl>
      <CostSpeedPlot outputTokens={values.outputTokens} localCost={estimate.localCost}
        cloudHourlyPrice={values.cloudHourlyPrice} selectedSpeed={values.cloudDecodeTps} />
      <div className="hc-cost-footprint">
        <div>
          <h3>Local electricity use</h3>
          <p className="hc-cost-energy-value">{quantity(estimate.localKwh)} <span>kWh</span></p>
          <p>For {approximateQuantity(estimate.localHours * 60)} minutes of decoding at {values.localWatts} W total.</p>
        </div>
        <div>
          <h3>Operational emissions</h3>
          <p className="hc-cost-energy-value">{approximateQuantity(estimate.localKgCo2e * 1000)} <span>g CO₂e</span></p>
          <p>Based on the selected grid factor. No cloud emissions or measured CO₂ reduction is claimed.</p>
        </div>
      </div>
    </> : <p className="hc-cost-input-error" role="status">Enter valid values within the stated ranges to calculate this scenario.</p>}

    <details className="hc-cost-assumptions">
      <summary>Power and pricing assumptions</summary>
      <p className="hc-cost-assumption-intro">Power, electricity price, and cloud speed are illustrative. Change them to match your hardware and workload. Changing the model resets power and rental price to that model’s example.</p>
      <div className="hc-cost-assumption-controls">
        <NumberField label="Total local wall power · W" value={watts} onChange={setWatts} min={0} max={5000} hint="All participating computers" />
        <NumberField label="Electricity · USD/kWh" value={electricity} onChange={setElectricity} min={0} max={10} hint="$0.15 is an illustrative rate" />
        <NumberField label="Grid emissions · kg CO₂e/kWh" value={gridFactor} onChange={setGridFactor} min={0} max={2} hint="Default: Northwest regional average" />
        <NumberField label="Total GPU rental · USD/hour" value={rental} onChange={setRental} min={0} max={1000} hint={`Example: ${model.cloudConfiguration}`} />
      </div>
      <p>Rental examples use <a href="https://www.runpod.io/pricing" target="_blank" rel="noreferrer">Runpod’s A100 80 GB rate</a> of $1.79 per GPU-hour, checked October 24, 2026. GPU counts are capacity examples, not tested cloud deployments. Availability, model compatibility, and runtime memory must be checked.</p>
      <p>The grid default is 0.288152 kg CO₂e/kWh, converted from 635.267 lb/MWh in <a href="https://www.epa.gov/egrid/summary-data" target="_blank" rel="noreferrer">EPA eGRID 2023, Northwest</a>. It is a regional annual average, not the actual campus electricity mix.</p>
      <p>Decode time = (output tokens − 1) ÷ tokens/s. Local energy = total watts × decode hours ÷ 1,000. Electricity cost = kWh × tariff. Cloud cost = decode hours × rental rate. Emissions = kWh × grid factor.</p>
      <p>These estimates exclude first-token latency, model loading, prompt processing, storage, hardware purchase, idle rental time, and billing minimums. Compare the same model, quantization, and context. Emissions cover electricity only; hardware manufacturing and a measured cloud comparison are not included. <a href="https://sci.greensoftware.foundation/" target="_blank" rel="noreferrer">Calculation methodology</a></p>
    </details>
    <p className="hc-cost-current-assumptions">Current scenario: {watts || 'n/a'} W total · ${electricity || 'n/a'}/kWh · ${rental || 'n/a'}/rental hour. Values are estimates, not recorded savings.</p>
  </section>;
}
