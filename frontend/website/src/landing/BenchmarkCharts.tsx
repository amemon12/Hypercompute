import type { ReactNode } from 'react';
import type { BarRow, ChartSpec, RangeRow, RatioRow } from './benchmark-charts';

// Every bar shares its chart's linear, zero-based scale.
function pct(value: number, max: number): string {
  return `${Math.min(100, Math.max(0, value / max * 100))}%`;
}

function ChartFrame<R>({ spec, children, kind }: { spec: ChartSpec<R>; children: ReactNode; kind: string }) {
  return (
    <figure className={`hc-chart hc-chart-${kind}`}>
      <figcaption>
        <h2 className="hc-chart-title">{spec.title}</h2>
        <span className="hc-chart-unit">{spec.unit}</span>
      </figcaption>
      <div className="hc-chart-rows">{children}</div>
    </figure>
  );
}

function RowLabel({ label, note }: { label: string; note?: string }) {
  return <div className="hc-chart-label"><strong>{label}</strong>{note && <span>{note}</span>}</div>;
}

export function RatioChart({ spec }: { spec: ChartSpec<RatioRow> }) {
  return (
    <ChartFrame spec={spec} kind="ratios">
      {spec.rows.map(row => {
        const [value, interval] = row.display.split(' [');
        return (
          <div className="hc-chart-row" key={`${row.label} ${row.note}`}>
            <RowLabel label={row.label} note={row.note} />
            <div className="hc-chart-track" aria-hidden="true">
              <i className="hc-chart-ratio-bar" data-tone={row.tone} style={{ width: pct(row.value, spec.max) }} />
              {spec.reference && <i className="hc-chart-reference" style={{ left: pct(spec.reference.at, spec.max) }} />}
            </div>
            <div className="hc-chart-result" data-tone={row.tone}>
              <strong>{value}</strong>
              {interval && <span aria-label={`95% interval ${interval.replace(']', '')}`}>{interval.replace(']', '').replace(', ', '–')}</span>}
            </div>
          </div>
        );
      })}
      <p className="hc-chart-footnote">1× = {spec.reference?.label ?? 'same speed'}. Ranges are 95% intervals across repeats.</p>
    </ChartFrame>
  );
}

export function BarChart({ spec }: { spec: ChartSpec<BarRow> }) {
  const series = spec.series ?? [];
  const paired = series.length > 1;
  const unit = spec.valueUnit ?? 'tok/s';
  return (
    <ChartFrame spec={spec} kind={paired ? 'paired' : 'single'}>
      {spec.rows.map(row => (
        <div className="hc-chart-row" key={`${row.label} ${row.note}`}>
          <RowLabel label={row.label} note={row.note} />
          <div className="hc-chart-series">
            {row.values.map((value, index) => (
              <div className="hc-chart-series-row" key={index}>
                {paired && <span className="hc-chart-series-name">{series[index]}</span>}
                <div className="hc-chart-track" aria-hidden="true">
                  <i className="hc-chart-bar" data-series={paired ? index : 1} style={{ width: pct(value, spec.max) }} />
                </div>
                <span className="hc-chart-bar-value">{value.toLocaleString('en-US', { maximumFractionDigits: 3 })}<small>{unit}</small></span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </ChartFrame>
  );
}

export function RangeChart({ spec }: { spec: ChartSpec<RangeRow> }) {
  return (
    <ChartFrame spec={spec} kind="ranges">
      {spec.rows.map(row => (
        <div className="hc-chart-row" key={row.label}>
          <RowLabel label={row.label} />
          <div className="hc-chart-track" aria-hidden="true">
            <i className="hc-chart-range" data-pooled={row.pooled} style={{ left: pct(row.lo, spec.max), width: `calc(${pct(row.hi, spec.max)} - ${pct(row.lo, spec.max)})` }} />
          </div>
          <span className="hc-chart-range-value">{row.display}</span>
        </div>
      ))}
    </ChartFrame>
  );
}
