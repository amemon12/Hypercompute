import type { ConcurrencyRow, ResultCard } from './benchmark-sheet';
import type { BarRow, ChartSpec } from './benchmark-charts';

export function ComparisonResults({ spec }: { spec: ChartSpec<BarRow> }) {
  return <dl className="hc-bm-result-metrics">
    {spec.rows.map(row => <div key={`${row.label} ${row.note}`}>
      <dt>{row.label}{row.note && ` · ${row.note}`}</dt>
      <dd>{row.display}</dd>
    </div>)}
  </dl>;
}

export function ResultCards({ cards }: { cards: readonly ResultCard[] }) {
  return (
    <div className="hc-bm-result-list">
      {cards.map(card => (
        <article className="hc-bm-result" key={card.model}>
          <div className="hc-bm-result-heading">
            <h3>{card.model}</h3>
            <p className="hc-bm-spec">{card.spec}</p>
          </div>
          <div className="hc-bm-result-values">
            <p><strong>{card.headline}</strong> <span>{card.interval}</span> · {card.metric}</p>
            <p>{card.summary}</p>
          </div>
          <dl className="hc-bm-result-metrics">
            {card.details.map(detail => (
              <div key={detail.label}><dt>{detail.label}</dt><dd>{detail.value}</dd></div>
            ))}
          </dl>
        </article>
      ))}
    </div>
  );
}

export function ModelCell({ model, spec }: { model: string; spec: string }) {
  return <th scope="row"><span className="hc-bm-model">{model}</span><span className="hc-bm-cell-note">{spec}</span></th>;
}

export function RatioCell({ ratio, interval }: { ratio: string; interval: string }) {
  return <td className="hc-bm-num"><strong className="hc-bm-ratio">{ratio}</strong> <span className="hc-bm-interval">{interval}</span></td>;
}

export function ConcurrencyTable({ rows, label }: { rows: readonly ConcurrencyRow[]; label: string }) {
  return (
    <div className="hc-bm-scroll" role="region" aria-label={label} tabIndex={0}>
      <table className="hc-bm-table">
        <thead>
          <tr><th scope="col">Model</th><th scope="col">Prompt / answer</th><th scope="col">1 chat</th><th scope="col">4 chats</th><th scope="col">8 chats</th><th scope="col">8 vs 1</th></tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.model}>
              <ModelCell model={row.model} spec={row.spec} />
              <td className="hc-bm-mono">{row.lengths}</td>
              {row.totals.map((total, index) => <td className="hc-bm-num" key={index}>{total}</td>)}
              <RatioCell ratio={row.eightVsOne} interval={row.interval} />
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
