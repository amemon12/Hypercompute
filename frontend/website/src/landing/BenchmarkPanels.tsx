import type { ReactNode } from 'react';
import { ComparisonResults, ConcurrencyTable, ModelCell, RatioCell, ResultCards } from './BenchmarkBlocks';
import { BarChart, RangeChart, RatioChart } from './BenchmarkCharts';
import {
  coldStart, everyDecodeRatio, everyFirstToken, gx10Decode, gx10EightVsOne, laptopDecodeRatio, laptopFirstToken,
  laptopManyChats, noHelpRatio,
} from './benchmark-charts';
import {
  earlierConditions, earlierNote, earlierRows, everyCards, everyConditions, everyManyConditions, everyManyNote,
  everyManyRows, fourPeers, gx10Conditions, gx10ManyConditions, gx10ManyNote, gx10Notes, gx10Rows, gx10Single,
  laptopCards, laptopConditions, laptopManyConditions, laptopManyNote, noHelpReasons,
} from './benchmark-sheet';

function Lede({ children }: { children: string }) {
  return <p className="hc-bm-lede">{children}</p>;
}

function AllResults({ children }: { children: ReactNode }) {
  return (
    <details className="hc-bm-details">
      <summary>All results</summary>
      {children}
    </details>
  );
}

export function LaptopsPanel() {
  return <>
    <Lede>RTX 5070 Ti laptop, alone and with two M4 MacBook Airs. Home Wi-Fi, models already loaded.</Lede>
    <RatioChart spec={laptopDecodeRatio} />
    <BarChart spec={laptopFirstToken} />
    <p className="hc-bm-caption">Short prompts, mostly one run per setup. The largest gain comes from moving weights off the laptop’s SSD.</p>
    <AllResults>
      <p className="hc-bm-caption">{laptopConditions}</p>
      <ResultCards cards={laptopCards} />
      <h3 className="hc-bm-subhead">Building Wi-Fi</h3>
      <p className="hc-bm-caption">{earlierConditions}</p>
      <div className="hc-bm-scroll" role="region" aria-label="Laptop runs on building Wi-Fi" tabIndex={0}>
        <table className="hc-bm-table">
          <thead><tr><th scope="col">Model and pool</th><th scope="col">Prompt / answer</th><th scope="col">Decode</th><th scope="col">First token</th><th scope="col">Prompt reading</th></tr></thead>
          <tbody>
            {earlierRows.map(row => (
              <tr key={`${row.model} ${row.pool}`}>
                <ModelCell model={row.model} spec={row.pool} />
                <td className="hc-bm-mono">{row.lengths}</td>
                <RatioCell ratio={row.decode} interval={row.decodeInterval} />
                <td><span className="hc-bm-num">{row.first}</span>{row.firstNote && <span className="hc-bm-cell-note">{row.firstNote}</span>}</td>
                <td className="hc-bm-num">{row.prompt}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hc-bm-caption">{earlierNote}</p>
    </AllResults>
  </>;
}

export function Gx10Panel() {
  return <>
    <Lede>NVIDIA GB10 · 121.6 GiB unified memory · one machine, one chat, after loading.</Lede>
    <BarChart spec={gx10Decode} />
    <p className="hc-bm-caption">The 400B, 671B and 1T models also read weights from SSD. These are short-output tests.</p>
    <AllResults>
      <p className="hc-bm-caption">{gx10Conditions}</p>
      <div className="hc-bm-scroll" role="region" aria-label="The GX10 alone, one chat" tabIndex={0}>
        <table className="hc-bm-table">
          <thead><tr><th scope="col">Model</th><th scope="col">Prompt / answer</th><th scope="col">Decode, tok/s</th><th scope="col">First token</th><th scope="col">Full answer</th><th scope="col">SSD reads per token</th></tr></thead>
          <tbody>
            {gx10Single.map(row => (
              <tr key={row.model}>
                <ModelCell model={row.model} spec={row.spec} />
                <td className="hc-bm-mono">{row.lengths}</td>
                <td className="hc-bm-num">{row.decode}</td>
                <td className="hc-bm-num">{row.first}</td>
                <td className="hc-bm-num">{row.answer}</td>
                <td className="hc-bm-num">{row.ssd}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="hc-bm-notes">{gx10Notes.map(note => <li key={note}>{note}</li>)}</ul>
    </AllResults>
  </>;
}

export function EveryMachinePanel() {
  return <>
    <Lede>GX10 with a gaming laptop and two or three M4 MacBook Airs. Timings start after loading.</Lede>
    <RatioChart spec={everyDecodeRatio} />
    <BarChart spec={everyFirstToken} />
    <p className="hc-bm-caption">With three peers, 671B and 1T start responding sooner, with no clear gain in generation speed.</p>
    <AllResults>
      <p className="hc-bm-caption">{everyConditions}</p>
      <ResultCards cards={everyCards} />
      <h3 className="hc-bm-subhead">Time to first token · shared vs alone</h3>
      <ComparisonResults spec={everyFirstToken} />
      <h3 className="hc-bm-subhead">Four-peer setup</h3>
      <p className="hc-bm-caption">{fourPeers.conditions}</p>
      <div className="hc-bm-scroll" role="region" aria-label="GX10 with four peers" tabIndex={0}>
        <table className="hc-bm-table">
          <thead><tr><th scope="col">Model</th><th scope="col">Prompt reading</th><th scope="col">First token</th><th scope="col">Decode</th><th scope="col">SSD reads per token</th></tr></thead>
          <tbody>
            {fourPeers.rows.map(row => (
              <tr key={row.model}>
                <th scope="row"><span className="hc-bm-model">{row.model}</span></th>
                <td className="hc-bm-num">{row.prompt}</td>
                <td><span className="hc-bm-num">{row.first}</span><span className="hc-bm-cell-note">{row.firstTimes}</span></td>
                <td className="hc-bm-num">{row.decode}</td>
                <td className="hc-bm-num">{row.ssd}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </AllResults>
  </>;
}

export function ManyUsersPanel() {
  return <>
    <Lede>Laptop alone vs laptop with two M4 Airs; then the GX10 alone. Rates are total output across all chats.</Lede>
    <BarChart spec={laptopManyChats} />
    <RatioChart spec={gx10EightVsOne} />
    <p className="hc-bm-caption">More chats increase total output, but each response takes longer.</p>
    <AllResults>
      <h3 className="hc-bm-subhead">Laptop with two MacBook Airs</h3>
      <p className="hc-bm-caption">{laptopManyConditions}</p>
      <ComparisonResults spec={laptopManyChats} />
      <p className="hc-bm-caption">{laptopManyNote}</p>
      <h3 className="hc-bm-subhead">GX10 with three peers</h3>
      <p className="hc-bm-caption">{everyManyConditions}</p>
      <div className="hc-bm-scroll" role="region" aria-label="Every machine with many chats" tabIndex={0}>
        <table className="hc-bm-table hc-bm-table-narrow">
          <thead><tr><th scope="col">Model</th><th scope="col">Chats</th><th scope="col">GX10 alone</th><th scope="col">Pooled</th><th scope="col">Pooled vs alone</th></tr></thead>
          <tbody>
            {everyManyRows.map(row => (
              <tr key={`${row.model} ${row.chats}`}>
                <th scope="row"><span className="hc-bm-model">{row.model}</span></th>
                <td className="hc-bm-num">{row.chats}</td>
                <td className="hc-bm-num">{row.alone}</td>
                <td className="hc-bm-num">{row.pooled}</td>
                <RatioCell ratio={row.ratio} interval={row.interval} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hc-bm-caption">{everyManyNote}</p>
      <h3 className="hc-bm-subhead">The GX10 alone</h3>
      <p className="hc-bm-caption">{gx10ManyConditions}</p>
      <ConcurrencyTable rows={gx10Rows} label="The GX10 alone with 1, 4 and 8 chats" />
      <p className="hc-bm-caption">{gx10ManyNote}</p>
    </AllResults>
  </>;
}

export function NoHelpPanel() {
  return <>
    <Lede>Sharing can slow generation when one machine already holds most of the model.</Lede>
    <RatioChart spec={noHelpRatio} />
    <RangeChart spec={coldStart} />
    <AllResults>
      <ul className="hc-bm-notes">{noHelpReasons.map(note => <li key={note}>{note}</li>)}</ul>
    </AllResults>
  </>;
}
