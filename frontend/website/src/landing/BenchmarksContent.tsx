import { BarChart } from './BenchmarkCharts';
import { concurrentComparison, firstResponseComparison, generationComparison } from './benchmark-story';
import './benchmarks.css';
import './benchmark-sheet.css';

const sourceUrl = 'https://github.com/amemon12/Hypercompute/blob/main/docs/pitch/NUMBERS.md';

export function BenchmarksContent() {
  return (
    <main className="hc-benchmarks-content hc-benchmark-story" id="main-content" tabIndex={-1}>
      <header className="hc-benchmarks-intro" id="benchmarks-top">
        <h1>Benchmarks</h1>
        <p className="hc-benchmarks-deck">What changes when you borrow your friends’ computers?</p>
      </header>

      <section className="hc-bm-story-section" id="laptops" aria-label="Generation speed">
        <BarChart spec={generationComparison} />
        <p className="hc-bm-setup-line">Shared means multiple computers working together. Each model uses its recorded setup.</p>
        <p className="hc-bm-takeaway">Generation speed varies by model. DeepSeek is slower in this shared setup.</p>
        <p className="hc-bm-source">Kimi and Qwen: team-reported results. Sources are described in Run details.</p>
      </section>

      <section className="hc-bm-story-section" id="every-machine" aria-label="Time to first response">
        <span id="gx10" className="hc-bm-anchor" aria-hidden="true" />
        <BarChart spec={firstResponseComparison} />
        <p className="hc-bm-takeaway">All five comparisons show less waiting for the first response. DeepSeek still generates subsequent tokens more slowly.</p>
      </section>

      <section className="hc-bm-story-section" id="many-users" aria-label="Total output across chats">
        <BarChart spec={concurrentComparison} />
        <p className="hc-bm-setup-line">gpt-oss-120b · total output across simultaneous chats.</p>
        <p className="hc-bm-takeaway">More chats increase total output. With eight chats sharing the pool, each gets about <strong>2.05 tokens/s</strong>, down from 8.74 with one chat.</p>
      </section>

      <details className="hc-bm-details" id="no-help">
        <summary>Run details</summary>
        <dl className="hc-bm-run-details">
          <div>
            <dt>Llama, gpt-oss and Qwen</dt>
            <dd>Llama, gpt-oss and Qwen3-Next compare an RTX 5070 Ti laptop alone with the same laptop plus two M4 MacBook Airs. Llama: Q4_K_M, 64 / 32 prompt / output tokens, 2 repeats per setup. gpt-oss: MXFP4, 128 / 64 tokens, 3 repeats per setup. Qwen3-Next: Q3_K_M, 128 / 64 tokens, 9 recorded repeats across 3 runs. Home network; each Air contributed 10.5 GiB.</dd>
            <dd>Shared / alone ratios and 95% intervals: Llama 5.44× [5.42, 5.47]; gpt-oss 2.28× [2.00, 2.54]. Qwen3-Next: 10.005 → 21.373 tokens/s, 2.13×; first response 4.31 s → 1.56 s. Llama’s result is one pair of runs and still below 1 token/s.</dd>
          </div>
          <div>
            <dt>DeepSeek and Kimi</dt>
            <dd>Recorded setups: an ASUS Ascent GX10 alone versus the GX10 plus an RTX 5070 Ti laptop and three M4 MacBook Airs. DeepSeek-V3.1 671B: UD-TQ1_0. Kimi-K2.6 1T: IQ1_S. Both use SSD-backed weights, 64 / 32 prompt / output tokens and 4 repeats per setup. DeepSeek included a relayed peer; Kimi’s peers were direct.</dd>
            <dd>Generation speed, alone → shared: DeepSeek 1.805 → 1.497 tokens/s, 0.83× [0.76, 0.91]. Kimi: 1.540 → 1.799 tokens/s, 1.16×. Kimi’s first response is 28.25 s alone and 19.08 s shared.</dd>
          </div>
          <div>
            <dt>Sources</dt>
            <dd>Qwen’s generation and first-response values, and Kimi’s generation values, are team-reported results confirmed by the project owner on October 25, 2026. They were not recomputed from the archived run records; no confidence intervals are shown for these reported generation ratios. Kimi’s first-response times and the other comparisons use the archived results.</dd>
          </div>
          <div>
            <dt>Concurrent chats</dt>
            <dd>gpt-oss-120b, MXFP4, 128 / 64 prompt / output tokens. RTX 5070 Ti laptop alone versus the same laptop plus two M4 MacBook Airs. Three repeats at one chat; three batches at four and eight chats. Rates count all chats generating together.</dd>
            <dd>Shared / alone ratios: 2.28× [2.00, 2.54] at one chat; 1.46× [1.29, 1.68] at four; 1.34× [1.24, 1.48] at eight.</dd>
          </div>
          <div>
            <dt>Scope</dt>
            <dd>Short prompts, runtime build b11770. Intervals describe variation across repeats, often within one model load. Loading and peer seeding are excluded; a cold shared start in these tests took roughly 15–32 minutes.</dd>
          </div>
        </dl>
      </details>
      <p className="hc-bm-source"><a href={sourceUrl} target="_blank" rel="noreferrer">Original results and source data</a></p>
    </main>
  );
}
