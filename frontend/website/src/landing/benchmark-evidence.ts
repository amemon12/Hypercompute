/**
 * Curated from successful raw run.json records at the pinned revision below.
 * These records are observations, not forecasts. Warmup-labelled samples and
 * refused/failed runs are excluded. Qwen 235B uses all nine steady-state reps
 * (D4/D5/D6), not the fastest single run. The matching baseline is A3/A4/A5.
 * No successful distributed DeepSeek 671B run exists in this evidence set.
 * Host SSD counts are system-wide physical reads during decode, not energy.
 * Parameter counts/active counts come from the linked model-owner cards.
 */

export type Sample = {
  run: string;
  rep: string;
  promptTokens: number;
  outputTokens: number;
  requestSeconds: number;
  timeToFirstTokenSeconds: number;
  decodeTokensPerSecond: number;
  prefillTokensPerSecond: number;
  ssdReadMiBPerDecodedToken: number;
};

export type Batch = {
  label: string;
  loadSeconds: number;
  /** Includes setup, load, warmups, measured requests and teardown. */
  harnessWallSeconds: number;
  sourceUrl: string;
};

export type LargeModelRun = {
  id: string;
  model: string;
  parametersBillions: number;
  activeParametersBillions: number;
  modelCardUrl: string;
  quantization: string;
  mode: 'distributed' | 'local';
  peerCount: number;
  requestedPromptTokens: number;
  outputTokens: number;
  cohort: string;
  notes: string[];
  fileBytes: number;
  measuredSamples: Sample[];
  batches: Batch[];
};

export type MetricSummary = { mean: number; median: number; min: number; max: number; n: number };

export function summarizeMetric(values: readonly number[]): MetricSummary {
  if (!values.length || values.some(value => !Number.isFinite(value))) {
    throw new Error('Benchmark summaries require finite, nonempty measurements.');
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return {
    mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    median: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    min: sorted[0], max: sorted[sorted.length - 1], n: sorted.length,
  };
}

export function summarizeModel(run: Pick<LargeModelRun, 'measuredSamples' | 'batches'>) {
  const metric = (key: keyof Omit<Sample, 'run' | 'rep'>) => summarizeMetric(run.measuredSamples.map(sample => sample[key]));
  return {
    sampleCount: run.measuredSamples.length,
    requestSeconds: metric('requestSeconds'),
    timeToFirstTokenSeconds: metric('timeToFirstTokenSeconds'),
    decodeTokensPerSecond: metric('decodeTokensPerSecond'),
    prefillTokensPerSecond: metric('prefillTokensPerSecond'),
    ssdReadMiBPerDecodedToken: metric('ssdReadMiBPerDecodedToken'),
    promptTokens: metric('promptTokens'),
    loadSeconds: summarizeMetric(run.batches.map(batch => batch.loadSeconds)),
  };
}

/**
 * Extrapolation anchored at each measured request's end-to-end duration.
 * Extend (or shorten) generation at its recorded engine decode rate. This
 * avoids mixing first-visible-chunk TTFT with engine timing: 120B can generate
 * initial channel/formatting tokens before the first visible stream chunk.
 * Holds that sample's short prompt, hardware and cache conditions fixed.
 * The min/max are a sample envelope, NOT a confidence interval.
 * Loading, downloads, seeding and warmup are excluded. Long contexts and
 * sustained generation have not been validated by these short tests.
 */
export function projectOutputRuntime(run: LargeModelRun, outputTokens: number): MetricSummary {
  if (!Number.isFinite(outputTokens) || outputTokens < 1) {
    throw new Error('Output length must be at least one token.');
  }
  return summarizeMetric(run.measuredSamples.map(sample =>
    sample.requestSeconds + (Math.floor(outputTokens) - sample.outputTokens) / sample.decodeTokensPerSecond,
  ));
}

export const benchmarkSources = {
  "commit": "bef20200c7be7e6a0bc180df7a11f602fc97df82",
  "paper": "https://github.com/amemon12/Hypercompute/blob/main/docs/research/PAPER-RUNS-2026-10-22.md",
  "checkpoint": "https://github.com/amemon12/Hypercompute/blob/main/docs/research/CHECKPOINT-2026-10-23.md",
  "qwenIndex": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/paper-2026-10-22/index_235b_tight.jsonl",
  "gptOssIndex": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/paper-2026-10-22/index_120b.jsonl",
  "deepseekSummary": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/paper-2026-10-23/summary_671b.md",
  "hotspotFailures": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/paper-2026-10-23/summary_235b_hs.md"
} as const;

export const largeModelRuns: LargeModelRun[] = [
  {
    "id": "qwen-235b",
    "model": "Qwen3-235B-A22B",
    "parametersBillions": 235,
    "activeParametersBillions": 22,
    "modelCardUrl": "https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507",
    "quantization": "UD-Q2_K_XL",
    "mode": "distributed",
    "peerCount": 3,
    "requestedPromptTokens": 64,
    "outputTokens": 64,
    "cohort": "Steady-state runs D4, D5 and D6",
    "notes": [
      "G14 plus three M4 MacBook Airs. Nine measured requests; warmups excluded.",
      "Macs hold 22 of 94 model layers. Routed experts for the remaining layers stay file-backed on the host NVMe.",
      "These are quantized mixture-of-experts inference runs, not a model quality evaluation.",
      "An earlier cold 235B distributed load took 2,289 seconds. The listed steady-state load times use populated peer caches.",
      "The later hotspot batch had no valid distributed run; it is excluded from this cohort."
    ],
    "fileBytes": 88767695488,
    "measuredSamples": [
      {
        "run": "235b-3mac-4",
        "rep": "rep0",
        "promptTokens": 63,
        "outputTokens": 64,
        "requestSeconds": 70.705481,
        "timeToFirstTokenSeconds": 25.678872,
        "decodeTokensPerSecond": 1.399023016,
        "prefillTokensPerSecond": 2.458133691,
        "ssdReadMiBPerDecodedToken": 310.286691
      },
      {
        "run": "235b-3mac-4",
        "rep": "rep1",
        "promptTokens": 62,
        "outputTokens": 64,
        "requestSeconds": 81.276165,
        "timeToFirstTokenSeconds": 29.221187,
        "decodeTokensPerSecond": 1.210168318,
        "prefillTokensPerSecond": 2.126358366,
        "ssdReadMiBPerDecodedToken": 309.440399
      },
      {
        "run": "235b-3mac-4",
        "rep": "rep2",
        "promptTokens": 64,
        "outputTokens": 64,
        "requestSeconds": 76.615185,
        "timeToFirstTokenSeconds": 27.199853,
        "decodeTokensPerSecond": 1.274782015,
        "prefillTokensPerSecond": 2.357458483,
        "ssdReadMiBPerDecodedToken": 310.142206
      },
      {
        "run": "235b-3mac-5",
        "rep": "rep0",
        "promptTokens": 66,
        "outputTokens": 64,
        "requestSeconds": 69.245869,
        "timeToFirstTokenSeconds": 26.046682,
        "decodeTokensPerSecond": 1.458235679,
        "prefillTokensPerSecond": 2.541443531,
        "ssdReadMiBPerDecodedToken": 312.226082
      },
      {
        "run": "235b-3mac-5",
        "rep": "rep1",
        "promptTokens": 66,
        "outputTokens": 64,
        "requestSeconds": 74.716487,
        "timeToFirstTokenSeconds": 26.47588,
        "decodeTokensPerSecond": 1.305742862,
        "prefillTokensPerSecond": 2.497995642,
        "ssdReadMiBPerDecodedToken": 298.515346
      },
      {
        "run": "235b-3mac-5",
        "rep": "rep2",
        "promptTokens": 64,
        "outputTokens": 64,
        "requestSeconds": 75.716697,
        "timeToFirstTokenSeconds": 34.565752,
        "decodeTokensPerSecond": 1.530826769,
        "prefillTokensPerSecond": 1.854354587,
        "ssdReadMiBPerDecodedToken": 293.109453
      },
      {
        "run": "235b-3mac-6",
        "rep": "rep0",
        "promptTokens": 64,
        "outputTokens": 64,
        "requestSeconds": 67.175246,
        "timeToFirstTokenSeconds": 25.441809,
        "decodeTokensPerSecond": 1.509430816,
        "prefillTokensPerSecond": 2.521341879,
        "ssdReadMiBPerDecodedToken": 289.774391
      },
      {
        "run": "235b-3mac-6",
        "rep": "rep1",
        "promptTokens": 66,
        "outputTokens": 64,
        "requestSeconds": 70.125011,
        "timeToFirstTokenSeconds": 27.717219,
        "decodeTokensPerSecond": 1.485527216,
        "prefillTokensPerSecond": 2.386093414,
        "ssdReadMiBPerDecodedToken": 232.356577
      },
      {
        "run": "235b-3mac-6",
        "rep": "rep2",
        "promptTokens": 64,
        "outputTokens": 64,
        "requestSeconds": 73.639623,
        "timeToFirstTokenSeconds": 28.425565,
        "decodeTokensPerSecond": 1.393227213,
        "prefillTokensPerSecond": 2.257209041,
        "ssdReadMiBPerDecodedToken": 306.096951
      }
    ],
    "batches": [
      {
        "label": "235b-3mac-4",
        "loadSeconds": 151.591,
        "harnessWallSeconds": 544.8,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261022T233750Z-big-qwen3-235b-a22b-instruct-2507-ud-q2_k_xl/run.json"
      },
      {
        "label": "235b-3mac-5",
        "loadSeconds": 130.268,
        "harnessWallSeconds": 508.2,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261022T234753Z-big-qwen3-235b-a22b-instruct-2507-ud-q2_k_xl/run.json"
      },
      {
        "label": "235b-3mac-6",
        "loadSeconds": 175.498,
        "harnessWallSeconds": 559.0,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261023T000553Z-big-qwen3-235b-a22b-instruct-2507-ud-q2_k_xl/run.json"
      }
    ]
  },
  {
    "id": "deepseek-671b",
    "model": "DeepSeek-V3.1",
    "parametersBillions": 671,
    "activeParametersBillions": 37,
    "modelCardUrl": "https://huggingface.co/deepseek-ai/DeepSeek-V3.1",
    "quantization": "UD-TQ1_0",
    "mode": "local",
    "peerCount": 0,
    "requestedPromptTokens": 64,
    "outputTokens": 32,
    "cohort": "Local run 671b-local-1",
    "notes": [
      "G14 alone. Two measured requests after one warmup.",
      "About 10.1 GiB of dense weights and head on the GPU; about 149 GiB of routed experts file-backed on NVMe.",
      "The GGUF mixes IQ1_S through Q8_0 tensors despite its UD-TQ1_0 filename.",
      "The 32-token outputs are recorded in the reasoning channel; this proves token generation, not answer quality.",
      "The three-Mac layouts were planned but did not run. No distributed speedup is established for this model."
    ],
    "fileBytes": 170499768256,
    "measuredSamples": [
      {
        "run": "671b-local-1",
        "rep": "rep0",
        "promptTokens": 67,
        "outputTokens": 32,
        "requestSeconds": 157.871686,
        "timeToFirstTokenSeconds": 93.495687,
        "decodeTokensPerSecond": 0.481508834,
        "prefillTokensPerSecond": 0.717952569,
        "ssdReadMiBPerDecodedToken": 2526.113281
      },
      {
        "run": "671b-local-1",
        "rep": "rep1",
        "promptTokens": 65,
        "outputTokens": 32,
        "requestSeconds": 148.378873,
        "timeToFirstTokenSeconds": 83.855007,
        "decodeTokensPerSecond": 0.480433315,
        "prefillTokensPerSecond": 0.775782903,
        "ssdReadMiBPerDecodedToken": 2443.603579
      }
    ],
    "batches": [
      {
        "label": "671b-local-1",
        "loadSeconds": 50.088,
        "harnessWallSeconds": 534.1,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261023T104113Z-big-deepseek-v3.1-ud-tq1_0/run.json"
      }
    ]
  },
  {
    "id": "gpt-oss-120b",
    "model": "gpt-oss-120b",
    "parametersBillions": 117,
    "activeParametersBillions": 5.1,
    "modelCardUrl": "https://huggingface.co/openai/gpt-oss-120b",
    "quantization": "MXFP4",
    "mode": "distributed",
    "peerCount": 3,
    "requestedPromptTokens": 128,
    "outputTokens": 64,
    "cohort": "Runs D1, D2 and D3",
    "notes": [
      "G14 plus three M4 MacBook Airs. Nine measured requests; warmups excluded.",
      "Each Mac holds four complete layers; 12 of the 36 layers run on the Macs.",
      "A background code workflow shared the host during D2 and D3, affecting cache pressure.",
      "First cold start included 566 seconds of peer seeding before the 126-second load."
    ],
    "fileBytes": 63387346208,
    "measuredSamples": [
      {
        "run": "120b-3mac-1",
        "rep": "rep0",
        "promptTokens": 132,
        "outputTokens": 64,
        "requestSeconds": 23.242068,
        "timeToFirstTokenSeconds": 10.530855,
        "decodeTokensPerSecond": 4.732788632,
        "prefillTokensPerSecond": 13.341833024,
        "ssdReadMiBPerDecodedToken": 32.480856
      },
      {
        "run": "120b-3mac-1",
        "rep": "rep1",
        "promptTokens": 132,
        "outputTokens": 64,
        "requestSeconds": 23.146991,
        "timeToFirstTokenSeconds": 10.311547,
        "decodeTokensPerSecond": 4.63398574,
        "prefillTokensPerSecond": 13.87992623,
        "ssdReadMiBPerDecodedToken": 31.82418
      },
      {
        "run": "120b-3mac-1",
        "rep": "rep2",
        "promptTokens": 132,
        "outputTokens": 64,
        "requestSeconds": 29.095931,
        "timeToFirstTokenSeconds": 13.366686,
        "decodeTokensPerSecond": 3.761665417,
        "prefillTokensPerSecond": 10.728041467,
        "ssdReadMiBPerDecodedToken": 22.624426
      },
      {
        "run": "120b-3mac-2",
        "rep": "rep0",
        "promptTokens": 130,
        "outputTokens": 64,
        "requestSeconds": 22.21983,
        "timeToFirstTokenSeconds": 11.573848,
        "decodeTokensPerSecond": 5.655948644,
        "prefillTokensPerSecond": 11.786975265,
        "ssdReadMiBPerDecodedToken": 24.275491
      },
      {
        "run": "120b-3mac-2",
        "rep": "rep1",
        "promptTokens": 131,
        "outputTokens": 64,
        "requestSeconds": 30.037191,
        "timeToFirstTokenSeconds": 12.467742,
        "decodeTokensPerSecond": 3.374529193,
        "prefillTokensPerSecond": 11.59540567,
        "ssdReadMiBPerDecodedToken": 38.927548
      },
      {
        "run": "120b-3mac-2",
        "rep": "rep2",
        "promptTokens": 128,
        "outputTokens": 64,
        "requestSeconds": 23.747674,
        "timeToFirstTokenSeconds": 13.060097,
        "decodeTokensPerSecond": 5.197752624,
        "prefillTokensPerSecond": 11.085563314,
        "ssdReadMiBPerDecodedToken": 44.615606
      },
      {
        "run": "120b-3mac-3",
        "rep": "rep0",
        "promptTokens": 132,
        "outputTokens": 64,
        "requestSeconds": 29.095701,
        "timeToFirstTokenSeconds": 13.661328,
        "decodeTokensPerSecond": 3.723350122,
        "prefillTokensPerSecond": 10.906412078,
        "ssdReadMiBPerDecodedToken": 51.712891
      },
      {
        "run": "120b-3mac-3",
        "rep": "rep1",
        "promptTokens": 131,
        "outputTokens": 64,
        "requestSeconds": 36.89761,
        "timeToFirstTokenSeconds": 18.187919,
        "decodeTokensPerSecond": 3.152364431,
        "prefillTokensPerSecond": 7.757907795,
        "ssdReadMiBPerDecodedToken": 70.516943
      },
      {
        "run": "120b-3mac-3",
        "rep": "rep2",
        "promptTokens": 131,
        "outputTokens": 64,
        "requestSeconds": 33.903185,
        "timeToFirstTokenSeconds": 17.761537,
        "decodeTokensPerSecond": 3.546641688,
        "prefillTokensPerSecond": 8.1492961,
        "ssdReadMiBPerDecodedToken": 55.93478
      }
    ],
    "batches": [
      {
        "label": "120b-3mac-1",
        "loadSeconds": 126.184,
        "harnessWallSeconds": 877.1,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261023T002843Z-big-gpt-oss-120b-mxfp4/run.json"
      },
      {
        "label": "120b-3mac-2",
        "loadSeconds": 158.262,
        "harnessWallSeconds": 347.2,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261023T005040Z-big-gpt-oss-120b-mxfp4/run.json"
      },
      {
        "label": "120b-3mac-3",
        "loadSeconds": 167.227,
        "harnessWallSeconds": 377.0,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261023T011537Z-big-gpt-oss-120b-mxfp4/run.json"
      }
    ]
  }
];

export const localBaselines: { id: string; measuredSamples: Sample[]; batches: Batch[] }[] = [
  {
    "id": "qwen-235b",
    "measuredSamples": [
      {
        "run": "235b-local-3",
        "rep": "rep0",
        "promptTokens": 60,
        "outputTokens": 64,
        "requestSeconds": 96.013857,
        "timeToFirstTokenSeconds": 42.496254,
        "decodeTokensPerSecond": 1.17711212,
        "prefillTokensPerSecond": 1.413966201,
        "ssdReadMiBPerDecodedToken": 687.895934
      },
      {
        "run": "235b-local-3",
        "rep": "rep1",
        "promptTokens": 60,
        "outputTokens": 64,
        "requestSeconds": 97.224153,
        "timeToFirstTokenSeconds": 43.074939,
        "decodeTokensPerSecond": 1.16334639,
        "prefillTokensPerSecond": 1.394925267,
        "ssdReadMiBPerDecodedToken": 702.260936
      },
      {
        "run": "235b-local-3",
        "rep": "rep2",
        "promptTokens": 58,
        "outputTokens": 64,
        "requestSeconds": 106.417306,
        "timeToFirstTokenSeconds": 47.492878,
        "decodeTokensPerSecond": 1.068979264,
        "prefillTokensPerSecond": 1.22318868,
        "ssdReadMiBPerDecodedToken": 730.808082
      },
      {
        "run": "235b-local-4",
        "rep": "rep0",
        "promptTokens": 65,
        "outputTokens": 64,
        "requestSeconds": 94.219367,
        "timeToFirstTokenSeconds": 45.561743,
        "decodeTokensPerSecond": 1.294598481,
        "prefillTokensPerSecond": 1.429907685,
        "ssdReadMiBPerDecodedToken": 584.517904
      },
      {
        "run": "235b-local-4",
        "rep": "rep1",
        "promptTokens": 65,
        "outputTokens": 64,
        "requestSeconds": 98.962424,
        "timeToFirstTokenSeconds": 49.044733,
        "decodeTokensPerSecond": 1.261926861,
        "prefillTokensPerSecond": 1.326863303,
        "ssdReadMiBPerDecodedToken": 629.936562
      },
      {
        "run": "235b-local-4",
        "rep": "rep2",
        "promptTokens": 64,
        "outputTokens": 64,
        "requestSeconds": 82.949967,
        "timeToFirstTokenSeconds": 40.987239,
        "decodeTokensPerSecond": 1.501208473,
        "prefillTokensPerSecond": 1.564349454,
        "ssdReadMiBPerDecodedToken": 611.180997
      },
      {
        "run": "235b-local-5",
        "rep": "rep0",
        "promptTokens": 65,
        "outputTokens": 64,
        "requestSeconds": 92.084408,
        "timeToFirstTokenSeconds": 43.021891,
        "decodeTokensPerSecond": 1.283941189,
        "prefillTokensPerSecond": 1.513331355,
        "ssdReadMiBPerDecodedToken": 611.945824
      },
      {
        "run": "235b-local-5",
        "rep": "rep1",
        "promptTokens": 62,
        "outputTokens": 64,
        "requestSeconds": 89.382252,
        "timeToFirstTokenSeconds": 39.448936,
        "decodeTokensPerSecond": 1.261853309,
        "prefillTokensPerSecond": 1.574372586,
        "ssdReadMiBPerDecodedToken": 672.088697
      },
      {
        "run": "235b-local-5",
        "rep": "rep2",
        "promptTokens": 65,
        "outputTokens": 64,
        "requestSeconds": 78.901968,
        "timeToFirstTokenSeconds": 40.372402,
        "decodeTokensPerSecond": 1.635143013,
        "prefillTokensPerSecond": 1.613461572,
        "ssdReadMiBPerDecodedToken": 590.281862
      }
    ],
    "batches": [
      {
        "label": "235b-local-3",
        "loadSeconds": 52.093,
        "harnessWallSeconds": 657.8,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261022T231710Z-big-qwen3-235b-a22b-instruct-2507-ud-q2_k_xl/run.json"
      },
      {
        "label": "235b-local-4",
        "loadSeconds": 52.1,
        "harnessWallSeconds": 573.2,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261022T232815Z-big-qwen3-235b-a22b-instruct-2507-ud-q2_k_xl/run.json"
      },
      {
        "label": "235b-local-5",
        "loadSeconds": 44.064,
        "harnessWallSeconds": 526.5,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261022T235644Z-big-qwen3-235b-a22b-instruct-2507-ud-q2_k_xl/run.json"
      }
    ]
  },
  {
    "id": "gpt-oss-120b",
    "measuredSamples": [
      {
        "run": "120b-local-1",
        "rep": "rep0",
        "promptTokens": 130,
        "outputTokens": 64,
        "requestSeconds": 58.710262,
        "timeToFirstTokenSeconds": 39.511558,
        "decodeTokensPerSecond": 2.979612216,
        "prefillTokensPerSecond": 3.471980278,
        "ssdReadMiBPerDecodedToken": 286.903909
      },
      {
        "run": "120b-local-1",
        "rep": "rep1",
        "promptTokens": 128,
        "outputTokens": 64,
        "requestSeconds": 60.374204,
        "timeToFirstTokenSeconds": 39.362125,
        "decodeTokensPerSecond": 2.89270901,
        "prefillTokensPerSecond": 3.325481992,
        "ssdReadMiBPerDecodedToken": 272.516501
      },
      {
        "run": "120b-local-1",
        "rep": "rep2",
        "promptTokens": 128,
        "outputTokens": 64,
        "requestSeconds": 54.530155,
        "timeToFirstTokenSeconds": 36.838175,
        "decodeTokensPerSecond": 3.381024019,
        "prefillTokensPerSecond": 3.57499369,
        "ssdReadMiBPerDecodedToken": 289.084806
      },
      {
        "run": "120b-local-2",
        "rep": "rep0",
        "promptTokens": 130,
        "outputTokens": 64,
        "requestSeconds": 55.794447,
        "timeToFirstTokenSeconds": 36.08884,
        "decodeTokensPerSecond": 3.07156363,
        "prefillTokensPerSecond": 3.694013811,
        "ssdReadMiBPerDecodedToken": 270.603516
      },
      {
        "run": "120b-local-2",
        "rep": "rep1",
        "promptTokens": 128,
        "outputTokens": 64,
        "requestSeconds": 49.978368,
        "timeToFirstTokenSeconds": 33.802221,
        "decodeTokensPerSecond": 3.721000559,
        "prefillTokensPerSecond": 3.886348593,
        "ssdReadMiBPerDecodedToken": 258.417628
      },
      {
        "run": "120b-local-2",
        "rep": "rep2",
        "promptTokens": 129,
        "outputTokens": 64,
        "requestSeconds": 53.883053,
        "timeToFirstTokenSeconds": 36.66723,
        "decodeTokensPerSecond": 3.497297172,
        "prefillTokensPerSecond": 3.60325825,
        "ssdReadMiBPerDecodedToken": 286.093657
      },
      {
        "run": "120b-local-3",
        "rep": "rep0",
        "promptTokens": 130,
        "outputTokens": 64,
        "requestSeconds": 53.526467,
        "timeToFirstTokenSeconds": 34.696555,
        "decodeTokensPerSecond": 3.158625258,
        "prefillTokensPerSecond": 3.877419961,
        "ssdReadMiBPerDecodedToken": 260.518772
      },
      {
        "run": "120b-local-3",
        "rep": "rep1",
        "promptTokens": 128,
        "outputTokens": 64,
        "requestSeconds": 50.846811,
        "timeToFirstTokenSeconds": 34.381208,
        "decodeTokensPerSecond": 3.649229989,
        "prefillTokensPerSecond": 3.820233995,
        "ssdReadMiBPerDecodedToken": 264.712247
      },
      {
        "run": "120b-local-3",
        "rep": "rep2",
        "promptTokens": 129,
        "outputTokens": 64,
        "requestSeconds": 52.313616,
        "timeToFirstTokenSeconds": 34.181075,
        "decodeTokensPerSecond": 3.310665783,
        "prefillTokensPerSecond": 3.885664825,
        "ssdReadMiBPerDecodedToken": 286.374581
      }
    ],
    "batches": [
      {
        "label": "120b-local-1",
        "loadSeconds": 38.071,
        "harnessWallSeconds": 394.9,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261023T004343Z-big-gpt-oss-120b-mxfp4/run.json"
      },
      {
        "label": "120b-local-2",
        "loadSeconds": 44.07,
        "harnessWallSeconds": 379.0,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261023T010236Z-big-gpt-oss-120b-mxfp4/run.json"
      },
      {
        "label": "120b-local-3",
        "loadSeconds": 42.071,
        "harnessWallSeconds": 385.7,
        "sourceUrl": "https://github.com/amemon12/Hypercompute/blob/main/results/capacity/20261023T012217Z-big-gpt-oss-120b-mxfp4/run.json"
      }
    ]
  }
];

// Published ratios of means and 95% percentile bootstrap intervals (10,000 resamples, seed 7).
export const poolingComparisons = [
  {
    "id": "qwen-235b",
    "model": "Qwen3-235B-A22B",
    "localSamples": 9,
    "distributedSamples": 9,
    "cohort": "Steady state: A3/A4/A5 versus D4/D5/D6",
    "decode": {
      "localMean": 1.294,
      "distributedMean": 1.396,
      "ratio": 1.08,
      "confidence95": [
        0.98,
        1.18
      ]
    },
    "prefill": {
      "localMean": 1.45,
      "distributedMean": 2.333,
      "ratio": 1.61,
      "confidence95": [
        1.48,
        1.74
      ]
    },
    "ssdMiBPerToken": {
      "localMean": 647,
      "distributedMean": 296,
      "ratio": 0.46,
      "confidence95": [
        0.42,
        0.49
      ]
    },
    "note": "The decode interval includes parity. Prefill and host SSD reads improved; warm decode did not reliably speed up on this Wi-Fi."
  },
  {
    "id": "gpt-oss-120b",
    "model": "gpt-oss-120b",
    "localSamples": 9,
    "distributedSamples": 9,
    "cohort": "A1/A2/A3 versus D1/D2/D3",
    "decode": {
      "localMean": 3.296,
      "distributedMean": 4.198,
      "ratio": 1.27,
      "confidence95": [
        1.1,
        1.46
      ]
    },
    "prefill": {
      "localMean": 3.68,
      "distributedMean": 11.03,
      "ratio": 2.99,
      "confidence95": [
        2.64,
        3.35
      ]
    },
    "ssdMiBPerToken": {
      "localMean": 275,
      "distributedMean": 41,
      "ratio": 0.15,
      "confidence95": [
        0.12,
        0.19
      ]
    },
    "note": "The host shared a background code workflow for part of the experiment. Moving from one to three Macs did not clearly improve decode."
  }
] as const;
