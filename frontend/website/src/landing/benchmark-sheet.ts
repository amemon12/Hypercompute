// Measured results for the public benchmarks page, copied from the team's numbers sheet.
// Absolute values are the sheet's table values. Ratios use the sheet's public roundings:
// toward the less impressive side, with intervals widened outward. Brackets are 95% intervals.

export type Tone = 'gain' | 'parity' | 'loss';

export interface Detail {
  readonly label: string;
  readonly value: string;
}

export interface ResultCard {
  readonly model: string;
  readonly spec: string;
  readonly headline: string;
  readonly metric: string;
  readonly interval: string;
  readonly tone: Tone;
  readonly summary: string;
  readonly details: readonly Detail[];
}

export interface ConcurrencyRow {
  readonly model: string;
  readonly spec: string;
  readonly lengths: string;
  readonly totals: readonly [string, string, string];
  readonly eightVsOne: string;
  readonly interval: string;
}

export const helpsWhen =
  'Pooling helps when no single machine can hold the model and the host would otherwise page weights from its SSD. It hurts when one machine already holds most of it.';

export const laptopConditions =
  'A gaming laptop (RTX 5070 Ti, 12 GB) alone, then the same laptop with two MacBook Airs (M4, 16 GB, 10.5 GiB shared each). Home network, both Airs direct, 5 to 8 ms before each pooled run. One run per side.';

export const laptopCards: readonly ResultCard[] = [
  {
    model: 'Llama 3.3 70B Instruct',
    spec: 'Q4_K_M · dense · about 40 GiB',
    headline: '5.44×',
    metric: 'decode, pooled vs alone',
    interval: '[5.42, 5.47]',
    tone: 'gain',
    summary: '0.165 → 0.898 tokens/s',
    details: [
      { label: 'Prompt / answer', value: '64 / 32 tokens, 2 reps per side' },
      { label: 'First token', value: '10.02 s → 5.01 s' },
      { label: 'Full answer', value: '197.91 s → 39.53 s' },
      { label: 'SSD reads per token', value: '23,454 → 2,761 MiB' },
      { label: 'Layers', value: 'Alone, 61 of 80 read from SSD. Pooled, 20 on each Air.' },
      { label: 'Caveat', value: 'One pair of runs. A second alone run warmed up about 9% faster, so 5.44× is likely the high end.' },
    ],
  },
  {
    model: 'gpt-oss-120b',
    spec: 'MXFP4 · MoE · 59 GiB',
    headline: '2.28×',
    metric: 'decode, pooled vs alone',
    interval: '[2.00, 2.54]',
    tone: 'gain',
    summary: 'First token 7.08× sooner [6.20, 8.23], 34.23 s → 4.83 s',
    details: [
      { label: 'Prompt / answer', value: '128 / 64 tokens, 3 reps per side' },
      { label: 'Decode', value: '3.829 → 8.740 tokens/s' },
      { label: 'Prompt reading', value: '7.75× faster [6.94, 8.86]' },
      { label: 'Full answer', value: '49.88 s → 11.67 s' },
      { label: 'SSD reads per token', value: '238.3 → 6.4 MiB' },
      { label: 'Layers', value: '6 on each Air, 24 on the laptop GPU, the rest of the experts read from SSD.' },
    ],
  },
  {
    model: 'Qwen2.5-VL 72B Instruct',
    spec: 'Q4_K_M · dense · text only',
    headline: '2.59×',
    metric: 'decode, pooled vs alone',
    interval: '[2.57, 2.62]',
    tone: 'gain',
    summary: '0.141 → 0.365 tokens/s',
    details: [
      { label: 'Prompt / answer', value: '64 / 32 tokens, 2 reps per side' },
      { label: 'First token', value: '13.70 s → 9.10 s' },
      { label: 'Full answer', value: '233.60 s → 93.93 s' },
      { label: 'SSD reads per token', value: '26,896 → 8,956 MiB' },
      { label: 'Layers', value: 'Alone, 64 of 80 read from SSD. Pooled, 18 on each Air.' },
      { label: 'Scope', value: 'Text only. The vision projector was not loaded.' },
    ],
  },
  {
    model: 'Qwen3-Next 80B-A3B',
    spec: 'Q3_K_M · MoE · 3B active',
    headline: '0.46×',
    metric: 'decode, slower pooled',
    interval: '[0.36, 0.62]',
    tone: 'loss',
    summary: '21.373 → 10.005 tokens/s',
    details: [
      { label: 'Prompt / answer', value: '128 / 64 tokens, 6 alone reps (2 runs) vs 3 pooled' },
      { label: 'First token', value: '1.56 s → 4.31 s, later pooled' },
      { label: 'Full answer', value: '4.76 s → 10.65 s' },
      { label: 'Why', value: 'It fits the laptop. Only 3B parameters are active per token, so the Airs remove almost no disk reads and each adds a network round trip per token.' },
    ],
  },
];

export const laptopManyConditions =
  'gpt-oss-120b on the gaming laptop alone and with two MacBook Airs, home network, 128 / 64 tokens, 3 reps at one chat and 3 batches at 4 and 8.';

export const laptopManyNote =
  'More chats get more total output, and each answer takes longer: at 8 chats each pooled chat got about 2.05 tok/s. The pool’s lead shrinks as chats are added.';

export const gx10Conditions =
  'One ASUS Ascent GX10 (NVIDIA GB10, 121.6 GiB unified memory). One machine, no pooling.';

export interface SingleRow {
  readonly model: string;
  readonly spec: string;
  readonly lengths: string;
  readonly decode: string;
  readonly first: string;
  readonly answer: string;
  readonly ssd: string;
}

export const gx10Single: readonly SingleRow[] = [
  { model: 'gpt-oss-120b', spec: 'MXFP4 · 59 GiB · all in GPU memory', lengths: '128 / 64', decode: '54.621', first: '0.42 s', answer: '1.52 s', ssd: 'none' },
  { model: 'Gemma 3 27B IT', spec: 'Q4_K_M · all in GPU memory', lengths: '128 / 64', decode: '12.341', first: '0.35 s', answer: '5.46 s', ssd: 'none' },
  { model: 'QwQ 32B', spec: 'Q4_K_M · all in GPU memory', lengths: '128 / 64', decode: '10.643', first: '0.23 s', answer: '6.15 s', ssd: 'none' },
  { model: 'Llama 4 Maverick 400B', spec: 'UD-Q2_K_XL · 142 GiB · 17B active', lengths: '128 / 64', decode: '9.785', first: '2.41 s', answer: '8.86 s', ssd: '53.1 MiB' },
  { model: 'DeepSeek-V3.1 671B', spec: 'UD-TQ1_0 · 158.8 GiB · 37B active', lengths: '64 / 32', decode: '1.792', first: '23.49 s', answer: '40.77 s', ssd: '686.3 MiB' },
  { model: 'Kimi-K2.6 1T', spec: 'IQ1_S · 193.16 GiB · 32B active', lengths: '64 / 32', decode: '1.779', first: '27.01 s', answer: '44.42 s', ssd: '657.3 MiB' },
];

export const gx10Notes: readonly string[] = [
  'One chat at a time: 6 batches for gpt-oss-120b, 3 reps for Gemma, QwQ and Maverick, 2 for the 671B and 1T.',
  'gpt-oss-120b, Gemma and QwQ fit in GPU memory. Maverick, the 671B and the 1T are larger than the memory, so the GX10 reads part of their experts from its SSD.',
  'gpt-oss-120b in a separate run of 9 reps averaged 56.00 tok/s while a model download ran on the box.',
  'Mistral Large 2411 (123B dense, 121.33 GiB, about the size of the whole memory) did not load on the GX10 alone. It was not tried pooled.',
];

export const gx10ManyConditions =
  'The GX10 alone, one machine, no pooling. Total tok/s with 1, 4 and 8 chats at once.';

export const gx10Rows: readonly ConcurrencyRow[] = [
  { model: 'gpt-oss-120b', spec: 'MXFP4 · 59 GiB', lengths: '128 / 64', totals: ['54.621', '111.861', '149.156'], eightVsOne: '2.73×', interval: '[2.58, 2.91]' },
  { model: 'Gemma 3 27B IT', spec: 'Q4_K_M', lengths: '128 / 64', totals: ['12.341', '45.282', '85.044'], eightVsOne: '6.89×', interval: '[6.88, 6.91]' },
  { model: 'QwQ 32B', spec: 'Q4_K_M', lengths: '128 / 64', totals: ['10.643', '39.125', '75.169'], eightVsOne: '7.06×', interval: '[7.04, 7.08]' },
  { model: 'Llama 4 Maverick 400B', spec: 'UD-Q2_K_XL · 142 GiB', lengths: '128 / 64', totals: ['9.785', '29.859', '43.362'], eightVsOne: '4.43×', interval: '[4.08, 4.81]' },
  { model: 'DeepSeek-V3.1 671B', spec: 'UD-TQ1_0 · 158.8 GiB', lengths: '64 / 32', totals: ['1.792', '3.444', '5.547'], eightVsOne: '3.09×', interval: '[2.52, 3.69]' },
  { model: 'Kimi-K2.6 1T', spec: 'IQ1_S · 193.16 GiB', lengths: '64 / 32', totals: ['1.779', '3.250', '4.640'], eightVsOne: '2.60×', interval: '[2.37, 2.85]' },
];

export const gx10ManyNote =
  'Each chat slows as more join: the full answer per chat went from 1.52 s to 4.30 s on gpt-oss-120b and from 40.77 s to 92.66 s on the 671B, from one chat to eight. Batches per count: 6 for gpt-oss-120b, 3 for Gemma, QwQ and Maverick, 2 for the 671B and 1T.';

export const everyManyConditions =
  'The GX10 alone against the GX10 with the gaming laptop and two MacBook Airs, same chat count, one run per side. Total tok/s.';

export const everyManyRows = [
  { model: 'Llama 4 Maverick 400B', chats: '4', alone: '29.859', pooled: '9.826', ratio: '0.32×', interval: '[0.31, 0.35]' },
  { model: 'Llama 4 Maverick 400B', chats: '8', alone: '43.362', pooled: '15.071', ratio: '0.34×', interval: '[0.32, 0.37]' },
  { model: 'Kimi-K2.6 1T', chats: '8', alone: '4.640', pooled: '4.986', ratio: '1.07×', interval: '[0.97, 1.20]' },
] as const;

export const everyManyNote = 'The 1T interval includes no gain, and its alone runs came about 4 h earlier, in another session.';

export const everyConditions =
  'The ASUS Ascent GX10 hosts, with the gaming laptop (9 GiB shared) and two MacBook Airs (11.5 GiB shared each) as peers. Home network, all peers direct, 5 to 16 ms before each pooled run. One run per side unless noted.';

export const everyCards: readonly ResultCard[] = [
  {
    model: 'Kimi-K2.6 1T',
    spec: 'IQ1_S · 193 GiB · 32B active',
    headline: '0.99×',
    metric: 'decode, at parity',
    interval: '[0.96, 1.03]',
    tone: 'parity',
    summary: 'First token 25% sooner [22%, 28%], 27.01 s → 20.22 s',
    details: [
      { label: 'Prompt / answer', value: '64 / 32 tokens, 2 reps per side' },
      { label: 'Prompt reading', value: '1.43× faster [1.39, 1.48]' },
      { label: 'Full answer', value: '44.42 s → 37.70 s, 15% sooner [12%, 19%]' },
      { label: 'SSD reads per token', value: '657.3 → 333.8 MiB' },
      { label: '8 chats at once', value: '4.640 → 4.986 tokens/s total, 1.07× [0.97, 1.20], no clear gain' },
      { label: 'Caveat', value: 'The alone runs came about 4 h earlier, in another session.' },
    ],
  },
  {
    model: 'DeepSeek-V3.1 671B',
    spec: 'UD-TQ1_0 · 158.8 GiB · 37B active',
    headline: '1.00×',
    metric: 'decode, at parity',
    interval: '[0.92, 1.09]',
    tone: 'parity',
    summary: 'First token 1.58× sooner, 25.02 s → 15.77 s',
    details: [
      { label: 'Prompt / answer', value: '64 / 32 tokens, 4 vs 4 reps' },
      { label: 'Order', value: 'Alone and pooled runs alternating in one session' },
      { label: 'Prompt reading', value: '1.58× faster [1.47, 1.71]' },
      { label: 'Full answer', value: '42.63 s → 33.37 s' },
      { label: 'SSD reads per token', value: '698.3 → 260.6 MiB' },
    ],
  },
  {
    model: 'Llama 4 Maverick 400B',
    spec: 'UD-Q2_K_XL · 142 GiB · 17B active',
    headline: '0.43×',
    metric: 'decode, slower pooled',
    interval: '[0.40, 0.48]',
    tone: 'loss',
    summary: '9.785 → 4.277 tokens/s. It runs best on the GX10 alone.',
    details: [
      { label: 'Prompt / answer', value: '128 / 64 tokens, 3 reps per side' },
      { label: 'Full answer', value: '8.86 s → 18.71 s, 2.12× as long [1.95, 2.28]' },
      { label: '4 chats at once', value: '29.859 → 9.826 tokens/s total, 0.32× [0.31, 0.35]' },
      { label: '8 chats at once', value: '43.362 → 15.071 tokens/s total, 0.34× [0.32, 0.37]' },
      { label: 'SSD reads per token', value: '53.1 → 13.8 MiB' },
    ],
  },
];

export const fourPeers = {
  title: 'A day earlier, four peers',
  conditions: 'The GX10 with the gaming laptop and three MacBook Airs on home Wi-Fi. One Air was on the relay in the 671B runs, which mix two layouts. 64 / 32 tokens, 4 vs 4 reps, alone and pooled alternating.',
  rows: [
    { model: 'DeepSeek-V3.1 671B', prompt: '2.41× [2.19, 2.73]', first: '2.38× sooner [2.15, 2.69]', firstTimes: '24.67 s → 10.36 s', decode: '0.83× [0.76, 0.91]', ssd: '683.6 → 37.1 MiB' },
    { model: 'Kimi-K2.6 1T', prompt: '1.50× [1.41, 1.59]', first: '1.48× sooner [1.37, 1.60]', firstTimes: '28.25 s → 19.08 s', decode: '0.85× [0.81, 0.89]', ssd: '671.0 → 254.1 MiB' },
  ],
} as const;

export const earlierConditions =
  'The gaming laptop alone, then with MacBook Airs at 7.5 GiB shared each. Building Wi-Fi, laptops on different subnets, peers direct through NAT. 9 reps per side unless noted.';

export const earlierRows = [
  { model: 'gpt-oss-120b', pool: '+ 3 Airs', lengths: '128 / 64', decode: '1.27×', decodeInterval: '[1.10, 1.47]', first: '36.17 s → 13.44 s', firstNote: '2.69× sooner [2.35, 3.08]', prompt: '2.99× [2.63, 3.35]' },
  { model: 'gpt-oss-120b', pool: '+ 1 Air, 6 pooled reps', lengths: '128 / 64', decode: '1.25×', decodeInterval: '[1.10, 1.41]', first: '36.17 s → 23.97 s', firstNote: '', prompt: '1.48× [1.40, 1.57]' },
  { model: 'Qwen3-235B-A22B', pool: '+ 3 Airs, steady state', lengths: '64 / 64', decode: '1.07×', decodeInterval: '[0.97, 1.19]', first: '43.50 s → 27.86 s', firstNote: '', prompt: '1.60× [1.48, 1.75]' },
  { model: 'DeepSeek-V3.1 671B', pool: 'laptop alone, 2 reps', lengths: '64 / 32', decode: '0.481', decodeInterval: 'tokens/s', first: '88.68 s', firstNote: '', prompt: '0.747 tokens/s' },
] as const;

export const earlierNote =
  'The 235B decode interval includes parity. Three Airs against one on the 120B: 1.01× [0.86, 1.20], no clear gain from the extra two.';

export const noHelpReasons: readonly string[] = [
  'Qwen3-Next 80B fits the laptop. Only 3B parameters are active per token, so the Airs remove almost no disk reads and each adds a network round trip per token.',
  'On the GX10 each peer is one blocking network round trip per token, and the SSD reads the peers save cost less than the extra stages. Maverick nearly fits the GX10, so it runs best there alone.',
  'Qwen3-235B on building Wi-Fi, and more Airs on gpt-oss-120b: once the host’s page cache is warm, extra stages cost about what they save.',
  'Every cold pooled start first seeds its peers from Hugging Face, so the first answer comes 15 to 32 minutes after starting.',
  'Our planner ranked the pool ahead on both GX10 models: 1.77× predicted on the 671B, against 0.83× measured.',
  'On client-isolated apartment Wi-Fi the peers reached the host only through the relay, at about 1 MiB/s, and those runs failed or were refused.',
];

export const method: readonly Detail[] = [
  { label: 'Machines', value: 'Gaming laptop: RTX 5070 Ti Laptop GPU (12 GB), 31 GiB RAM, Windows 11. MacBook Air: Apple M4, 16 GB. ASUS Ascent GX10: NVIDIA GB10, 121.6 GiB unified memory.' },
  { label: 'Runs', value: 'Temperature 0, seed 7, fixed-length prompts of 58 to 133 tokens, no prompt cache, every answer generated to full length. llama.cpp b11770, peers over Iroh.' },
  { label: 'Metrics', value: 'Decode is llama.cpp’s own generation rate. Prompt reading is prefill. SSD reads are the host’s system-wide physical reads per generated token.' },
  { label: 'Intervals', value: 'Ratios are pooled over alone, or N chats over one. Brackets are 95% percentile bootstrap intervals of the ratio of means, 10,000 resamples. With one run per side they show spread between reps inside one load, not between loads.' },
  { label: 'Charts', value: 'Every axis is linear from zero. Bars are means and carry no whiskers, because the sheet gives intervals on ratios, not on single means. Dots with whiskers are ratios with their 95% intervals.' },
  { label: 'Cold start', value: 'Timings exclude loading. A cold pooled start first seeds each peer from Hugging Face: about 15 to 16 minutes on the laptops and 17 to 32 on the GX10, against 34 to 42 s alone on the laptop and 239 to 266 s on the GX10.' },
  { label: 'Scope', value: 'Short prompts only. Long prompts are next.' },
];
