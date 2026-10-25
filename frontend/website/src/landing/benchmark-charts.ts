// Chart data for the benchmarks tabs. Plotted values are the tool values printed in
// docs/pitch/NUMBERS.md and bench/RESULTS.md. Labels use the sheet's public roundings.
// Every scale starts at zero. Ratio ranges are 95% intervals; single means carry none.

import type { Tone } from './benchmark-sheet';

export interface RatioRow {
  readonly label: string;
  readonly note: string;
  readonly value: number;
  readonly lo: number;
  readonly hi: number;
  readonly display: string;
  readonly tone: Tone;
}

export interface BarRow {
  readonly label: string;
  readonly note: string;
  readonly values: readonly number[];
  readonly display: string;
}

export interface RangeRow {
  readonly label: string;
  readonly lo: number;
  readonly hi: number;
  readonly display: string;
  readonly pooled: boolean;
}

export interface ChartSpec<R> {
  readonly title: string;
  readonly unit: string;
  readonly valueUnit?: 's' | 'tok/s';
  readonly max: number;
  readonly ticks: readonly number[];
  readonly rows: readonly R[];
  readonly reference?: { readonly at: number; readonly label: string };
  readonly series?: readonly string[];
}

const parity = { at: 1, label: 'same speed' };

export const laptopDecodeRatio: ChartSpec<RatioRow> = {
  title: 'Generation speed',
  unit: 'Shared ÷ alone · higher is better',
  max: 6,
  ticks: [0, 1, 2, 3, 4, 5, 6],
  reference: parity,
  rows: [
    { label: 'Llama 3.3 70B', note: 'dense', value: 5.444, lo: 5.423, hi: 5.464, display: '5.44× [5.42, 5.47]', tone: 'gain' },
    { label: 'Qwen2.5-VL 72B', note: 'dense, text only', value: 2.592, lo: 2.571, hi: 2.613, display: '2.59× [2.57, 2.62]', tone: 'gain' },
    { label: 'gpt-oss-120b', note: 'MoE', value: 2.283, lo: 2.0, hi: 2.538, display: '2.28× [2.00, 2.54]', tone: 'gain' },
    { label: 'Qwen3-Next 80B', note: 'MoE, fits the laptop', value: 0.468, lo: 0.362, hi: 0.611, display: '0.46× [0.36, 0.62]', tone: 'loss' },
  ],
};

export const laptopFirstToken: ChartSpec<BarRow> = {
  title: 'Time to first token',
  unit: 'Seconds · lower is better',
  valueUnit: 's',
  max: 40,
  ticks: [0, 10, 20, 30, 40],
  series: ['Alone', 'Shared'],
  rows: [
    { label: 'Llama 3.3 70B', note: '64-token prompt', values: [10.02, 5.01], display: '10.02 s → 5.01 s' },
    { label: 'Qwen2.5-VL 72B', note: '64-token prompt', values: [13.7, 9.1], display: '13.70 s → 9.10 s' },
    { label: 'gpt-oss-120b', note: '128-token prompt', values: [34.23, 4.83], display: '34.23 s → 4.83 s' },
    { label: 'Qwen3-Next 80B', note: '128-token prompt', values: [1.56, 4.31], display: '1.56 s → 4.31 s' },
  ],
};

export const gx10Decode: ChartSpec<BarRow> = {
  title: 'Generation speed',
  unit: 'Tokens per second · one chat',
  max: 60,
  ticks: [0, 10, 20, 30, 40, 50, 60],
  series: ['GX10 alone'],
  rows: [
    { label: 'gpt-oss-120b', note: '128 prompt / 64 output tokens', values: [54.621], display: '54.621 tok/s' },
    { label: 'Gemma 3 27B IT', note: '128 prompt / 64 output tokens', values: [12.341], display: '12.341 tok/s' },
    { label: 'QwQ 32B', note: '128 prompt / 64 output tokens', values: [10.643], display: '10.643 tok/s' },
    { label: 'Llama 4 Maverick 400B', note: '128 prompt / 64 output tokens', values: [9.785], display: '9.785 tok/s' },
    { label: 'DeepSeek-V3.1 671B', note: '64 prompt / 32 output tokens', values: [1.792], display: '1.792 tok/s' },
    { label: 'Kimi-K2.6 1T', note: '64 prompt / 32 output tokens', values: [1.779], display: '1.779 tok/s' },
  ],
};

export const everyDecodeRatio: ChartSpec<RatioRow> = {
  title: 'Generation speed',
  unit: 'Shared ÷ GX10 alone · higher is better',
  max: 1.25,
  ticks: [0, 0.25, 0.5, 0.75, 1, 1.25],
  reference: parity,
  rows: [
    { label: 'Kimi-K2.6 1T', note: '3 peers', value: 0.997, lo: 0.964, hi: 1.03, display: '0.99× [0.96, 1.03]', tone: 'parity' },
    { label: 'DeepSeek-V3.1 671B', note: '3 peers · alternating runs', value: 1.006, lo: 0.926, hi: 1.081, display: '1.00× [0.92, 1.09]', tone: 'parity' },
    { label: 'Llama 4 Maverick 400B', note: '3 peers', value: 0.437, lo: 0.4, hi: 0.477, display: '0.43× [0.40, 0.48]', tone: 'loss' },
    { label: 'DeepSeek-V3.1 671B', note: '4 peers · separate session', value: 0.83, lo: 0.769, hi: 0.903, display: '0.83× [0.76, 0.91]', tone: 'loss' },
    { label: 'Kimi-K2.6 1T', note: '4 peers · separate session', value: 0.856, lo: 0.818, hi: 0.886, display: '0.85× [0.81, 0.89]', tone: 'loss' },
  ],
};

export const everyFirstToken: ChartSpec<BarRow> = {
  title: 'Time to first token',
  unit: 'Seconds · lower is better',
  valueUnit: 's',
  max: 30,
  ticks: [0, 10, 20, 30],
  series: ['Alone', 'Shared'],
  rows: [
    { label: 'Kimi-K2.6 1T', note: '3 peers', values: [27.01, 20.22], display: '25% sooner [22%, 28%]' },
    { label: 'DeepSeek-V3.1 671B', note: '3 peers · alternating runs', values: [25.02, 15.77], display: '1.58× sooner' },
    { label: 'Llama 4 Maverick 400B', note: '3 peers', values: [2.41, 3.94], display: '1.64× as long [1.26, 2.86]' },
    { label: 'DeepSeek-V3.1 671B', note: '4 peers · separate session', values: [24.67, 10.36], display: '2.38× sooner [2.15, 2.69]' },
    { label: 'Kimi-K2.6 1T', note: '4 peers · separate session', values: [28.25, 19.08], display: '1.48× sooner [1.37, 1.60]' },
  ],
};

export const laptopManyChats: ChartSpec<BarRow> = {
  title: 'gpt-oss-120b on the laptops',
  unit: 'Tokens per second · all chats combined',
  max: 20,
  ticks: [0, 5, 10, 15, 20],
  series: ['Alone', 'Shared'],
  rows: [
    { label: '1 chat', note: '', values: [3.829, 8.74], display: '2.28× [2.00, 2.54]' },
    { label: '4 chats', note: '', values: [8.095, 11.854], display: '1.46× [1.29, 1.68]' },
    { label: '8 chats', note: '', values: [12.187, 16.445], display: '1.34× [1.24, 1.48]' },
  ],
};

export const gx10EightVsOne: ChartSpec<RatioRow> = {
  title: 'Eight chats on one GX10',
  unit: 'Total output · 8 chats ÷ 1 chat',
  max: 8,
  ticks: [0, 2, 4, 6, 8],
  reference: { at: 1, label: 'same total output' },
  rows: [
    { label: 'gpt-oss-120b', note: '54.621 → 149.156 tok/s', value: 2.731, lo: 2.583, hi: 2.901, display: '2.73× [2.58, 2.91]', tone: 'gain' },
    { label: 'QwQ 32B', note: '10.643 → 75.169 tok/s', value: 7.063, lo: 7.043, hi: 7.077, display: '7.06× [7.04, 7.08]', tone: 'gain' },
    { label: 'Gemma 3 27B IT', note: '12.341 → 85.044 tok/s', value: 6.891, lo: 6.883, hi: 6.903, display: '6.89× [6.88, 6.91]', tone: 'gain' },
    { label: 'Llama 4 Maverick 400B', note: '9.785 → 43.362 tok/s', value: 4.431, lo: 4.087, hi: 4.809, display: '4.43× [4.08, 4.81]', tone: 'gain' },
    { label: 'DeepSeek-V3.1 671B', note: '1.792 → 5.547 tok/s', value: 3.096, lo: 2.525, hi: 3.686, display: '3.09× [2.52, 3.69]', tone: 'gain' },
    { label: 'Kimi-K2.6 1T', note: '1.779 → 4.640 tok/s', value: 2.608, lo: 2.372, hi: 2.844, display: '2.60× [2.37, 2.85]', tone: 'gain' },
  ],
};

export const noHelpRatio: ChartSpec<RatioRow> = {
  title: 'Generation speed',
  unit: 'Compared with baseline · below 1× is slower',
  max: 1.5,
  ticks: [0, 0.5, 1, 1.5],
  reference: parity,
  rows: [
    { label: 'Llama 4 Maverick 400B', note: 'GX10 + 3 peers, 8 chats', value: 0.348, lo: 0.329, hi: 0.37, display: '0.34× [0.32, 0.37]', tone: 'loss' },
    { label: 'Llama 4 Maverick 400B', note: 'GX10 + 3 peers, 1 chat', value: 0.437, lo: 0.4, hi: 0.477, display: '0.43× [0.40, 0.48]', tone: 'loss' },
    { label: 'Qwen3-Next 80B', note: 'laptop + 2 Airs', value: 0.468, lo: 0.362, hi: 0.611, display: '0.46× [0.36, 0.62]', tone: 'loss' },
    { label: 'DeepSeek-V3.1 671B', note: 'GX10 + 4 peers', value: 0.83, lo: 0.769, hi: 0.903, display: '0.83× [0.76, 0.91]', tone: 'loss' },
    { label: 'Kimi-K2.6 1T', note: 'GX10 + 4 peers', value: 0.856, lo: 0.818, hi: 0.886, display: '0.85× [0.81, 0.89]', tone: 'loss' },
    { label: 'gpt-oss-120b', note: '3 Airs vs 1 Air, building Wi-Fi', value: 1.016, lo: 0.86, hi: 1.199, display: '1.01× [0.86, 1.20]', tone: 'parity' },
    { label: 'Qwen3-235B-A22B', note: 'laptop + 3 Airs, building Wi-Fi', value: 1.079, lo: 0.976, hi: 1.184, display: '1.07× [0.97, 1.19]', tone: 'parity' },
    { label: 'Kimi-K2.6 1T', note: 'GX10 + 3 peers, 8 chats', value: 1.075, lo: 0.975, hi: 1.194, display: '1.07× [0.97, 1.20]', tone: 'parity' },
  ],
};

export const coldStart: ChartSpec<RangeRow> = {
  title: 'Starting from cold',
  unit: 'Seconds · peer seeding and model loading',
  max: 2000,
  ticks: [0, 500, 1000, 1500, 2000],
  rows: [
    { label: 'Laptop alone', lo: 34, hi: 42, display: '34 to 42 s', pooled: false },
    { label: 'Laptop + Airs', lo: 873, hi: 948, display: '873 to 948 s', pooled: true },
    { label: 'GX10 alone', lo: 239, hi: 266, display: '239 to 266 s', pooled: false },
    { label: 'GX10 + peers', lo: 1043, hi: 1919, display: '1043 to 1919 s', pooled: true },
  ],
};
