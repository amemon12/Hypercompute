import { everyFirstToken, laptopFirstToken, laptopManyChats, type BarRow, type ChartSpec } from './benchmark-charts';

// A single before/after format for the three questions in the pitch.
// Sources: docs/pitch/NUMBERS.md at b685ae6, sections 1, 2 and 3 (L7 and G6),
// plus the team-reported website values in docs/RESULTS.md.
// Keep the archived comparisons in benchmark-charts.ts unchanged.
export const generationComparison: ChartSpec<BarRow> = {
  title: 'Generation speed',
  unit: 'Tokens per second · higher is better',
  valueUnit: 'tok/s',
  max: 24,
  ticks: [0, 8, 16, 24],
  series: ['Alone', 'Shared'],
  rows: [
    { label: 'Llama 3.3 70B', note: '5.44× faster', values: [0.165, 0.898], display: '5.44× [5.42, 5.47]' },
    { label: 'gpt-oss-120b', note: '2.28× faster', values: [3.829, 8.740], display: '2.28× [2.00, 2.54]' },
    { label: 'Qwen3-Next 80B', note: '2.13× faster', values: [10.005, 21.373], display: '2.13×' },
    { label: 'DeepSeek-V3.1 671B', note: '17% slower', values: [1.805, 1.497], display: '0.83× [0.76, 0.91]' },
    { label: 'Kimi-K2.6 1T', note: '1.16× faster', values: [1.540, 1.799], display: '1.16×' },
  ],
};

export const firstResponseComparison: ChartSpec<BarRow> = {
  ...laptopFirstToken,
  title: 'Time to first response',
  rows: [
    ...laptopFirstToken.rows.filter(row => generationComparison.rows.some(model => model.label === row.label)),
    ...everyFirstToken.rows.filter(row => row.note === '4 peers · separate session'),
  ].map(row => {
    // Qwen uses the team-reported first-response pair. Kimi uses its recorded pair.
    const values = row.label === 'Qwen3-Next 80B'
      ? [row.values[1], row.values[0]]
      : row.values;
    const [alone, shared] = values;
    const change = Math.round(Math.abs(shared / alone - 1) * 100);
    return {
      ...row,
      values,
      display: row.label === 'Qwen3-Next 80B' ? '4.31 s → 1.56 s' : row.display,
      note: `${change}% ${shared > alone ? 'more' : 'less'} waiting`,
    };
  }),
};

export const concurrentComparison: ChartSpec<BarRow> = {
  ...laptopManyChats,
  title: 'Total output across chats',
  unit: 'Tokens per second, combined · higher is better',
};
