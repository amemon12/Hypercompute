import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { assignment, candidate, enginesStatus, extraEngine, modelView, artifact, node, plan, tuneResult } from '../testing/fixtures';
import { AdvancedRunOptions } from './AdvancedRunOptions';
import { EnginesPanel, engineRows } from './EnginesPanel';
import { CapacityBanner, PlanView } from './PlanView';
import { CandidateTable, TuneSavedNote, tuneBlockedReason } from './TuningPanel';
import { DEFAULT_RUN_OPTIONS } from '../lib/runOptions';

const noop = (): void => undefined;

function renderPlan(p = plan()): string {
  return renderToStaticMarkup(
    <PlanView
      plan={p}
      nodes={undefined}
      canRun
      runBlockedReason={null}
      running={false}
      runError={null}
      onRun={noop}
      onDismissError={noop}
    />,
  );
}

describe('PlanView', () => {
  it('shows engine and tier badges and the estimate label', () => {
    const html = renderPlan();
    expect(html).toContain('llama.cpp');
    expect(html).toContain('All on GPUs');
    expect(html).toContain('Estimate');
    expect(html).toContain('Fastest that fits');
    expect(html).not.toContain('capacity-banner');
  });

  it('shows CPU layers, -ngl and where the overflow went', () => {
    const html = renderPlan(plan({ tier: 'gpu_ram', cpu_layers: 6, n_gpu_units: 23 }));
    expect(html).toContain('6 layers on this PC&#x27;s CPU');
    expect(html).toContain('-ngl 23');
    expect(html).toContain('(from RAM)');
  });

  it('shows a llama.cpp device that holds only the output head as 0 + output layer', () => {
    const head = assignment({ device: 'CUDA1', layers: [], holds_output: true, weight_bytes: 638582784 });
    const html = renderPlan(plan({ assignments: [assignment({ holds_output: false }), head] }));
    expect(html).toContain('<td>0<span class="muted small block">+ output layer</span></td>');
    expect(html).not.toContain('decided by');
  });

  it('labels a plan that reuses a measured profile', () => {
    const html = renderPlan(plan({ heuristic: false, profile_id: 'abcdef0123456789' }));
    expect(html).toContain('Measured profile');
    expect(html).toContain('abcdef01');
    expect(html).not.toContain('not yet measured');
  });

  it('puts a capacity-mode banner on disk-backed plans', () => {
    const html = renderPlan(plan({ engine: 'airllm', tier: 'disk', capacity_mode: true, assignments: [] }));
    expect(html).toContain('capacity-banner');
    expect(html).toContain('Disk (capacity mode)');
    expect(html).not.toContain('token embeddings');
    expect(html).toContain('every layer is read from disk for every token');
    const paged = renderToStaticMarkup(<CapacityBanner plan={plan({ tier: 'gpu_disk', capacity_mode: true, cpu_layers: 6 })} />);
    expect(paged).toContain('seconds or more per token');
    expect(paged).toContain('paged from the GGUF file on disk');
    expect(paged).not.toContain('every layer');
  });

  it('says EXO decides the per-Mac layer split instead of showing 0 layers', () => {
    const mac = assignment({ device: 'MLX', layers: [], holds_output: false, weight_bytes: 364904448, total_bytes_est: 400000000 });
    const exo = plan({ engine: 'exo', device_list: [], tensor_split: [], assignments: [mac, { ...mac, node_id: 'mac2' }] });
    const html = renderPlan(exo);
    expect(html.match(/decided by EXO/g)).toHaveLength(2);
    expect(html).not.toContain('<td>0</td>');
  });
});

describe('AdvancedRunOptions', () => {
  it('disables engines with the reason and keeps automatic selectable', () => {
    const html = renderToStaticMarkup(
      <AdvancedRunOptions
        options={DEFAULT_RUN_OPTIONS}
        engine="auto"
        engines={enginesStatus()}
        artifact={artifact({ format: 'hf' })}
        onChange={noop}
      />,
    );
    expect(html).toContain('Automatic');
    expect(html).toContain('Unavailable: not for this model (safetensors files run on AirLLM or vLLM)');
    expect(html).toContain('Unavailable: not supported on this computer (vLLM needs Linux or WSL2)');
    expect(html).toContain('Single user');
    expect(html).toContain('Allow capacity mode');
  });
});

describe('EnginesPanel', () => {
  const nodes = [
    node({ id: 'host', display_name: 'G14', is_local: true }),
    node({ id: 'mac', display_name: 'Mac mini', extras: [extraEngine({ engine: 'exo', status: 'running', installed: true, endpoint: '10.0.0.7:52415' })] }),
    node({ id: 'gone', display_name: 'Left friend', state: 'left' }),
  ];

  it('lists the host first, then active friends, preferring the gateway node report', () => {
    const status = enginesStatus({ nodes: { mac: [extraEngine({ engine: 'exo', status: 'installed', installed: true })] } });
    const rows = engineRows(status, nodes);
    expect(rows.map((r) => r.machine)).toEqual(['G14', 'Mac mini']);
    expect(rows[1]?.extras[0]?.status).toBe('installed');
    expect(engineRows(enginesStatus(), nodes)[1]?.extras[0]?.status).toBe('running');
  });

  it('renders status, detail and the install command', () => {
    const html = renderToStaticMarkup(<EnginesPanel engines={enginesStatus()} loadError={null} nodes={nodes} />);
    expect(html).toContain('b11770 · cuda, smoke test passed');
    expect(html).toContain('uv run hypercompute engine install --engine airllm');
    expect(html).toContain('EXO runs on macOS only');
    expect(html).toContain('10.0.0.7:52415');
    expect(html).not.toContain('Left friend');
  });
});

describe('Tuning', () => {
  it('highlights the measured winner and shows every metric column', () => {
    const t = tuneResult({
      status: 'done',
      best_index: 1,
      candidates: [
        candidate({ label: 'peer split', status: 'done', load_s: 21.5, ttft_ms: 300, decode_tokens_per_s: 20, prefill_tokens_per_s: 500, score: 3.5 }),
        candidate({ label: 'RAM tier', status: 'done', load_s: 12, ttft_ms: 180, decode_tokens_per_s: 31.25, prefill_tokens_per_s: 800, score: 2.23 }),
        candidate({ label: 'fa off', status: 'failed', error: 'load timed out' }),
      ],
    });
    const html = renderToStaticMarkup(<CandidateTable tune={t} />);
    expect(html).toContain('class="winner"');
    expect(html).toContain('fastest');
    expect(html).toContain('12.0 s');
    expect(html).toContain('180 ms');
    expect(html).toContain('31.3 tok/s');
    expect(html).toContain('800.0 tok/s');
    expect(html).toContain('2.23 s');
    expect(html).toContain('load timed out');
    expect(html.match(/class="winner"/g)).toHaveLength(1);
  });

  it('explains why tuning cannot start', () => {
    const sel = { model_id: 'qwen2.5-7b-q4', node_ids: [], strategy: 'auto' as const, ctx_size: 4096 };
    const gguf = modelView();
    expect(tuneBlockedReason(null, null, null, null)).toContain('Pick a downloaded model');
    expect(tuneBlockedReason(sel, gguf, tuneResult(), null)).toContain('already running');
    expect(tuneBlockedReason(sel, gguf, null, 'ready')).toContain('A run is ready');
    expect(tuneBlockedReason(sel, modelView({ artifact: artifact({ format: 'mlx' }) }), null, null)).toContain('runs on EXO');
    expect(tuneBlockedReason({ ...sel, engine: 'vllm' }, gguf, null, null)).toContain('runs on vLLM');
    expect(tuneBlockedReason(sel, gguf, null, null)).toBeNull();
  });

  it('claims a saved profile is reused only when the tune measured llama.cpp', () => {
    const done = tuneResult({ status: 'done', profile_id: 'p1', best_index: 0, candidates: [candidate({ status: 'done' })] });
    const llama = renderToStaticMarkup(<TuneSavedNote tune={done} />);
    expect(llama).toContain('Saved as a measured profile');
    expect(llama).toContain('later plans for this model and these machines reuse it');
    const vllmWinner = candidate({ status: 'done', plan: plan({ engine: 'vllm', model_id: 'qwen2.5-7b-awq' }) });
    const vllm = renderToStaticMarkup(<TuneSavedNote tune={{ ...done, candidates: [vllmWinner] }} />);
    expect(vllm).not.toContain('reuse it');
    expect(vllm).toContain('measured with vLLM');
    expect(renderToStaticMarkup(<TuneSavedNote tune={{ ...done, profile_id: null }} />)).toBe('');
  });

  it('keeps the heading and detail consistent when the winner is missing', () => {
    const done = tuneResult({ status: 'done', profile_id: 'p1', best_index: null, candidates: [candidate({ status: 'failed' })] });
    const html = renderToStaticMarkup(<TuneSavedNote tune={done} />);
    expect(html).toContain('Saved, reuse unconfirmed');
    expect(html).toContain('the measured winner is unavailable');
    expect(html).not.toContain('not used by later plans');
  });

  it("shows the tune's note on why the winner was not saved as a profile", () => {
    const note = 'the winner ran on vLLM; plans replay only llama.cpp profiles';
    const unsaved = tuneResult({ status: 'done', profile_id: null, best_index: 0, candidates: [candidate({ status: 'done' })], note });
    const html = renderToStaticMarkup(<TuneSavedNote tune={unsaved} />);
    expect(html).toContain('Not saved as a profile');
    expect(html).toContain('the winner ran on vLLM; plans replay only llama.cpp profiles');
    const saved = renderToStaticMarkup(<TuneSavedNote tune={{ ...unsaved, profile_id: 'p1', note: 'kept the flags of run 2' }} />);
    expect(saved).toContain('Saved as a measured profile');
    expect(saved).toContain('kept the flags of run 2');
    expect(renderToStaticMarkup(<TuneSavedNote tune={{ ...unsaved, note: '  ' }} />)).toBe('');
  });
});
