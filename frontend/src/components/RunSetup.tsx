import { useEffect, useRef, useState } from 'react';
import { api, type ApiError } from '../api/client';
import type { EnginesStatus, ModelView, NodeView, Plan, PlanRequest, PlanStrategy } from '../api/types';
import { useAction } from '../hooks/useAction';
import { ENGINE_LABELS, FORMAT_LABELS, artifactEngines, engineManagedText, isModelUsable, modelSizeBytes } from '../lib/engines';
import { STRATEGY_HINTS, STRATEGY_LABELS, formatBytes } from '../lib/format';
import { DEFAULT_RUN_OPTIONS, effectiveEngine, planRequestFor, type RunOptions } from '../lib/runOptions';
import { AdvancedRunOptions } from './AdvancedRunOptions';
import { Card, ErrorNote } from './common';

const STRATEGIES: PlanStrategy[] = ['auto', 'all_selected', 'fastest_fit', 'local_only'];
const MIN_CTX = 256;
const MAX_CTX = 262144;

interface ModelRowProps {
  model: ModelView;
  selected: boolean;
  onSelect: () => void;
  onChanged: () => void;
}

function ModelRow({ model, selected, onSelect, onChanged }: ModelRowProps) {
  const download = useAction((id: string) => api.downloadModel(id), onChanged);
  const a = model.artifact;
  const ready = model.status === 'downloaded';
  const pct = model.total_bytes > 0 ? Math.floor((model.downloaded_bytes / model.total_bytes) * 100) : null;
  const engines = artifactEngines(a).map((e) => ENGINE_LABELS[e]).join(', ');
  return (
    <li className={`model-row ${selected ? 'selected' : ''}`}>
      <label className="model-pick">
        <input type="radio" name="model" checked={selected} disabled={!isModelUsable(model)} onChange={onSelect} />
        <span>
          <strong>{a.display_name}</strong>
          <span className="muted small">
            {' '}
            {a.params_b} B params · {a.quantization} · {FORMAT_LABELS[a.format]} for {engines} · {formatBytes(modelSizeBytes(model))}
            {a.role === 'plumbing' ? ' · small test model' : a.role === 'capacity_candidate' ? ' · large model' : ''}
          </span>
        </span>
      </label>
      <div className="model-status small">
        {ready && <span>downloaded{model.verified_sha256 ? ', checksum verified' : ', checksum not verified'}</span>}
        {model.status === 'engine_managed' && <span>{engineManagedText(a)}</span>}
        {model.status === 'downloading' && (
          <span className="progress-line">
            <progress value={model.downloaded_bytes} max={model.total_bytes || undefined} />
            {formatBytes(model.downloaded_bytes)} of {formatBytes(model.total_bytes > 0 ? model.total_bytes : null)}
            {pct !== null ? ` (${pct}%)` : ''}
          </span>
        )}
        {(model.status === 'not_downloaded' || model.status === 'error' || model.status === 'corrupt') && (
          <button type="button" disabled={download.busy} onClick={() => void download.run(a.id)}>
            {download.busy ? 'Starting…' : model.status === 'not_downloaded' ? 'Download' : 'Download again'}
          </button>
        )}
        {model.status === 'corrupt' && <span> checksum mismatch</span>}
        {model.error && <span> {model.error}</span>}
      </div>
      <ErrorNote error={download.error} onDismiss={download.clearError} />
    </li>
  );
}

interface RunSetupProps {
  models: ModelView[] | undefined;
  modelsError: ApiError | null;
  nodes: NodeView[] | undefined;
  engines: EnginesStatus | undefined;
  onPlan: (plan: Plan) => void;
  /** The request "Check plan" would send right now (null while no model is usable); the tuner reuses it. */
  onSelection: (req: PlanRequest | null) => void;
  onChanged: () => void;
}

export function RunSetup({ models, modelsError, nodes, engines, onPlan, onSelection, onChanged }: RunSetupProps) {
  const [modelId, setModelId] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<PlanStrategy>('auto');
  // Friends are included by default; we remember who was unticked so new friends show up ticked.
  const [unticked, setUnticked] = useState<ReadonlySet<string>>(new Set());
  const [ctx, setCtx] = useState<number | null>(null);
  const [options, setOptions] = useState<RunOptions>(DEFAULT_RUN_OPTIONS);
  const check = useAction(api.createPlan, onPlan);

  const usable = (models ?? []).filter(isModelUsable);
  // Default to the first usable model until the owner picks one.
  const selectedModel = usable.find((m) => m.artifact.id === modelId) ?? usable[0] ?? null;
  const artifact = selectedModel?.artifact ?? null;
  const engine = effectiveEngine(options.engine, engines, artifact);

  const friends = (nodes ?? []).filter((n) => !n.is_local && n.state !== 'left' && n.state !== 'removed');
  const eligibleIds = friends.filter((n) => n.state === 'eligible').map((n) => n.id);
  const chosenIds = eligibleIds.filter((id) => !unticked.has(id));
  const ctxSize = ctx ?? artifact?.default_ctx ?? 4096;
  const ctxValid = Number.isInteger(ctxSize) && ctxSize >= MIN_CTX && ctxSize <= MAX_CTX;
  const request =
    artifact && ctxValid ? planRequestFor(artifact.id, strategy, chosenIds, ctxSize, { ...options, engine }) : null;

  // Report the selection only when it changes, so the parent re-renders once per edit.
  const requestKey = JSON.stringify(request);
  const onSelectionRef = useRef(onSelection);
  onSelectionRef.current = onSelection;
  useEffect(() => {
    onSelectionRef.current(JSON.parse(requestKey) as PlanRequest | null);
  }, [requestKey]);

  const toggle = (id: string, on: boolean): void => {
    const next = new Set(unticked);
    if (on) next.delete(id);
    else next.add(id);
    setUnticked(next);
  };

  return (
    <Card title="Choose what to run" subtitle="Only downloaded models, or ones EXO fetches on each Mac, can be selected.">
      <ErrorNote error={modelsError} />
      {models && models.length === 0 && <p className="muted">The gateway lists no models.</p>}
      <ul className="model-list">
        {(models ?? []).map((m) => (
          <ModelRow
            key={m.artifact.id}
            model={m}
            selected={m.artifact.id === artifact?.id}
            onSelect={() => setModelId(m.artifact.id)}
            onChanged={onChanged}
          />
        ))}
      </ul>

      <fieldset className="choice-group">
        <legend>How to split the work</legend>
        {STRATEGIES.map((s) => (
          <label key={s} className="choice">
            <input type="radio" name="strategy" checked={strategy === s} onChange={() => setStrategy(s)} />
            <span>
              {STRATEGY_LABELS[s]}
              <span className="muted small block">{STRATEGY_HINTS[s]}</span>
            </span>
          </label>
        ))}
      </fieldset>

      <fieldset className="choice-group" disabled={strategy === 'local_only'}>
        <legend>Friends to include</legend>
        {friends.length === 0 && <p className="muted small">No friends have joined yet.</p>}
        {friends.map((n) => {
          const eligible = n.state === 'eligible';
          return (
            <label key={n.id} className="choice">
              <input
                type="checkbox"
                disabled={!eligible}
                checked={eligible && !unticked.has(n.id)}
                onChange={(e) => toggle(n.id, e.target.checked)}
              />
              <span>
                {n.display_name}
                {!eligible && <span className="muted small"> ({n.state}{n.state_reason ? `: ${n.state_reason}` : ''})</span>}
              </span>
            </label>
          );
        })}
      </fieldset>

      <AdvancedRunOptions options={options} engine={engine} engines={engines} artifact={artifact} onChange={setOptions} />

      <div className="form-row">
        <label>
          Context length (tokens){' '}
          <input
            type="number"
            min={MIN_CTX}
            max={MAX_CTX}
            step={256}
            value={ctxSize}
            onChange={(e) => setCtx(e.target.value === '' ? null : Number(e.target.value))}
          />
        </label>
        {!ctxValid && <span className="small">Use a whole number from {MIN_CTX} to {MAX_CTX}.</span>}
        <button
          type="button"
          className="primary"
          disabled={!request || check.busy}
          onClick={() => request && void check.run(request)}
        >
          {check.busy ? 'Checking…' : 'Check plan'}
        </button>
      </div>
      <ErrorNote error={check.error} onDismiss={check.clearError} />
    </Card>
  );
}
