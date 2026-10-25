import { useCallback, useState } from 'react';
import { api, type ApiError } from '../api/client';
import type { HcEvent, MeasuredProfile, ModelView, PlanRequest, TuneResult } from '../api/types';
import { useAction } from '../hooks/useAction';
import { usePoll } from '../hooks/usePoll';
import { ENGINE_LABELS, OBJECTIVE_LABELS, TIER_LABELS, artifactEngines } from '../lib/engines';
import { formatDateTime, formatMs, formatRate } from '../lib/format';
import {
  CANDIDATE_STATUS_TEXT,
  TUNE_STATUS_TEXT,
  currentTune,
  formatScore,
  formatSeconds,
  isTuneEvent,
  profileReuse,
  profileSummary,
  tuneRequestFrom,
  winnerIndex,
} from '../lib/tuning';
import { Card, ErrorNote, StatusText } from './common';

const POLL_MS = 2000;
const PROFILE_POLL_MS = 10_000;
const TUNER_MISSING = 'tuner_unavailable';
const UNREACHABLE = 'gateway_unreachable'; // reported once in the page banner

/** Why "Start tuning" is disabled, or null when it may run. */
export function tuneBlockedReason(
  sel: PlanRequest | null,
  model: ModelView | null,
  running: TuneResult | null,
  activeRunState: string | null,
): string | null {
  if (!sel || !model) return 'Pick a downloaded model in "Choose what to run" first.';
  if (running) return 'A tune is already running. Wait for it or cancel it.';
  if (activeRunState) return `A run is ${activeRunState}. Stop it first: tuning loads each candidate on the GPU in turn.`;
  if (!artifactEngines(model.artifact).includes('llama_rpc') || (sel.engine && sel.engine !== 'llama_rpc')) {
    const engines = artifactEngines(model.artifact).map((e) => ENGINE_LABELS[e]).join(' or ');
    return `Tuning measures llama.cpp splits. This choice runs on ${sel.engine ? ENGINE_LABELS[sel.engine] : engines}.`;
  }
  return null;
}

export function CandidateTable({ tune }: { tune: TuneResult }) {
  const best = winnerIndex(tune);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Candidate</th>
            <th>Status</th>
            <th>Load</th>
            <th>First token</th>
            <th>Decode</th>
            <th>Prefill</th>
            <th>Score (lower is better)</th>
          </tr>
        </thead>
        <tbody>
          {tune.candidates.map((c, i) => (
            <tr key={`${i}-${c.label}`} className={i === best ? 'winner' : undefined}>
              <td>
                {c.label}
                {i === best && <span className="tag">fastest</span>}
                <span className="muted small block">
                  {ENGINE_LABELS[c.plan.engine]} · {TIER_LABELS[c.plan.tier]}
                </span>
              </td>
              <td>
                {CANDIDATE_STATUS_TEXT[c.status]}
                {c.error && <span className="small block">{c.error}</span>}
              </td>
              <td>{formatSeconds(c.load_s)}</td>
              <td>{formatMs(c.ttft_ms)}</td>
              <td>{formatRate(c.decode_tokens_per_s)}</td>
              <td>{formatRate(c.prefill_tokens_per_s)}</td>
              <td>{formatScore(c.score)}</td>
            </tr>
          ))}
          {tune.candidates.length === 0 && (
            <tr>
              <td colSpan={7} className="muted">
                No candidates yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

/**
 * What happens to the profile a finished tune saved, or the tune's note on why none was saved. Reuse is claimed
 * only when the planner will replay it.
 */
export function TuneSavedNote({ tune }: { tune: TuneResult }) {
  const reuse = profileReuse(tune);
  const note = tune.note?.trim() || null;
  if (!reuse && !note) return null;
  return (
    <p className="small">
      <StatusText tone={reuse?.reused ? 'ok' : 'warn'}>{reuse ? reuse.heading : 'Not saved as a profile'}</StatusText>
      <span className="muted"> · {[reuse?.text, note].filter(Boolean).join('; ')}</span>
    </p>
  );
}

function ProfileRow({ p }: { p: MeasuredProfile }) {
  const s = profileSummary(p);
  const setup = [
    s.engine && (ENGINE_LABELS as Record<string, string>)[s.engine],
    s.tier && (TIER_LABELS as Record<string, string>)[s.tier],
    s.cpuLayers ? `${s.cpuLayers} CPU layer${s.cpuLayers === 1 ? '' : 's'}` : null,
  ].filter(Boolean);
  return (
    <tr>
      <td className="small">{formatDateTime(p.created_at)}</td>
      <td>{OBJECTIVE_LABELS[p.objective]}</td>
      <td>
        {setup.join(' · ') || 'unavailable'}
        {s.devices.length > 0 && <span className="mono small block">{s.devices.join(', ')}</span>}
        {s.flags.length > 0 && <span className="mono small block">{s.flags.join(' ')}</span>}
      </td>
      <td>{formatMs(p.ttft_ms)}</td>
      <td>{formatRate(p.decode_tokens_per_s)}</td>
      <td>{formatRate(p.prefill_tokens_per_s)}</td>
      <td>{formatScore(p.score)}</td>
    </tr>
  );
}

function ProfilesList({ modelId, tick }: { modelId: string; tick: number }) {
  const fetchProfiles = useCallback(() => api.profiles(modelId), [modelId]);
  const profiles = usePoll(fetchProfiles, PROFILE_POLL_MS, tick);
  return (
    <>
      <h3>Saved profiles for {modelId}</h3>
      {profiles.error?.code !== UNREACHABLE && <ErrorNote error={profiles.error} />}
      {profiles.data && profiles.data.length === 0 && (
        <p className="muted small">None yet. Plans use estimates until a tune saves a measured profile.</p>
      )}
      {profiles.data && profiles.data.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Saved</th>
                <th>For</th>
                <th>Setup</th>
                <th>First token</th>
                <th>Decode</th>
                <th>Prefill</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {profiles.data.map((p) => (
                <ProfileRow key={p.id} p={p} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

interface TuningPanelProps {
  selection: PlanRequest | null;
  models: ModelView[] | undefined;
  /** State of a deployment that is not stopped or failed, else null. */
  activeRunState: string | null;
  events: HcEvent[];
  tick: number;
  onChanged: () => void;
}

export function TuningPanel({ selection, models, activeRunState, events, tick, onChanged }: TuningPanelProps) {
  const tunes = usePoll(api.tunes, POLL_MS, tick);
  const [maxCandidates, setMaxCandidates] = useState(4);
  const [tuneFlags, setTuneFlags] = useState(true);
  const start = useAction(api.startTune, onChanged);
  const cancel = useAction((id: string) => api.cancelTune(id), onChanged);

  const model = models?.find((m) => m.artifact.id === selection?.model_id) ?? null;
  const tune = currentTune(tunes.data, selection?.model_id ?? null);
  const running = tune?.status === 'running' ? tune : null;
  const missing: ApiError | null = tunes.error?.code === TUNER_MISSING ? tunes.error : null;
  const blocked = missing ? 'The measured tuner is not part of this gateway build.' : tuneBlockedReason(selection, model, running, activeRunState);
  const latest = running ? events.find(isTuneEvent) : undefined;
  const friends = selection?.strategy === 'local_only' ? 0 : selection?.node_ids?.length ?? 0;

  const onStart = (): void => {
    if (selection && !blocked) void start.run(tuneRequestFrom(selection, { maxCandidates, tuneFlags }));
  };

  return (
    <Card
      title="Tune"
      subtitle="Loads a few candidate setups one after another, measures each and saves the fastest as a profile."
      actions={
        running && (
          <button type="button" disabled={cancel.busy} onClick={() => void cancel.run(running.id)}>
            {cancel.busy ? 'Cancelling…' : 'Cancel tune'}
          </button>
        )
      }
    >
      {selection && (
        <p className="small">
          Tunes <strong>{model?.artifact.display_name ?? selection.model_id}</strong> with {friends} friend
          {friends === 1 ? '' : 's'} · context {selection.ctx_size} tokens · {OBJECTIVE_LABELS[selection.objective ?? 'interactive']}
          <span className="muted"> (from &quot;Choose what to run&quot;)</span>
        </p>
      )}
      <div className="form-row">
        <label>
          Candidates{' '}
          <input
            type="number"
            min={1}
            max={8}
            value={maxCandidates}
            onChange={(e) => setMaxCandidates(Math.min(8, Math.max(1, Math.round(Number(e.target.value) || 1))))}
          />
        </label>
        <label className="choice">
          <input type="checkbox" checked={tuneFlags} onChange={(e) => setTuneFlags(e.target.checked)} />
          <span>Also try flash attention and batch size</span>
        </label>
        <button type="button" className="primary" disabled={blocked !== null || start.busy} onClick={onStart}>
          {start.busy ? 'Starting…' : 'Start tuning'}
        </button>
      </div>
      {blocked && <p className="small">{blocked}</p>}
      <p className="muted small">The GPU is busy while a tune runs. Each candidate loads the model again, so this takes minutes.</p>
      <ErrorNote error={start.error ?? cancel.error} onDismiss={() => [start, cancel].forEach((a) => a.clearError())} />
      {!missing && tunes.error?.code !== UNREACHABLE && <ErrorNote error={tunes.error} />}

      {tune && (
        <>
          <h3>
            {TUNE_STATUS_TEXT[tune.status]} · {tune.request.model_id}
            <span className="muted small"> · started {formatDateTime(tune.created_at)}</span>
          </h3>
          {latest && (
            <p className="small">
              <span className="muted">Now:</span> {latest.message}
            </p>
          )}
          <TuneSavedNote tune={tune} />
          {tune.error && <p className="small">Error: {tune.error}</p>}
          <CandidateTable tune={tune} />
        </>
      )}

      {selection && <ProfilesList key={selection.model_id} modelId={selection.model_id} tick={tick} />}
    </Card>
  );
}
