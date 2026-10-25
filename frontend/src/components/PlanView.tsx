import type { ApiError } from '../api/client';
import type { NodeView, Plan } from '../api/types';
import { ENGINE_LABELS, TIER_LABELS, capacityWarning, cpuLayersText, layersText, overflowText, planSourceLabel } from '../lib/engines';
import { FIT_LABELS, STRATEGY_LABELS, formatBytes } from '../lib/format';
import { Card, ErrorNote, StatusText } from './common';

/** Shown on any disk-backed plan or run, so nobody waits on a slow reply without knowing why. */
export function CapacityBanner({ plan }: { plan: Plan }) {
  const text = capacityWarning(plan);
  if (!text) return null;
  return (
    <div className="capacity-banner" role="note">
      <strong>Slow: disk-backed run.</strong> {text}
    </div>
  );
}

/** Engine and tier tags, plus whether the numbers come from a measured profile or an estimate. */
export function PlanBadges({ plan }: { plan: Plan }) {
  const source = planSourceLabel(plan);
  return (
    <p className="badges">
      <span className="tag">{ENGINE_LABELS[plan.engine]}</span>
      <span className="tag">{TIER_LABELS[plan.tier]}</span>
      <span className={`tag ${plan.profile_id ? '' : 'tag-muted'}`}>{source}</span>
    </p>
  );
}

function SourceNote({ plan }: { plan: Plan }) {
  if (plan.profile_id) {
    return <span className="muted small"> · reuses measured profile {plan.profile_id.slice(0, 8)} for this model and these machines</span>;
  }
  if (!plan.heuristic) return null;
  return <span className="muted small"> · estimate from model file sizes, not yet measured on this setup</span>;
}

interface PlanViewProps {
  plan: Plan;
  nodes: NodeView[] | undefined;
  canRun: boolean;
  runBlockedReason: string | null;
  running: boolean;
  runError: ApiError | null;
  onRun: () => void;
  onDismissError: () => void;
}

export function PlanView({ plan, nodes, canRun, runBlockedReason, running, runError, onRun, onDismissError }: PlanViewProps) {
  const byId = new Map((nodes ?? []).map((n) => [n.id, n]));
  const nameOf = (id: string): string => byId.get(id)?.display_name ?? id.slice(0, 8);
  const machineIds = [...new Set(plan.assignments.map((a) => a.node_id))];
  const contributed = machineIds.reduce<number | null>((acc, id) => {
    const node = byId.get(id);
    return acc === null || !node ? null : acc + node.contribution.max_bytes;
  }, 0);
  const usable = plan.assignments.reduce((acc, a) => acc + a.budget_bytes, 0);
  const needed = plan.assignments.reduce((acc, a) => acc + a.total_bytes_est, 0);
  const tone = plan.fit_label === 'estimated_fit' ? 'ok' : plan.fit_label === 'does_not_fit' ? 'bad' : 'warn';
  const cpuText = cpuLayersText(plan.cpu_layers);
  const overflow = overflowText(plan);

  return (
    <Card
      title="Plan"
      subtitle={`${STRATEGY_LABELS[plan.strategy]} · context ${plan.ctx_size} tokens · ${plan.model_id}`}
      actions={
        <button type="button" className="primary" disabled={!canRun || running} onClick={onRun}>
          {running ? 'Starting…' : 'Run together'}
        </button>
      }
    >
      <PlanBadges plan={plan} />
      <p>
        <StatusText tone={tone}>{FIT_LABELS[plan.fit_label]}</StatusText>
        {' · '}
        {plan.feasible ? 'feasible' : 'not feasible'}
        <SourceNote plan={plan} />
      </p>
      <CapacityBanner plan={plan} />
      {runBlockedReason && <p className="small">{runBlockedReason}</p>}
      <ErrorNote error={runError} onDismiss={onDismissError} />

      <dl className="fields">
        <dt>Contributed / estimated usable</dt>
        <dd>
          {formatBytes(contributed)} / {formatBytes(usable)} across {plan.assignments.length} device
          {plan.assignments.length === 1 ? '' : 's'} on {machineIds.length} machine{machineIds.length === 1 ? '' : 's'}
          <div className="muted small">
            Each machine keeps its own memory. Layers are split between devices and talk over the network, so this is not one
            fast pooled memory.
          </div>
        </dd>
        <dt>This plan needs (estimate)</dt>
        <dd>{formatBytes(needed)}</dd>
        {cpuText && (
          <>
            <dt>Runs on the CPU</dt>
            <dd>
              {cpuText}
              {plan.n_gpu_units !== null && (
                <span className="muted small"> (-ngl {plan.n_gpu_units}: the remaining layers and the output head run on GPUs)</span>
              )}
            </dd>
          </>
        )}
        {overflow && (
          <>
            <dt>What does not fit this machine&apos;s GPU</dt>
            <dd>{overflow}</dd>
          </>
        )}
        {plan.engine === 'llama_rpc' && (
          <>
            <dt>Kept in this machine&apos;s RAM</dt>
            <dd>{formatBytes(plan.cpu_resident_bytes)} (token embeddings)</dd>
          </>
        )}
      </dl>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Machine</th>
              <th>Device</th>
              <th>Layers</th>
              <th>Estimated total</th>
              <th>Budget</th>
              <th>Fits</th>
            </tr>
          </thead>
          <tbody>
            {plan.assignments.map((a) => (
              <tr key={`${a.node_id}-${a.device}`}>
                <td>{nameOf(a.node_id)}</td>
                <td className="mono">
                  {a.device}
                  {a.endpoint && <span className="muted small block">{a.endpoint}</span>}
                </td>
                <td>
                  {layersText(plan, a)}
                  {a.holds_output && <span className="muted small block">+ output layer</span>}
                </td>
                <td title={`weights ${formatBytes(a.weight_bytes)}, KV ${formatBytes(a.kv_bytes)}, compute ${formatBytes(a.compute_bytes_est)}, overhead ${formatBytes(a.overhead_bytes_est)}`}>
                  {formatBytes(a.total_bytes_est)}
                </td>
                <td>{formatBytes(a.budget_bytes)}</td>
                <td>{a.fits ? 'yes' : <strong>no</strong>}</td>
              </tr>
            ))}
            {plan.assignments.length === 0 && (
              <tr>
                <td colSpan={6} className="muted">
                  No device assignments.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {plan.reasons.length > 0 && (
        <>
          <h3>Why</h3>
          <ul>
            {plan.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </>
      )}
      {plan.warnings.length > 0 && (
        <>
          <h3>Warnings</h3>
          <ul>
            {plan.warnings.map((w) => (
              <li key={w}>
                <strong>Warning:</strong> {w}
              </li>
            ))}
          </ul>
        </>
      )}
      {plan.candidates.length > 0 && (
        <>
          <h3>Options considered</h3>
          <ul className="candidates">
            {plan.candidates.map((c) => (
              <li key={`${c.label}-${c.devices.join(',')}`}>
                <strong>{c.label}</strong> <span className="mono small">[{c.devices.join(', ')}]</span>{' '}
                {c.feasible ? 'feasible' : 'not feasible'}: {c.reason}
              </li>
            ))}
          </ul>
        </>
      )}
    </Card>
  );
}
