import { api, type ApiError } from '../api/client';
import type { NodeState, NodeView } from '../api/types';
import { useAction } from '../hooks/useAction';
import { STALE_AFTER_S, formatAge, formatBytes, formatTime, isStale, mainDeviceName, osLabel } from '../lib/format';
import { Card, ErrorNote, Field, StatusText } from './common';

const STATE_TEXT: Record<NodeState, string> = {
  verifying: 'Verifying',
  eligible: 'Ready to contribute',
  ineligible: 'Not usable',
  stale: 'Stale (no recent heartbeat)',
  left: 'Left',
  removed: 'Removed',
};

const STATE_TONE: Record<NodeState, 'ok' | 'warn' | 'bad' | 'neutral'> = {
  verifying: 'warn',
  eligible: 'ok',
  ineligible: 'bad',
  stale: 'warn',
  left: 'neutral',
  removed: 'neutral',
};

const GONE: ReadonlySet<NodeState> = new Set<NodeState>(['left', 'removed']);

/** Coordinator first, then active friends, then those who left. */
export function sortNodes(nodes: NodeView[]): NodeView[] {
  const rank = (n: NodeView): number => (n.is_local ? 0 : GONE.has(n.state) ? 2 : 1);
  return [...nodes].sort((a, b) => rank(a) - rank(b) || a.display_name.localeCompare(b.display_name));
}

interface PeerCardProps {
  node: NodeView;
  onChanged: () => void;
}

function PeerCard({ node, onChanged }: PeerCardProps) {
  const remove = useAction((id: string) => api.removeNode(id), onChanged);
  const verify = useAction((id: string) => api.verifyNode(id), onChanged);
  const stale = isStale(node.heartbeat_age_s);
  const gone = GONE.has(node.state);
  const v = node.verification;
  const problems = node.inventory.diagnostics.filter((d) => d.level !== 'info');

  const onRemove = (): void => {
    if (window.confirm(`Remove ${node.display_name} from this session? Their worker is told to stop.`)) {
      void remove.run(node.id);
    }
  };

  return (
    <article className={`peer ${gone ? 'peer-gone' : ''}`}>
      <header className="peer-head">
        <h3>
          {node.display_name}
          {node.is_local && <span className="tag">this machine</span>}
        </h3>
        <StatusText tone={STATE_TONE[node.state]}>{STATE_TEXT[node.state]}</StatusText>
      </header>
      {node.state_reason && <p className="small">{node.state_reason}</p>}
      <dl className="fields">
        <Field label="System">{osLabel(node.inventory.os)}</Field>
        <Field label="Main device">{mainDeviceName(node.inventory)}</Field>
        <Field label="Contributes up to">
          {formatBytes(node.contribution.max_bytes)}
          <span className="muted small">
            {' '}
            (keeps {formatBytes(node.contribution.reserve_bytes)} for the owner; soft admission limit)
          </span>
        </Field>
        <Field label="Devices offered">
          <span className="mono">{node.contribution.devices.join(', ') || 'none listed'}</span>
        </Field>
        <Field label="Heartbeat">
          {stale ? (
            <StatusText tone="warn">
              stale, last {formatAge(node.heartbeat_age_s)} (over {STALE_AFTER_S} s)
            </StatusText>
          ) : (
            formatAge(node.heartbeat_age_s)
          )}
        </Field>
        <Field label="Worker">
          {node.is_local ? 'not needed (this machine runs llama-server and owns the model)' : node.worker_state ?? 'unavailable'}
          {node.worker && (
            <span className="mono small">
              {' '}
              {node.worker.host}:{node.worker.port}
            </span>
          )}
        </Field>
        <Field label="Engine">
          <span className="mono">
            {node.inventory.engine.release ?? 'unavailable'} · {node.inventory.engine.accelerator}
          </span>
        </Field>
        <Field label="Verification">
          {v ? (
            <>
              <StatusText tone={v.reachable && v.engine_match ? 'ok' : 'bad'}>
                {v.reachable ? 'reachable' : 'not reachable'} · {v.engine_match ? 'engine matches' : 'engine mismatch'}
              </StatusText>
              <span className="muted small"> at {formatTime(v.checked_at)}</span>
              <div className="small">{v.detail}</div>
              {v.fix && (
                <div className="small fix">
                  <strong>Fix:</strong> {v.fix}
                </div>
              )}
            </>
          ) : node.is_local ? (
            <StatusText tone={node.inventory.engine.smoke === 'passed' ? 'ok' : 'bad'}>
              engine smoke test {node.inventory.engine.smoke}
            </StatusText>
          ) : (
            'not checked yet'
          )}
        </Field>
      </dl>
      {problems.length > 0 && (
        <ul className="diagnostics">
          {problems.map((d) => (
            <li key={d.code}>
              <strong>{d.level}:</strong> {d.message}
              {d.fix && <span className="fix"> Fix: {d.fix}</span>}
            </li>
          ))}
        </ul>
      )}
      <ErrorNote error={remove.error ?? verify.error} onDismiss={() => [remove, verify].forEach((a) => a.clearError())} />
      {!node.is_local && !gone && (
        <div className="row-actions">
          <button type="button" disabled={verify.busy} onClick={() => void verify.run(node.id)}>
            {verify.busy ? 'Checking…' : 'Re-check'}
          </button>
          <button type="button" disabled={remove.busy} onClick={onRemove}>
            {remove.busy ? 'Removing…' : 'Remove'}
          </button>
        </div>
      )}
    </article>
  );
}

interface PeerCardsProps {
  nodes: NodeView[] | undefined;
  loadError: ApiError | null;
  onChanged: () => void;
}

export function PeerCards({ nodes, loadError, onChanged }: PeerCardsProps) {
  const sorted = nodes ? sortNodes(nodes) : [];
  const friends = sorted.filter((n) => !n.is_local && !GONE.has(n.state)).length;
  return (
    <Card
      title="Machines"
      subtitle={nodes ? `${friends} friend${friends === 1 ? '' : 's'} connected` : 'Loading…'}
    >
      <ErrorNote error={loadError} />
      {nodes && nodes.length === 0 && <p className="muted">No machines reported yet.</p>}
      <div className="peer-grid">
        {sorted.map((n) => (
          <PeerCard key={n.id} node={n} onChanged={onChanged} />
        ))}
      </div>
    </Card>
  );
}
