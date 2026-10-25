import { api } from '../api/client';
import { TERMINAL_STATES, type Deployment, type DeploymentState } from '../api/types';
import { useAction } from '../hooks/useAction';
import { TRIAL_NOTE, isTuningTrial } from '../lib/deployments';
import { ENGINE_LABELS, TIER_LABELS } from '../lib/engines';
import { formatTime } from '../lib/format';
import { Card, ErrorNote, StatusText } from './common';
import { CapacityBanner } from './PlanView';

const STEPS: { state: DeploymentState; label: string }[] = [
  { state: 'planned', label: 'Planned' },
  { state: 'reserving', label: 'Reserving' },
  { state: 'preparing', label: 'Preparing' },
  { state: 'loading', label: 'Loading' },
  { state: 'warming', label: 'Warming up' },
  { state: 'ready', label: 'Ready' },
];

type StepStatus = 'done' | 'current' | 'failed' | 'todo';

/** Map history onto the fixed stepper; a failure marks the step that was running when it failed. */
export function stepStatuses(d: Deployment): StepStatus[] {
  const reached = new Set(d.history.map((h) => h.state));
  reached.add(d.state);
  const lastBeforeFailure = [...d.history].reverse().find((h) => h.state !== 'failed')?.state ?? 'planned';
  return STEPS.map(({ state }) => {
    if (d.state === 'failed' && state === lastBeforeFailure) return 'failed';
    if (state === d.state) return 'current';
    return reached.has(state) ? 'done' : 'todo';
  });
}

interface DeploymentPanelProps {
  deployment: Deployment | null;
  onChanged: () => void;
}

export function DeploymentPanel({ deployment, onChanged }: DeploymentPanelProps) {
  const stop = useAction((id: string) => api.stopDeployment(id), onChanged);
  if (!deployment) {
    return (
      <Card title="Run" subtitle="Nothing is running">
        <p className="muted">Check a plan, then press Run together.</p>
      </Card>
    );
  }
  const d = deployment;
  const statuses = stepStatuses(d);
  const terminal = TERMINAL_STATES.has(d.state);
  const timeOf = (state: DeploymentState): string | null => {
    const t = [...d.history].reverse().find((h) => h.state === state);
    return t ? formatTime(t.at) : null;
  };

  const trial = isTuningTrial(d);
  return (
    <Card
      title={trial ? 'Tuning trial' : 'Run'}
      subtitle={`${d.plan.model_id} · ${ENGINE_LABELS[d.plan.engine]} · ${TIER_LABELS[d.plan.tier]} · started ${formatTime(d.created_at)}`}
      actions={
        !terminal && (
          <button type="button" disabled={stop.busy || d.state === 'draining'} onClick={() => void stop.run(d.id)}>
            {stop.busy || d.state === 'draining' ? 'Stopping…' : 'Stop'}
          </button>
        )
      }
    >
      <ol className="stepper" aria-label="Deployment progress">
        {STEPS.map((step, i) => {
          const status = statuses[i] ?? 'todo';
          const at = timeOf(step.state);
          return (
            <li key={step.state} className={`step step-${status}`} aria-current={status === 'current' ? 'step' : undefined}>
              <span className="step-label">{step.label}</span>
              <span className="step-sub small">
                {status === 'failed' ? 'failed here' : status === 'current' ? 'now' : status === 'done' ? (at ?? 'done') : ''}
              </span>
            </li>
          );
        })}
      </ol>

      {trial && <p className="small">{TRIAL_NOTE} Chat and benchmarks stay off for it.</p>}
      {!terminal && <CapacityBanner plan={d.plan} />}
      {d.state === 'draining' && <p>Stopping: releasing the engine and the friends&apos; workers.</p>}
      {d.state === 'stopped' && <p>Stopped.</p>}
      {d.progress && !terminal && (
        <p>
          <span className="muted">Now:</span> {d.progress}
        </p>
      )}
      {d.state === 'failed' && (
        <div className="error-note" role="alert">
          <p className="error-detail">Failed: {d.error ?? 'no error message was recorded'}</p>
          <p className="small">
            <strong>Cleanup:</strong> {d.cleanup ?? 'unavailable'}
          </p>
        </div>
      )}
      {(d.state === 'ready' || d.participation_detail) && (
        <p className="small">
          {d.participation_verified ? (
            <StatusText tone="ok">Every planned device holds model weights (verified from the engine load log)</StatusText>
          ) : (
            <StatusText tone="warn">Participation not verified yet</StatusText>
          )}
          {d.participation_detail && <span className="muted"> · {d.participation_detail}</span>}
        </p>
      )}
      <ErrorNote error={stop.error} onDismiss={stop.clearError} />
    </Card>
  );
}
