import { useCallback, useReducer, useRef, useState } from 'react';
import { api, getBootstrap, newIdempotencyKey, type ApiError } from './api/client';
import { TERMINAL_STATES, type Deployment, type Plan, type PlanRequest } from './api/types';
import { ActivityFeed } from './components/ActivityFeed';
import { ChatPanel } from './components/ChatPanel';
import { ErrorNote } from './components/common';
import { DeploymentPanel } from './components/DeploymentPanel';
import { EnginesPanel } from './components/EnginesPanel';
import { Header } from './components/Header';
import { PeerCards } from './components/PeerCards';
import { PlanView } from './components/PlanView';
import { RunSetup } from './components/RunSetup';
import { RunsPanel } from './components/RunsPanel';
import { SessionCard } from './components/SessionCard';
import { TechnicalDetails } from './components/TechnicalDetails';
import { TuningPanel } from './components/TuningPanel';
import { useAction } from './hooks/useAction';
import { useEvents } from './hooks/useEvents';
import { usePoll } from './hooks/usePoll';
import { AttemptKeys } from './lib/attemptKeys';
import { isTuningTrial } from './lib/deployments';

const POLL_MS = 2000;
const HEALTH_POLL_MS = 5000;
const BOOTSTRAP_POLL_MS = 5000;
const ENGINES_POLL_MS = 10_000; // engine status runs local checks on the gateway; no need to follow every event
const UNREACHABLE = 'gateway_unreachable';
const fetchRuns = () => api.runs(50);

/** A down gateway is reported once in the banner instead of in every card. */
function cardError(err: ApiError | null): ApiError | null {
  return err?.code === UNREACHABLE ? null : err;
}

function runBlockedReason(plan: Plan, active: Deployment | null): string | null {
  if (active && isTuningTrial(active)) return 'A tune is measuring a tuning trial. Wait for the tune or cancel it first.';
  if (active) return `A run is already ${active.state}. Stop it before starting another.`;
  if (!plan.feasible) return 'This plan is not feasible. Change the model, strategy or friends and check again.';
  return null;
}

export function App() {
  const [tick, bump] = useReducer((x: number) => x + 1, 0);
  const refresh = useCallback(() => bump(), []);
  const feed = useEvents(refresh);

  // Cached after the first success; retried until the gateway answers.
  const bootstrap = usePoll(getBootstrap, BOOTSTRAP_POLL_MS);
  const health = usePoll(api.health, HEALTH_POLL_MS, tick);
  const session = usePoll(api.currentSession, POLL_MS, tick);
  const nodes = usePoll(api.nodes, POLL_MS, tick);
  const models = usePoll(api.models, POLL_MS, tick);
  const deployments = usePoll(api.deployments, POLL_MS, tick);
  const runs = usePoll(fetchRuns, POLL_MS, tick);
  const benchmarks = usePoll(api.benchmarks, POLL_MS, tick);
  const engines = usePoll(api.engines, ENGINES_POLL_MS);

  const [plan, setPlan] = useState<Plan | null>(null);
  const [selection, setSelection] = useState<PlanRequest | null>(null);
  // One idempotency key per attempt: a retry cannot start two deployments, a later press starts a new run.
  const attempts = useRef(new AttemptKeys(newIdempotencyKey));
  const deploy = useAction(
    (p: Plan) => attempts.current.run(p.id, (key) => api.createDeployment(p.id, key)),
    refresh,
  );

  const latest = deployments.data?.[0] ?? null;
  const active = latest && !TERMINAL_STATES.has(latest.state) ? latest : null;
  const ready = latest?.state === 'ready' ? latest : null; // may be a tuning trial: chat and benchmarks refuse those
  const maxTokens = models.data?.find((m) => m.artifact.id === latest?.plan.model_id)?.artifact.max_output_tokens;
  const blocked = plan ? runBlockedReason(plan, active) : null;
  const polls = [bootstrap, session, nodes, models, deployments, runs, benchmarks];
  const gatewayDown = polls.map((p) => p.error).find((e) => e?.code === UNREACHABLE) ?? null;

  return (
    <div className="app">
      <Header bootstrap={bootstrap.data ?? null} health={health.data} healthError={health.error} />
      {gatewayDown && (
        <div className="banner">
          <ErrorNote error={gatewayDown} />
        </div>
      )}
      {!gatewayDown && bootstrap.error && (
        <div className="banner">
          <ErrorNote error={bootstrap.error} />
          <p className="small">
            The owner dashboard only works on the coordinator computer itself, opened as http://127.0.0.1 or
            http://localhost on the gateway port.
          </p>
        </div>
      )}
      <div className="layout">
        <main className="main-col">
          <SessionCard session={session.data} loadError={cardError(session.error)} onChanged={refresh} />
          <PeerCards nodes={nodes.data} loadError={cardError(nodes.error)} onChanged={refresh} />
          <EnginesPanel engines={engines.data} loadError={cardError(engines.error)} nodes={nodes.data} />
          <RunSetup
            models={models.data}
            modelsError={cardError(models.error)}
            nodes={nodes.data}
            engines={engines.data}
            onPlan={setPlan}
            onSelection={setSelection}
            onChanged={refresh}
          />
          {plan && (
            <PlanView
              plan={plan}
              nodes={nodes.data}
              canRun={blocked === null}
              runBlockedReason={blocked}
              running={deploy.busy}
              runError={deploy.error}
              onRun={() => void deploy.run(plan)}
              onDismissError={deploy.clearError}
            />
          )}
          <DeploymentPanel deployment={latest} onChanged={refresh} />
          <ErrorNote error={cardError(deployments.error)} />
          <ChatPanel deployment={latest} maxTokens={maxTokens} />
          <TechnicalDetails deployment={latest} plan={plan} nodes={nodes.data} models={models.data} health={health.data} />
          <RunsPanel
            runs={runs.data}
            runsError={cardError(runs.error)}
            benchmarks={benchmarks.data}
            benchmarksError={cardError(benchmarks.error)}
            readyDeployment={ready}
            onChanged={refresh}
          />
          <TuningPanel
            selection={selection}
            models={models.data}
            activeRunState={active?.state ?? null}
            events={feed.events}
            tick={tick}
            onChanged={refresh}
          />
        </main>
        <aside className="side-col">
          <ActivityFeed feed={feed} />
        </aside>
      </div>
    </div>
  );
}
