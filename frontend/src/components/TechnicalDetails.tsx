import { isFullEngineInfo, type Deployment, type EngineInfo, type Health, type ModelView, type NodeView, type Plan } from '../api/types';
import { ENGINE_LABELS, FORMAT_LABELS } from '../lib/engines';
import { formatBytes, formatCount, formatDateTime, formatTime, quoteArg } from '../lib/format';
import { sortNodes } from './PeerCards';

/** "flash_attn=on ubatch=512", or "none". */
export function formatEngineFlags(flags: Record<string, string | number>): string {
  const parts = Object.entries(flags).map(([k, v]) => `${k}=${v}`);
  return parts.length > 0 ? parts.join(' ') : 'none';
}

interface TechnicalDetailsProps {
  deployment: Deployment | null;
  plan: Plan | null;
  nodes: NodeView[] | undefined;
  models: ModelView[] | undefined;
  health: Health | undefined;
}

function DeploymentDetails({ d }: { d: Deployment }) {
  const p = d.placement;
  return (
    <section>
      <h3>Deployment {d.id}</h3>
      <dl className="fields">
        <dt>State</dt>
        <dd>{d.state}</dd>
        <dt>Engine process</dt>
        <dd className="mono">
          pid {d.pid ?? 'unavailable'} · upstream port {d.upstream_port ?? 'unavailable'}
        </dd>
        <dt>Participation verified</dt>
        <dd>
          {d.participation_verified ? 'yes' : 'no'}
          {d.participation_detail && <span className="muted"> · {d.participation_detail}</span>}
        </dd>
        <dt>Idempotency key</dt>
        <dd className="mono small">{d.idempotency_key ?? 'none'}</dd>
      </dl>
      <h4>Engine arguments</h4>
      <pre className="code">{d.engine_args.length > 0 ? d.engine_args.map(quoteArg).join(' ') : 'not launched yet'}</pre>
      <h4>Observed placement (parsed from the {ENGINE_LABELS[d.plan.engine]} load log)</h4>
      {p ? (
        <>
          <p className="small">
            Offloaded layers: <span className="mono">{p.offloaded ?? 'unavailable'}</span> · parsed {formatTime(p.parsed_at)}
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Device</th>
                  <th>Model buffer</th>
                  <th>KV buffer</th>
                  <th>Compute buffer</th>
                  <th>Layers (est.)</th>
                </tr>
              </thead>
              <tbody>
                {p.devices.map((dev) => (
                  <tr key={dev.name}>
                    <td className="mono">{dev.name}</td>
                    <td>{formatBytes(dev.model_bytes)}</td>
                    <td>{formatBytes(dev.kv_bytes)}</td>
                    <td>{formatBytes(dev.compute_bytes)}</td>
                    <td>{formatCount(dev.layers_est)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {p.evidence_lines.length > 0 && (
            <details>
              <summary className="small">Evidence lines ({p.evidence_lines.length})</summary>
              <pre className="code">{p.evidence_lines.join('\n')}</pre>
            </details>
          )}
        </>
      ) : (
        <p className="muted small">Not observed yet (appears once the engine has loaded the model).</p>
      )}
      <h4>History</h4>
      <ul className="small">
        {d.history.map((h, i) => (
          <li key={`${h.state}-${i}`}>
            <span className="mono">{formatTime(h.at)}</span> {h.state}
            {h.detail ? ` · ${h.detail}` : ''}
          </li>
        ))}
      </ul>
    </section>
  );
}

function PlanDetails({ plan, models }: { plan: Plan; models: ModelView[] | undefined }) {
  const model = models?.find((m) => m.artifact.id === plan.model_id);
  return (
    <section>
      <h3>Plan {plan.id}</h3>
      <dl className="fields">
        <dt>Engine / tier</dt>
        <dd className="mono">
          {plan.engine} · {plan.tier}
          {plan.capacity_mode ? ' · capacity mode' : ''}
        </dd>
        <dt>Measured profile</dt>
        <dd className="mono">{plan.profile_id ?? 'none (estimate)'}</dd>
        {plan.cpu_layers > 0 && (
          <>
            <dt>CPU layers / -ngl</dt>
            <dd className="mono">
              {plan.cpu_layers} / {plan.n_gpu_units ?? 'unavailable'}
            </dd>
          </>
        )}
        <dt>Engine flags</dt>
        <dd className="mono">{formatEngineFlags(plan.engine_flags)}</dd>
        <dt>--device</dt>
        <dd className="mono">{plan.device_list.join(',') || 'none'}</dd>
        <dt>--tensor-split</dt>
        <dd className="mono">{plan.tensor_split.join(',') || 'none'}</dd>
        <dt>--rpc</dt>
        <dd className="mono">{plan.rpc_endpoints.join(',') || 'none'}</dd>
        <dt>Context</dt>
        <dd>{plan.ctx_size} tokens</dd>
      </dl>
      {Object.keys(plan.engine_options).length > 0 && (
        <>
          <h4>Engine options</h4>
          <pre className="code">{JSON.stringify(plan.engine_options, null, 2)}</pre>
        </>
      )}
      <h4>Exact model</h4>
      {model ? (
        <dl className="fields">
          <dt>Repository</dt>
          <dd className="mono">{model.artifact.repo}</dd>
          <dt>Revision</dt>
          <dd className="mono">{model.artifact.revision}</dd>
          <dt>Format</dt>
          <dd>
            {FORMAT_LABELS[model.artifact.format]}
            {model.artifact.logical_id && <span className="muted small"> · variant of {model.artifact.logical_id}</span>}
          </dd>
          {model.artifact.files.map((f) => (
            <FileRow key={f.path ?? f.name} name={f.path ?? f.name} sha256={f.sha256} size={f.size_bytes} />
          ))}
          {model.status === 'engine_managed' ? (
            <>
              <dt>Files</dt>
              <dd>
                fetched by the engine on each machine · {formatBytes(model.artifact.size_bytes)}
              </dd>
            </>
          ) : (
            <>
              <dt>Checksum verified locally</dt>
              <dd>{model.verified_sha256 ? 'yes' : 'no'}</dd>
            </>
          )}
          {model.artifact.format === 'gguf' && (
            <>
              <dt>GGUF</dt>
              <dd>
                {model.gguf
                  ? `${model.gguf.architecture ?? 'architecture unavailable'} · ${formatCount(model.gguf.n_layer)} layers · ${model.gguf.file_type ?? 'type unavailable'}`
                  : 'not read yet'}
              </dd>
            </>
          )}
          <dt>License</dt>
          <dd>{model.artifact.license}</dd>
        </dl>
      ) : (
        <p className="muted small">Model {plan.model_id} is not in the gateway&apos;s model list.</p>
      )}
    </section>
  );
}

function FileRow({ name, sha256, size }: { name: string; sha256: string; size: number }) {
  return (
    <>
      <dt>File</dt>
      <dd>
        <span className="mono">{name}</span> · {formatBytes(size)}
        <div className="mono small wrap">sha256 {sha256}</div>
      </dd>
    </>
  );
}

function EngineDetails({ engine: e }: { engine: EngineInfo }) {
  return (
    <section>
      <h3>Engine on this machine</h3>
      <dl className="fields">
        <dt>Release</dt>
        <dd className="mono">
          {e.release ?? 'unavailable'} (build {e.build_number ?? 'unavailable'}, commit {e.commit ?? 'unavailable'})
        </dd>
        <dt>RPC protocol</dt>
        <dd className="mono">{e.rpc_proto ?? 'unavailable'}</dd>
        <dt>Variant</dt>
        <dd className="mono">
          {e.variant ?? 'unavailable'} · {e.accelerator}
        </dd>
        <dt>Smoke test</dt>
        <dd>
          {e.smoke}
          {e.smoke_detail && <span className="muted"> · {e.smoke_detail}</span>}
        </dd>
        <dt>Engine devices</dt>
        <dd>
          {e.devices.length === 0
            ? 'none reported'
            : e.devices.map((dev) => (
                <div key={dev.name} className="small">
                  <span className="mono">{dev.name}</span> {dev.description} · total {formatBytes(dev.total_bytes)} · free{' '}
                  {formatBytes(dev.free_bytes)}
                </div>
              ))}
        </dd>
      </dl>
    </section>
  );
}

function MemoryPools({ nodes }: { nodes: NodeView[] }) {
  return (
    <section>
      <h3>Memory pools per machine</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Machine</th>
              <th>Pool</th>
              <th>Kind</th>
              <th>Total</th>
              <th>Free</th>
              <th>Source</th>
              <th>Measured</th>
            </tr>
          </thead>
          <tbody>
            {sortNodes(nodes).flatMap((n) =>
              n.inventory.memory_pools.map((pool) => (
                <tr key={`${n.id}-${pool.id}`}>
                  <td>{n.display_name}</td>
                  <td className="mono">{pool.id}</td>
                  <td>{pool.kind}</td>
                  <td>{formatBytes(pool.total_bytes)}</td>
                  <td>{formatBytes(pool.free_bytes)}</td>
                  <td className="small">
                    {pool.source}
                    {pool.note && <span className="muted block">{pool.note}</span>}
                  </td>
                  <td className="small">{formatDateTime(pool.measured_at)}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function TechnicalDetails({ deployment, plan, nodes, models, health }: TechnicalDetailsProps) {
  const shownPlan = deployment?.plan ?? plan;
  return (
    <details className="card tech">
      <summary>
        <h2>Technical details</h2>
      </summary>
      {deployment && <DeploymentDetails d={deployment} />}
      {shownPlan && <PlanDetails plan={shownPlan} models={models} />}
      {health && isFullEngineInfo(health.engine) && <EngineDetails engine={health.engine} />}
      {nodes && nodes.length > 0 && <MemoryPools nodes={nodes} />}
      {!deployment && !shownPlan && !health && <p className="muted">Nothing to show yet.</p>}
    </details>
  );
}
