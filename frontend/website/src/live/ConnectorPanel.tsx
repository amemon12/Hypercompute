import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { errorMessage } from './LiveRoom';
import { errorFromBody, newIdempotencyKey } from '../../../src/api/client';
import { formatBytes } from './calculations';
import { CopyButton } from './common';
import type { ConnectorBootstrap, ConnectorStatus } from './types';

let connectorToken: string | null = null;
async function connectorRequest<T>(path: string, body?: object, idempotencyKey?: string, retry = true): Promise<T> {
  const mutation = path !== '/bootstrap' && path !== '/status';
  if (mutation && !connectorToken) connectorToken = (await connectorRequest<ConnectorBootstrap>('/bootstrap')).control_token;
  const response = await fetch(`/local-api${path}`, { method: mutation ? 'POST' : 'GET', cache: 'no-store', headers: {
    Accept: 'application/json', ...(body ? { 'Content-Type': 'application/json' } : {}), ...(mutation ? { 'X-Hypercompute-Connector': connectorToken! } : {}),
    ...(idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {}),
  }, ...(body ? { body: JSON.stringify(body) } : {}) });
  if (response.status === 403 && mutation && retry) { connectorToken = null; return connectorRequest(path, body, idempotencyKey, false); }
  let data: unknown; try { data = await response.json(); } catch { throw new Error('The local connector did not return a readable response.'); }
  if (!response.ok) throw errorFromBody(response.status, data, `Connector request failed (${response.status}).`);
  return data as T;
}

export function ConnectorPanel() {
  const [status, setStatus] = useState<ConnectorStatus | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [online, setOnline] = useState(false); const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false); const [host, setHost] = useState(''); const [code, setCode] = useState('');
  const [name, setName] = useState(''); const [maxMemory, setMaxMemory] = useState(''); const [reserve, setReserve] = useState('');
  const [devices, setDevices] = useState<string[]>([]); const initializedDevices = useRef(false);
  const actionLock = useRef(false);
  const requestKey = useRef<string | null>(null); const inFlight = useRef(false); const alive = useRef(true);
  const refresh = useCallback(async () => {
    if (inFlight.current) return; inFlight.current = true;
    try {
      const bootstrap = await connectorRequest<ConnectorBootstrap>('/bootstrap');
      if (!alive.current) return;
      connectorToken = bootstrap.control_token;
      const next = { ...bootstrap.status, inventory: bootstrap.status.inventory ?? bootstrap.inventory };
      setStatus(next); setOnline(true); setConnectionError(null);
      if (next.stage === 'error' && !next.node_id) requestKey.current = null;
      if (!initializedDevices.current && next.inventory) {
        const available = next.inventory.engine.devices.map(d => d.name);
        const accel = available.filter(d => /^(CUDA|MTL|Vulkan|ROCm)/.test(d));
        setDevices(accel.length ? accel : ['CPU']); initializedDevices.current = true;
      }
    } catch (cause) { if (alive.current) { setOnline(false); setConnectionError(errorMessage(cause)); } }
    finally { inFlight.current = false; }
  }, []);
  useEffect(() => { alive.current = true; void refresh(); const timer = setInterval(() => { if (!document.hidden) void refresh(); }, 4000); return () => { alive.current = false; clearInterval(timer); }; }, [refresh]);
  const work = async (path: string, body?: object, key?: string) => {
    if (actionLock.current) return; actionLock.current = true; setBusy(true); setError(null);
    try { const next = await connectorRequest<ConnectorStatus>(path, body, key); setStatus(next); setOnline(true); setConnectionError(null); if (next.stage === 'error' && !next.node_id) requestKey.current = null; }
    catch (cause) { setError(errorMessage(cause)); }
    finally { actionLock.current = false; setBusy(false); void refresh(); }
  };
  const join = (event: FormEvent) => {
    event.preventDefault(); if (!devices.length) { setError('Choose at least one device to share.'); return; } requestKey.current ??= newIdempotencyKey();
    void work('/join', { host: host.trim(), code: code.trim(), ...(name.trim() ? { display_name: name.trim() } : {}),
      devices, ...(maxMemory.trim() ? { max_memory: `${maxMemory}GiB` } : {}), ...(reserve.trim() ? { reserve: `${reserve}GiB` } : {}) }, requestKey.current);
  };
  const joined = !!status && ['starting', 'verifying', 'ready', 'paused', 'stopping'].includes(status.stage);
  const availableDevices = status?.inventory ? [...status.inventory.engine.devices, ...(status.inventory.engine.devices.some(d => d.name === 'CPU') ? [] : [{ name: 'CPU', description: status.inventory.cpu.model || 'System processor' }])] : [];
  const resetKey = () => { requestKey.current = null; };
  return <div className="hc-connector">
    <div className="hc-live-status" role="status"><span className="hc-status-dot" data-online={online} /><span>{online ? 'Local connector connected' : 'Local connector not connected'}</span><button className="hc-live-text-button" onClick={() => { setError(null); void refresh(); }} type="button">Reconnect</button></div>
    {!online && <div className="hc-live-setup"><h3>Connect this computer</h3><p>Start the connector on this computer, then enter the host address and invite code.</p><div className="hc-live-command"><code>uv run hypercompute connect</code><CopyButton text="uv run hypercompute connect" /></div></div>}
    {status?.inventory && <p className="hc-live-fine">{status.inventory.hostname} · {status.inventory.os.system} · {status.inventory.devices.filter(d => d.kind !== 'cpu').map(d => d.name).join(', ') || status.inventory.cpu.model || 'Hardware details available locally'}</p>}
    {joined ? <div className="hc-live-run"><h3>{!online ? 'Last known contribution state.' : status.stage === 'ready' ? 'Ready to contribute.' : status.stage === 'paused' ? 'Sharing paused.' : status.stage === 'verifying' ? 'Verifying this computer…' : status.stage === 'starting' ? 'Connecting your computer…' : 'Stopping the local worker…'}</h3><p role="status">{status.detail || status.state || 'Waiting for the connector.'}</p>{status.contribution && <p className="hc-live-fine">Contribution limit {formatBytes(status.contribution.max_bytes)} · reserve {formatBytes(status.contribution.reserve_bytes)}. These limits guide placement but do not enforce a process memory cap.</p>}<div className="hc-live-actions">{status.stage === 'ready' && <button type="button" className="hc-live-button" disabled={!online || busy} onClick={() => void work('/pause')}>Pause sharing</button>}{status.stage === 'paused' && <button type="button" className="hc-live-button" disabled={!online || busy} onClick={() => void work('/resume')}>Resume sharing</button>}<button className="hc-live-button hc-live-button-quiet" type="button" disabled={!online || busy || status.stage === 'stopping'} onClick={() => void work('/leave')}>Leave session</button></div></div> :
      <form onSubmit={join} onChange={resetKey} className="hc-live-form">
        <div className="hc-live-fields"><label>Host address<input required value={host} onChange={e => setHost(e.target.value)} placeholder="192.168.1.10:8000" autoComplete="off" /></label><label>Invite code<input required value={code} onChange={e => setCode(e.target.value)} placeholder="From your host" autoComplete="off" /></label><label>This computer’s name<input value={name} onChange={e => setName(e.target.value)} placeholder={status?.inventory?.hostname || 'My computer'} maxLength={64} /></label></div>
        <details className="hc-live-details"><summary>Choose what to share</summary><div className="hc-live-fields"><label>Contribution limit · GiB<input type="number" min="0.1" step="0.1" value={maxMemory} onChange={e => setMaxMemory(e.target.value)} placeholder="Connector default" /></label><label>Keep for your own use · GiB<input type="number" min="0" step="0.1" value={reserve} onChange={e => setReserve(e.target.value)} placeholder="Connector default" /></label></div><fieldset className="hc-live-device-choices"><legend>Devices</legend>{availableDevices.map(d => <label key={d.name}><input type="checkbox" checked={devices.includes(d.name)} onChange={e => setDevices(previous => e.target.checked ? [...previous, d.name] : previous.filter(x => x !== d.name))} />{d.name} · {d.description}</label>)}</fieldset><p className="hc-live-fine">Blank memory fields keep the connector’s reserve and contribution policy.</p></details>
        <p className="hc-live-fine">Joining starts an RPC worker that has no authentication on the private network. Share only on a network you trust. Contribution limits guide placement; they do not enforce process or GPU memory caps.</p>
        <button className="hc-live-button" disabled={!online || busy || !devices.length} type="submit">{busy ? 'Joining…' : 'Join the session'} <span aria-hidden="true">↗</span></button>
      </form>}
    {status?.stage === 'error' && <button className="hc-live-button hc-live-button-quiet" type="button" disabled={!online || busy} onClick={() => void work('/leave')}>Stop sharing / retry cleanup</button>}
    {(error || connectionError || status?.error) && <div className="hc-live-error" role="alert">{error || connectionError || status?.error}{status?.fix && <p>{status.fix}</p>}</div>}
  </div>;
}
