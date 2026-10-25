import type { ApiError } from '../api/client';
import type { EngineId, EngineInfo, EnginesStatus, ExtraEngineInfo, NodeView } from '../api/types';
import { ENGINE_CHOICES, ENGINE_LABELS, extraEngineStatus, llamaEngineStatus, type StatusLine } from '../lib/engines';
import { Card, ErrorNote, StatusText } from './common';
import { sortNodes } from './PeerCards';

export interface EngineRow {
  key: string;
  machine: string;
  isHost: boolean;
  llama: EngineInfo;
  extras: ExtraEngineInfo[];
}

/** The host first (from the gateway's own checks), then each friend still in the session (from its inventory). */
export function engineRows(status: EnginesStatus, nodes: NodeView[] | undefined): EngineRow[] {
  const sorted = sortNodes(nodes ?? []);
  const host = sorted.find((n) => n.is_local);
  const rows: EngineRow[] = [
    { key: 'host', machine: host?.display_name ?? 'This machine', isHost: true, llama: status.llama_rpc, extras: status.host },
  ];
  for (const n of sorted) {
    if (n.is_local || n.state === 'left' || n.state === 'removed') continue;
    rows.push({
      key: n.id,
      machine: n.display_name,
      isHost: false,
      llama: n.inventory.engine,
      extras: status.nodes[n.id] ?? n.inventory.extra_engines ?? [],
    });
  }
  return rows;
}

interface EngineCellProps {
  line: StatusLine;
  info?: ExtraEngineInfo;
  usableByGateway?: boolean;
}

function EngineCell({ line, info, usableByGateway }: EngineCellProps) {
  return (
    <td>
      <StatusText tone={line.tone}>{line.text}</StatusText>
      {usableByGateway && <span className="muted small block">loaded by the gateway</span>}
      {info?.version && <span className="mono small block">{info.version}</span>}
      {info?.endpoint && <span className="mono small block">{info.endpoint}</span>}
      {info?.detail && <span className="muted small block">{info.detail}</span>}
      {line.install && <code className="small block install">{line.install}</code>}
    </td>
  );
}

function cellFor(row: EngineRow, engine: EngineId, available: string[]) {
  if (engine === 'llama_rpc') return <EngineCell key={engine} line={llamaEngineStatus(row.llama)} />;
  const info = row.extras.find((e) => e.engine === engine);
  return (
    <EngineCell
      key={engine}
      line={extraEngineStatus(info, engine)}
      info={info}
      usableByGateway={row.isHost && available.includes(engine)}
    />
  );
}

interface EnginesPanelProps {
  engines: EnginesStatus | undefined;
  loadError: ApiError | null;
  nodes: NodeView[] | undefined;
}

export function EnginesPanel({ engines, loadError, nodes }: EnginesPanelProps) {
  const rows = engines ? engineRows(engines, nodes) : [];
  return (
    <Card
      title="Engines"
      subtitle="What each machine can run. Optional engines install into their own folder; run the shown command on that machine."
    >
      <ErrorNote error={loadError} />
      {!engines && !loadError && <p className="muted">Loading…</p>}
      {engines && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Machine</th>
                {ENGINE_CHOICES.map((e) => (
                  <th key={e}>{ENGINE_LABELS[e]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.key}>
                  <td>
                    {row.machine}
                    {row.isHost && <span className="tag">host</span>}
                  </td>
                  {ENGINE_CHOICES.map((e) => cellFor(row, e, engines.available))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
