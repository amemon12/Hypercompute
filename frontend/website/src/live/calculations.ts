import type { MemoryPool, NodeView } from '../../../src/api/types';
export const GIB = 2 ** 30;
export function formatBytes(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'Unknown';
  return `${(value / GIB).toLocaleString(undefined, { maximumFractionDigits: 1 })} GiB`;
}
export function numeric(value: string): number | null { if (!value.trim()) return null; const n = Number(value); return Number.isFinite(n) && n >= 0 ? n : null; }
export function weightBytes(paramsB: number, bits: number): number | null { const bytes = paramsB * 1e9 * bits / 8; return paramsB > 0 && bits > 0 && Number.isFinite(bytes) ? bytes : null; }
export function uniquePools(nodes: NodeView[]): MemoryPool[] {
  const by = new Map<string, MemoryPool>();
  for (const node of nodes) for (const pool of node.inventory.memory_pools) by.set(`${node.id}:${pool.id}`, pool);
  return [...by.values()];
}
export function sumKnown(values: Array<number | null | undefined>): number | null { if (!values.length || values.some(v => v == null || !Number.isFinite(v))) return null; const sum = values.reduce<number>((total, value) => total + (value ?? 0), 0); return Number.isFinite(sum) ? sum : null; }
export interface ManualMachine {
  id: string; name: string; kind: 'unified' | 'discrete'; memoryGiB: string; freeGiB: string; reserveGiB: string;
  vramGiB: string; freeVramGiB: string; gpuReserveGiB: string; contributionLimitGiB: string; isHost: boolean;
}
export const initialRoom = (): ManualMachine[] => ['MacBook 1', 'MacBook 2', 'MacBook 3', 'Windows computer', 'ASUS GX10'].map((name, i) => ({
  id: `manual-${i}`, name, kind: i === 3 ? 'discrete' : 'unified', memoryGiB: '', freeGiB: '', reserveGiB: '',
  vramGiB: '', freeVramGiB: '', gpuReserveGiB: '0.5', contributionLimitGiB: '', isHost: i === 3,
}));
export function machineCapacity(machine: ManualMachine): { bytes: number | null; error: string | null; assumedFree: boolean } {
  const contributionLimit = machine.contributionLimitGiB ?? '';
  const fields = [contributionLimit, machine.memoryGiB, machine.freeGiB, machine.reserveGiB, ...(machine.kind === 'discrete' ? [machine.vramGiB, machine.freeVramGiB, machine.gpuReserveGiB] : [])];
  if (fields.some(value => value.trim() && (numeric(value) === null || !Number.isFinite(Number(value) * GIB)))) return { bytes: null, error: 'Enter finite, nonnegative memory values.', assumedFree: false };
  const total = numeric(machine.memoryGiB), free = numeric(machine.freeGiB), reserve = numeric(machine.reserveGiB);
  if (total === null || reserve === null || total <= 0) return { bytes: null, error: null, assumedFree: !machine.freeGiB };
  if ((free !== null && free > total) || reserve > total) return { bytes: null, error: 'Free memory and reserve cannot exceed installed memory.', assumedFree: false };
  const ram = Math.max(0, (free ?? total) - reserve);
  const limit = numeric(contributionLimit);
  const capped = (bytes: number) => Math.min(bytes, limit === null ? bytes : limit * GIB);
  if (machine.kind === 'unified') return { bytes: capped(ram * GIB), error: null, assumedFree: free === null };
  const vram = numeric(machine.vramGiB), freeVram = numeric(machine.freeVramGiB), gpuReserve = numeric(machine.gpuReserveGiB);
  if (vram === null || gpuReserve === null) return { bytes: null, error: null, assumedFree: free === null };
  if ((freeVram !== null && freeVram > vram) || gpuReserve > vram) return { bytes: null, error: 'Free GPU memory and its reserve cannot exceed VRAM.', assumedFree: false };
  const bytes = (Math.max(0, (freeVram ?? vram) - gpuReserve) + (machine.isHost ? ram : 0)) * GIB;
  return Number.isFinite(bytes) ? { bytes: capped(bytes), error: null, assumedFree: free === null || freeVram === null } : { bytes: null, error: 'Memory values are too large.', assumedFree: false };
}
export function copyLiveRoom(nodes: NodeView[]): ManualMachine[] {
  const format = (v: number | null | undefined) => v == null ? '' : String(v / GIB);
  return nodes.map(node => {
    const pools = [...new Map(node.inventory.memory_pools.map(pool => [pool.id, pool])).values()];
    const unified = pools.find(p => p.kind === 'unified');
    const ram = unified ?? pools.find(p => p.kind === 'system_ram');
    const vram = pools.filter(p => p.kind === 'discrete_vram');
    return { id: `copy-${node.id}`, name: node.display_name, kind: unified ? 'unified' : 'discrete',
      memoryGiB: format(ram?.total_bytes), freeGiB: format(ram?.free_bytes), reserveGiB: format(node.contribution.reserve_bytes), contributionLimitGiB: format(node.contribution.max_bytes),
      vramGiB: format(vram.length ? sumKnown(vram.map(p => p.total_bytes)) : 0), freeVramGiB: format(vram.length ? sumKnown(vram.map(p => p.free_bytes)) : 0), gpuReserveGiB: String(vram.length * 0.5), isHost: node.is_local,
    };
  });
}

/** Copper segment: eligible free memory withheld by reserves or the sharing limit. */
export function retainedReserveBytes(machine: ManualMachine): number | null {
  const available = machineCapacity(machine).bytes;
  if (available === null) return null;
  const ramFree = numeric(machine.freeGiB) ?? numeric(machine.memoryGiB)!;
  const gpuFree = numeric(machine.freeVramGiB) ?? numeric(machine.vramGiB) ?? 0;
  const eligibleFree = machine.kind === 'unified' ? ramFree : gpuFree + (machine.isHost ? ramFree : 0);
  const retained = eligibleFree * GIB - available;
  return Number.isFinite(retained) ? Math.max(0, retained) : null;
}
