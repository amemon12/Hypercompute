import { createContext, useContext } from 'react';
import type { BenchmarkResult, Deployment, LivePlan, MeasuredProfile, ModelView, NodeView, PlanRequest, SessionView } from './types';
export interface LiveRoomValue {
  online: boolean; connection: 'connecting' | 'online' | 'offline'; error: string | null; actionError: string | null;
  busy: string | null; updatedAt: number | null; nodes: NodeView[]; actualNodes: NodeView[];
  models: ModelView[]; session: SessionView | null; plan: LivePlan | null; deployments: Deployment[];
  activeDeployment: Deployment | null; benchmarks: BenchmarkResult[]; profiles: MeasuredProfile[];
  refresh: () => Promise<void>; clearPlan: () => void; clearError: () => void;
  createSession: () => Promise<void>; issueInvite: () => Promise<void>; checkPlan: (request: PlanRequest) => Promise<void>;
  deploy: () => Promise<void>; stop: () => Promise<void>; download: (id: string) => Promise<void>; benchmark: () => Promise<void>;
}
export const LiveRoomContext = createContext<LiveRoomValue | null>(null);
export function useLiveRoom(): LiveRoomValue { const value = useContext(LiveRoomContext); if (!value) throw new Error('useLiveRoom needs LiveRoomProvider'); return value; }
