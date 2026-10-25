import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { api, ApiError, getBootstrap, newIdempotencyKey, subscribeEvents } from '../../../src/api/client';
import type { BenchmarkResult, Deployment, LivePlan, MeasuredProfile, ModelView, NodeView, SessionView } from './types';
import { LiveRoomContext, type LiveRoomValue } from './context';
import './live.css';

export function errorMessage(error: unknown): string { return error instanceof ApiError ? [error.message, error.fix].filter(Boolean).join(' ') : error instanceof Error ? error.message : 'Something went wrong. Please try again.'; }

export function LiveRoomProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<LiveRoomValue['connection']>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [nodes, setNodes] = useState<NodeView[]>([]);
  const [models, setModels] = useState<ModelView[]>([]);
  const [session, setSession] = useState<SessionView | null>(null);
  const [plan, setPlan] = useState<LivePlan | null>(null);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([]);
  const [profiles, setProfiles] = useState<MeasuredProfile[]>([]);
  const planGeneration = useRef(0);
  const clearPlan = useCallback(() => { planGeneration.current++; setPlan(null); }, []);
  const fetching = useRef(false);
  const actionLock = useRef(false);
  const alive = useRef(true);
  const deploymentKey = useRef<{ planId: string; key: string } | null>(null);
  const refresh = useCallback(async () => {
    if (fetching.current) return;
    fetching.current = true;
    try {
      const freshNodes = await api.nodes();
      if (!alive.current) return;
      setNodes(freshNodes); setConnection('online'); setUpdatedAt(Date.now()); setError(null);
      const results = await Promise.allSettled([api.currentSession(), api.models(), api.deployments(), api.benchmarks(), api.profiles()]);
      if (!alive.current) return;
      const [s, m, d, b, p] = results;
      if (s.status === 'fulfilled') setSession(previous => s.value && previous?.id === s.value.id ? { ...s.value, invite: s.value.invite ?? previous.invite } : s.value);
      if (m.status === 'fulfilled') setModels(m.value);
      if (d.status === 'fulfilled') setDeployments(d.value);
      if (b.status === 'fulfilled') setBenchmarks(b.value);
      if (p.status === 'fulfilled') setProfiles(p.value);
      const failure = results.find(r => r.status === 'rejected');
      if (failure?.status === 'rejected') setError(`Some live details could not refresh: ${errorMessage(failure.reason)}`);
    } catch (cause) {
      if (alive.current) { setConnection('offline'); setError(errorMessage(cause)); }
    } finally { fetching.current = false; }
  }, []);
  useEffect(() => {
    alive.current = true; void refresh();
    const interval = window.setInterval(() => { if (!document.hidden) void refresh(); }, 6000);
    const onVisible = () => { if (!document.hidden) void refresh(); };
    document.addEventListener('visibilitychange', onVisible);
    return () => { alive.current = false; clearInterval(interval); document.removeEventListener('visibilitychange', onVisible); };
  }, [refresh]);
  useEffect(() => {
    if (connection !== 'online') return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = subscribeEvents(() => { if (!timer) timer = setTimeout(() => { timer = undefined; void refresh(); }, 750); }, () => {});
    return () => { unsubscribe(); clearTimeout(timer); };
  }, [connection, refresh]);
  const perform = useCallback(async (name: string, work: () => Promise<unknown>) => {
    if (actionLock.current) return;
    actionLock.current = true; setBusy(name); setActionError(null);
    try { await work(); await refresh(); }
    catch (cause) { setActionError(errorMessage(cause)); }
    finally { actionLock.current = false; setBusy(null); }
  }, [refresh]);
  const activeDeployment = useMemo(() => [...deployments].sort((a, b) => b.created_at.localeCompare(a.created_at)).find(d => !['failed', 'stopped'].includes(d.state)) ?? null, [deployments]);
  const value: LiveRoomValue = {
    online: connection === 'online', connection, error, actionError, busy, updatedAt, nodes, actualNodes: nodes,
    models, session, plan, deployments, activeDeployment, benchmarks, profiles, refresh,
    clearPlan, clearError: () => setActionError(null),
    createSession: () => perform('Creating session', async () => { await getBootstrap(); setSession(await api.createSession('Our compute room')); }),
    issueInvite: () => perform('Creating invite', async () => { if (session) setSession(await api.newInvite(session.id)); }),
    checkPlan: request => perform('Checking plan', async () => {
      const generation = ++planGeneration.current; setPlan(null);
      const result = await api.createPlan(request) as LivePlan;
      if (generation === planGeneration.current && alive.current) setPlan(result);
    }),
    deploy: () => perform('Starting run', async () => {
      if (!plan?.feasible) throw new Error('Check a feasible plan first.');
      if (!deploymentKey.current || deploymentKey.current.planId !== plan.id) deploymentKey.current = { planId: plan.id, key: newIdempotencyKey() };
      const next = await api.createDeployment(plan.id, deploymentKey.current.key);
      setDeployments(previous => [next, ...previous.filter(d => d.id !== next.id)]);
    }),
    stop: () => perform('Stopping run', async () => { if (activeDeployment) await api.stopDeployment(activeDeployment.id); }),
    download: id => perform('Starting download', () => api.downloadModel(id)),
    benchmark: () => perform('Starting benchmark', () => {
      if (!activeDeployment || activeDeployment.state !== 'ready' || activeDeployment.purpose === 'tune') throw new Error('A ready run is required.');
      return api.startBenchmark({ deployment_id: activeDeployment.id, prompt_set: 'both', repetitions: 3, warmup: 1, max_tokens: 128, label: 'Website room measurement' });
    }),
  };
  return <LiveRoomContext.Provider value={value}>{children}</LiveRoomContext.Provider>;
}
