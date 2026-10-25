import { describe, expect, it } from 'vitest';
import type { Deployment, DeploymentState } from '../api/types';
import { stepStatuses } from './DeploymentPanel';

function deployment(state: DeploymentState, history: DeploymentState[]): Deployment {
  return {
    state,
    history: history.map((s) => ({ state: s, at: '2026-10-20T12:00:00Z', detail: null })),
  } as unknown as Deployment;
}

describe('stepStatuses', () => {
  it('marks finished steps done and the live one current', () => {
    expect(stepStatuses(deployment('loading', ['planned', 'reserving', 'preparing', 'loading']))).toEqual([
      'done',
      'done',
      'done',
      'current',
      'todo',
      'todo',
    ]);
  });

  it('marks the step that was running when the deployment failed', () => {
    expect(stepStatuses(deployment('failed', ['planned', 'reserving', 'preparing', 'failed']))).toEqual([
      'done',
      'done',
      'failed',
      'todo',
      'todo',
      'todo',
    ]);
  });

  it('shows every step done once a ready deployment is stopping', () => {
    const all: DeploymentState[] = ['planned', 'reserving', 'preparing', 'loading', 'warming', 'ready', 'draining'];
    expect(stepStatuses(deployment('draining', all))).toEqual(['done', 'done', 'done', 'done', 'done', 'done']);
  });
});
