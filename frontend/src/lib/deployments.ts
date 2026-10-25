// Which deployment the owner's chat and benchmarks may use. Tuner trials (purpose "tune") are deployments too,
// but the tuner holds every inference slot while it measures them and stops them when it is done.

import type { Deployment } from '../api/types';

export function isTuningTrial(d: Deployment | null | undefined): boolean {
  return d?.purpose === 'tune';
}

/** The owner's own run when it is ready, else null. A tuning trial never is. */
export function readyRun(d: Deployment | null | undefined): Deployment | null {
  return d && d.state === 'ready' && !isTuningTrial(d) ? d : null;
}

export const TRIAL_NOTE = 'The tuner loaded this tuning trial to measure it and stops it when the measurement ends.';
