import { useCallback, useRef, useState } from 'react';
import { ApiError, toApiError } from '../api/client';

export interface Action<A extends unknown[], R> {
  run: (...args: A) => Promise<R | undefined>;
  busy: boolean;
  error: ApiError | null;
  clearError: () => void;
}

/** Wrap a mutation: tracks busy state and keeps the friendly {detail, code, fix} error. */
export function useAction<A extends unknown[], R>(
  fn: (...args: A) => Promise<R>,
  onSuccess?: (result: R) => void,
): Action<A, R> {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const onSuccessRef = useRef(onSuccess);
  onSuccessRef.current = onSuccess;

  const run = useCallback(async (...args: A): Promise<R | undefined> => {
    setBusy(true);
    setError(null);
    try {
      const result = await fnRef.current(...args);
      onSuccessRef.current?.(result);
      return result;
    } catch (err) {
      setError(toApiError(err));
      return undefined;
    } finally {
      setBusy(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);
  return { run, busy, error, clearError };
}
