import { useEffect, useRef, useState } from 'react';
import { ApiError, toApiError } from '../api/client';

export interface PollState<T> {
  /** Last successful value; kept while later polls fail so the screen does not blank out. */
  data: T | undefined;
  error: ApiError | null;
}

/**
 * Poll `fetcher` every `intervalMs` (measured from the end of the previous request, so requests
 * never overlap). Changing `refreshKey` triggers an immediate extra fetch.
 */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number, refreshKey = 0): PollState<T> {
  const [state, setState] = useState<PollState<T>>({ data: undefined, error: null });
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const inFlight = useRef(false);
  const rerun = useRef(false);
  const mounted = useRef(true);

  const runRef = useRef(async (): Promise<void> => {
    if (inFlight.current) {
      rerun.current = true;
      return;
    }
    inFlight.current = true;
    try {
      const data = await fetcherRef.current();
      if (mounted.current) setState({ data, error: null });
    } catch (err) {
      if (mounted.current) setState((prev) => ({ data: prev.data, error: toApiError(err) }));
    } finally {
      inFlight.current = false;
    }
    if (rerun.current) {
      rerun.current = false;
      await runRef.current();
    }
  });

  useEffect(() => {
    mounted.current = true;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const loop = async (): Promise<void> => {
      await runRef.current();
      if (!cancelled) timer = setTimeout(() => void loop(), intervalMs);
    };
    void loop();
    return () => {
      cancelled = true;
      mounted.current = false;
      clearTimeout(timer);
    };
  }, [intervalMs]);

  useEffect(() => {
    if (refreshKey > 0) void runRef.current();
  }, [refreshKey]);

  return state;
}
