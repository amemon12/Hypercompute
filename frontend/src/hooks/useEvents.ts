import { useEffect, useRef, useState } from 'react';
import { subscribeEvents, type EventStreamStatus } from '../api/client';
import type { HcEvent } from '../api/types';

const MAX_EVENTS = 200;
const ACTIVITY_THROTTLE_MS = 400;

export interface EventFeed {
  events: HcEvent[]; // newest first
  status: EventStreamStatus;
  detail: string | null;
}

/** Live activity feed from GET /api/events; `onActivity` is called (throttled) after events arrive. */
export function useEvents(onActivity: () => void): EventFeed {
  const [feed, setFeed] = useState<EventFeed>({ events: [], status: 'connecting', detail: null });
  const onActivityRef = useRef(onActivity);
  onActivityRef.current = onActivity;

  useEffect(() => {
    let pending: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = subscribeEvents(
      (event) => {
        setFeed((prev) => ({ ...prev, events: [event, ...prev.events].slice(0, MAX_EVENTS) }));
        if (pending === undefined) {
          pending = setTimeout(() => {
            pending = undefined;
            onActivityRef.current();
          }, ACTIVITY_THROTTLE_MS);
        }
      },
      (status, detail) => setFeed((prev) => ({ ...prev, status, detail: detail ?? null })),
    );
    return () => {
      clearTimeout(pending);
      unsubscribe();
    };
  }, []);

  return feed;
}
