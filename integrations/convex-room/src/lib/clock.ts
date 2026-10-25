import { useEffect, useRef } from "react";
import { useNow } from "./format";

/**
 * "Now" on the server's clock. Lease countdowns and the host-online check compare against server timestamps, and a
 * phone or projector clock that is off by half a minute would show a healthy host as offline. Each time the bridge's
 * heartbeat (bridgeSeenAt, every 10 s) arrives, arrival time minus that timestamp is the clock skew plus a small
 * delivery delay; the smallest of the last few samples is the estimate. The value loaded with the page can be old, so
 * only changes seen while subscribed count. Until then the local clock is used as is.
 */
export function useServerNow(bridgeSeenAt: number | undefined, everyMs = 1000): number {
  const now = useNow(everyMs);
  const initial = useRef<number | undefined>(undefined);
  const samples = useRef<number[]>([]);
  const skew = useRef(0);

  useEffect(() => {
    if (bridgeSeenAt === undefined) return;
    if (initial.current === undefined) {
      initial.current = bridgeSeenAt;
      return;
    }
    if (bridgeSeenAt === initial.current) return;
    samples.current = [...samples.current, Date.now() - bridgeSeenAt].slice(-6);
    skew.current = Math.min(...samples.current);
  }, [bridgeSeenAt]);

  return now - skew.current;
}
