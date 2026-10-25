import { eventKey } from '../api/client';
import type { EventFeed } from '../hooks/useEvents';
import { formatTime } from '../lib/format';
import { Card } from './common';

const STATUS_TEXT: Record<EventFeed['status'], string> = {
  connecting: 'connecting…',
  live: 'live',
  reconnecting: 'reconnecting…',
};

export function ActivityFeed({ feed }: { feed: EventFeed }) {
  return (
    <Card title="Activity" subtitle={`Event stream: ${STATUS_TEXT[feed.status]}`} className="activity">
      {feed.detail && <p className="small">{feed.detail}</p>}
      {feed.events.length === 0 && <p className="muted small">No events yet.</p>}
      <ol className="feed">
        {feed.events.map((e) => (
          <li key={eventKey(e)}>
            <span className="mono small muted">{formatTime(e.at)}</span>
            <span className="feed-type mono small">{e.type}</span>
            <span className="feed-msg">{e.message}</span>
          </li>
        ))}
      </ol>
    </Card>
  );
}
