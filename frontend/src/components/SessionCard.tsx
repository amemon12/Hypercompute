import { useState } from 'react';
import { api, type ApiError } from '../api/client';
import type { InviteView, SessionView } from '../api/types';
import { useAction } from '../hooks/useAction';
import { useNow } from '../hooks/useNow';
import { formatCountdown, formatDateTime } from '../lib/format';
import { Card, CopyBlock, ErrorNote, Field } from './common';

interface SessionCardProps {
  session: SessionView | null | undefined;
  loadError: ApiError | null;
  onChanged: () => void;
}

function InviteBlock({ invite }: { invite: InviteView }) {
  const now = useNow(1000);
  const remaining = formatCountdown(invite.expires_at, now);
  const expired = remaining === 'expired';
  return (
    <div className="invite">
      <p>
        Each friend runs this command in a terminal <strong>on their own computer</strong>. Opening this page
        in a browser cannot contribute GPU or RAM.
      </p>
      <CopyBlock text={invite.join_command} label="Join command" />
      <dl className="fields">
        <Field label="Code">
          <span className="mono big">{invite.code}</span>
        </Field>
        <Field label="Expires">
          {expired ? (
            <strong>Expired. Create a new invite.</strong>
          ) : (
            <span>
              in <span className="mono">{remaining}</span>
            </span>
          )}
        </Field>
        <Field label="Join address">
          <span className="mono">{invite.join_address}</span>
        </Field>
      </dl>
    </div>
  );
}

export function SessionCard({ session, loadError, onChanged }: SessionCardProps) {
  // The plaintext invite code is only returned when it is created, so it lives in memory here.
  const [invite, setInvite] = useState<InviteView | null>(null);
  const keepInvite = (view: SessionView): void => {
    setInvite(view.invite ?? null);
    onChanged();
  };
  const create = useAction(() => api.createSession(), keepInvite);
  const renew = useAction((id: string) => api.newInvite(id), keepInvite);
  const close = useAction((id: string) => api.closeSession(id), () => {
    setInvite(null);
    onChanged();
  });
  const error = create.error ?? renew.error ?? close.error;
  const active = session && session.status === 'active' ? session : null;
  const shownInvite = active ? invite : null;

  const onClose = (id: string): void => {
    if (window.confirm('End this session? Every friend is told they were removed and stops contributing.')) {
      void close.run(id);
    }
  };

  return (
    <Card
      title="Compute session"
      subtitle={active ? `${active.name} · started ${formatDateTime(active.created_at)}` : 'No session yet'}
      actions={
        active ? (
          <>
            <button type="button" className="primary" disabled={renew.busy} onClick={() => void renew.run(active.id)}>
              {renew.busy ? 'Creating…' : 'New invite'}
            </button>
            <button type="button" disabled={close.busy} onClick={() => onClose(active.id)}>
              End session
            </button>
          </>
        ) : (
          <button type="button" className="primary" disabled={create.busy || (session === undefined && !loadError)} onClick={() => void create.run()}>
            {create.busy ? 'Creating…' : 'Create compute session'}
          </button>
        )
      }
    >
      {session === undefined && !loadError && <p className="muted">Loading session…</p>}
      <ErrorNote error={loadError} />
      <ErrorNote error={error} onDismiss={() => [create, renew, close].forEach((a) => a.clearError())} />
      {!active && session !== undefined && (
        <p>Create a session to get a one-time join command for a friend on your network.</p>
      )}
      {active && shownInvite && <InviteBlock invite={shownInvite} />}
      {active && !shownInvite && (
        <p>
          Invite codes are shown only once, when they are created. Press <strong>New invite</strong> to get a join
          command for the next friend.
        </p>
      )}
      {active && (
        <p className="muted small">
          LAN addresses: <span className="mono">{active.lan_addresses.join(', ') || 'unavailable'}</span> · port{' '}
          <span className="mono">{active.gateway_port}</span>
        </p>
      )}
    </Card>
  );
}
