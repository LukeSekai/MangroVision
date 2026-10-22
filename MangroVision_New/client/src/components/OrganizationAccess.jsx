import { useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Modal from './Modal';
import { FormMessage } from './FormFeedback';
import { formRequest } from '../utils/formValidation';
import bookingPolicy from '../like/bookingPolicy.json' with { type: 'json' };

const API = import.meta.env.VITE_API_BASE || '';
const time = (value) => new Date(value).toLocaleString('en-PH', { timeZone: 'Asia/Manila', dateStyle: 'medium', timeStyle: 'short' });

export default function OrganizationAccess({ organization, activity, onCreated }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [access, setAccess] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [confirmation, setConfirmation] = useState(null);
  const inFlight = useRef(false);
  const url = activity ? `${API}/api/planters/organizations/${organization.organization_id}/activities/${activity.id}/access` : '';
  const request = async (action) => {
    if (inFlight.current || !activity) return;
    inFlight.current = true;
    setBusy(true); setError(''); setSuccess('');
    try {
      const result = await formRequest(url, action ? { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }) } : {}, 'Could not load the organization code.');
      setAccess(result.access); setLoaded(true); setConfirmation(null);
      if (action) {
        setSuccess(action === 'revoke' ? 'The code and this activity’s logins have been revoked. Saved planting records remain.'
          : action === 'replace' ? 'The replacement code is ready. Share it with the contact person; earlier emails keep their old contents.'
            : 'The organization code is ready. Share this one code with the contact person.');
        onCreated?.();
      }
    } catch (failure) {
      setError(failure.uncertain ? 'The result could not be verified. Load the current code before trying again.' : failure.message);
      setConfirmation(null);
    } finally { inFlight.current = false; setBusy(false); }
  };
  const handover = () => `${organization.organization_name} — ${activity.title}\nOpen: ${window.location.origin}/field\n`
    + `Organization code: ${access.code}\n`
    + (bookingPolicy.enforceActivityAccessWindow
      ? `Opens: ${time(access.start_at)}\nCloses: ${time(access.end_at)} (Philippine time)\n\n`
      : 'This code can be used before and after the scheduled planting time.\n\n')
    + 'Share this one code with all participants.\nFirst time: enter the organization code, then choose a personal username on the welcome screen. Returning: enter the same code and username.\n'
    + 'Choose Switch here to move an existing login to another browser. No password or PIN is needed.\n';
  const download = () => {
    const objectUrl = URL.createObjectURL(new Blob([handover()], { type: 'text/plain;charset=utf-8' }));
    const link = document.createElement('a'); link.href = objectUrl; link.download = 'organization-planting-access.txt'; link.click();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(handover()); setSuccess('Access details copied. Share them with the organization’s contact person.'); }
    catch { setError('Could not copy the code. Use Download instead.'); }
  };
  if (!activity) return <p className="text-sm">Select a confirmed planting activity above to view its organization code.{bookingPolicy.enforceActivityAccessWindow && ' Access follows that activity’s start and finish times.'}</p>;
  return <div className="assign-card">
    <p className="text-sm">One code for all {activity.expected_planters || organization.participant_count || 1} participants in this activity. Each planter chooses a unique username to keep their own progress.</p>
    <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => void request()}>{busy ? 'Loading…' : loaded ? 'Refresh code' : 'Show organization code'}</button>
    {loaded && !access && <button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={() => void request('issue')}>Create organization code</button>}
    {access && <>
      <p><strong style={{ fontFamily: 'monospace', fontSize: 20 }}>{access.code || 'Revoked'}</strong></p>
      <p className="text-sm">{bookingPolicy.enforceActivityAccessWindow
        ? <>Opens {time(access.start_at)} · Closes {time(access.end_at)} (Philippine time).</>
        : 'This code can be used before and after the scheduled planting time.'}</p>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {access.code && <><button type="button" className="btn btn-secondary btn-sm" onClick={copy}>Copy access details</button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={download}>Download</button></>}
        <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => setConfirmation('replace')}>{access.revoked ? 'Restore with new code' : 'Replace code'}</button>
        {!access.revoked && <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => setConfirmation('revoke')}>Revoke</button>}
      </div>
    </>}
    {confirmation && createPortal(<Modal open title={confirmation === 'revoke' ? 'Revoke organization code?' : 'Replace organization code?'}
      variant="warning" initialFocus="cancel" busy={busy} confirmLabel={confirmation === 'revoke' ? 'Revoke code' : 'Replace code'}
      onConfirm={() => void request(confirmation)} onCancel={() => { if (!busy) setConfirmation(null); }}>
      <p>{organization.organization_name} · {activity.title}</p>
      <p>Current logins for this activity will end. Participant usernames, assigned points and planting records will be kept.</p>
    </Modal>, document.body)}
    <FormMessage error={error} loading={busy ? 'Checking planting access…' : ''} success={success} />
  </div>;
}
