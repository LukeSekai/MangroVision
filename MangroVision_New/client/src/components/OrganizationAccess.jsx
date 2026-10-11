import { useState } from 'react';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary, FormMessage } from './FormFeedback';
import { formRequest } from '../utils/formValidation';

const API = import.meta.env.VITE_API_BASE || '';

export default function OrganizationAccess({ organization, expectedParticipants = 1, onCreated }) {
  const [count, setCount] = useState(expectedParticipants || 1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [codes, setCodes] = useState(null);
  const [confirmation, setConfirmation] = useState(null);
  const feedback = useFormFeedback({ participant_count: { label: 'Participants', serverTerms: ['participant count'] } });
  const pending = organization.registration_pending;
  const request = async (url, body, fallback) => {
    if (busy) return;
    setBusy(true); setError(''); setSuccess('');
    try {
      const result = await formRequest(`${API}${url}`, body ? {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      } : {}, fallback);
      setCodes(result.codes || []);
      return result;
    } catch (failure) {
      if (!feedback.fromServer(failure)) setError(failure.uncertain
        ? 'The result could not be verified. Load the current codes before trying again.' : failure.message);
      if (failure.uncertain) { setConfirmation(null); onCreated?.(); }
    } finally { setBusy(false); }
  };
  const create = async (event) => {
    event.preventDefault();
    if (busy || !feedback.validate()) return;
    const result = await request('/api/planters/', { organization_id: organization.organization_id,
      full_name: organization.organization_name, participant_count: Number(count) }, 'Could not create planter access.');
    if (result) { setSuccess('Planter codes created. Give each participant only their own code.'); onCreated?.(); }
  };
  const issue = async () => {
    const result = await request(`/api/planters/${organization.id}/access-codes`, {}, 'Could not issue planter codes.');
    if (result) setSuccess('Codes are ready. Existing codes, devices and assigned points are preserved.');
  };
  const changeCode = async () => {
    const { slot, action } = confirmation;
    const result = await request(`/api/planters/${organization.id}/participants/${slot}/access-code`, { action }, 'Could not change this code.');
    if (result) {
      setConfirmation(null);
      setSuccess(action === 'revoke' ? `Participant ${slot} can no longer sign in. Their records remain saved.`
        : `Participant ${slot} has a new code. Give it to them; their previous code no longer works.`);
    }
  };
  const handover = () => `${organization.organization_name} ? Individual planter codes\nOpen: ${window.location.origin}/field\n\n`
    + codes.map((entry) => `Participant ${entry.participant_slot}: ${entry.code || 'Revoked'}`).join('\n')
    + '\n\nGive each participant only their own code. Keep this full list with the group leader.\nCodes work across activities and browsers. Your code restores the same participant number and assigned points.\n';
  const download = () => {
    const url = URL.createObjectURL(new Blob([handover()], { type: 'text/plain;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'planter-codes-private.txt'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(handover()); setSuccess('Code list copied. Share it privately with the group leader.'); }
    catch { setError('Could not copy the codes. Use Download instead.'); }
  };
  return <div className="assign-card">
    <p className="text-sm">Each planter uses one private code across this organization?s activities. Their participant number and assigned points stay the same.</p>
    {pending && !codes && <form noValidate onSubmit={create} onChangeCapture={feedback.onChange}>
      <FormErrorSummary feedback={feedback} />
      <div className="form-group"><label className="form-label" htmlFor="group-account-participants">Participants</label>
        <input {...feedback.props('participant_count')} id="group-account-participants" className="form-input" type="number" required min="1" max="10000" step="1"
          value={count} onChange={(event) => setCount(event.target.value)} disabled={busy} />
        <FieldError feedback={feedback} field="participant_count" />
        <span className="text-sm">One code per participant. Reserved points are divided when access is first created.</span></div>
      <button type="submit" className="btn btn-primary btn-sm" disabled={busy}>{busy ? 'Creating codes?' : 'Create planter codes'}</button>
    </form>}
    {!pending && codes === null && <button type="button" className="btn btn-secondary btn-sm" disabled={busy}
      onClick={() => request(`/api/planters/${organization.id}/access-codes`, null, 'Could not load planter codes.')}>
      {busy ? 'Loading codes?' : 'Show planter codes'}</button>}
    {codes !== null && <>
      {codes.length < Number(organization.participant_count || count) && !pending && <button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={issue}>Issue missing planter codes</button>}
      {codes.length > 0 && <>
        <p className="text-sm">Keep this full list with the group leader. Give each planter only the code for their participant number.</p>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
          <button type="button" className="btn btn-secondary btn-sm" onClick={copy}>Copy list</button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={download}>Download</button>
          {!pending && <button type="button" className="btn btn-secondary btn-sm" disabled={busy}
            onClick={() => request(`/api/planters/${organization.id}/access-codes`, null, 'Could not refresh planter codes.')}>Refresh codes</button>}
        </div>
        <ul style={{ listStyle: 'none', padding: 0, maxHeight: 320, overflowY: 'auto' }}>
          {codes.map((entry) => <li key={entry.participant_slot} style={{ borderTop: '1px solid #dce5df', padding: '10px 0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
              <span className="text-sm">Participant {entry.participant_slot}</span>
              <strong style={{ fontFamily: 'monospace', fontSize: 15 }}>{entry.code || 'Revoked'}</strong>
            </div>
            {!pending && <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
              <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => setConfirmation({ slot: entry.participant_slot, action: 'regenerate' })}>{entry.revoked ? 'Restore with new code' : 'Replace code'}</button>
              {!entry.revoked && <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => setConfirmation({ slot: entry.participant_slot, action: 'revoke' })}>Revoke</button>}
            </div>}
          </li>)}
        </ul>
      </>}
    </>}
    {confirmation && <div role="alert" style={{ padding: 12, border: '1px solid #dce5df', borderRadius: 8 }}>
      <p className="text-sm">{confirmation.action === 'revoke' ? 'Revoke access' : 'Replace the code'} for participant {confirmation.slot}? Their current sign-in will end. Their device identity and planting records will stay saved.</p>
      <div style={{ display: 'flex', gap: 8 }}><button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={changeCode}>Confirm</button>
        <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => setConfirmation(null)}>Cancel</button></div>
    </div>}
    <FormMessage error={error} loading={busy ? 'Updating planter access?' : ''} success={success} />
  </div>;
}
