import { useEffect, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import Modal from './Modal';
import { appointmentTypeLabel, isPlantingAppointment } from '../utils/appointmentTypes';
import './WebsiteRequests.css';

const API = import.meta.env.VITE_API_BASE || '';
const manilaParts = (value) => {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Manila', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(new Date(value));
  const p = Object.fromEntries(parts.map(({ type, value: part }) => [type, part]));
  return { date: `${p.year}-${p.month}-${p.day}`, time: `${p.hour}:${p.minute}` };
};
const formatTime = (value) => new Intl.DateTimeFormat('en-PH', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Manila' }).format(new Date(value));
const isFutureTime = (value) => Date.parse(value) > Date.now();

async function requestJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not process the website request. Please check the details and try again.');
  return data;
}

function RequestReview({ request, organizations, schedules, assessmentFor, renderAdvice, onClose, onSaved }) {
  const start = manilaParts(request.start_at);
  const end = manilaParts(request.end_at);
  const [form, setForm] = useState({
    organization: request.organization, date: start.date, startTime: start.time, endTime: end.time,
    contacted: false, action: 'confirmed', decisionNotes: '', acceptCaution: false, email: request.email || '',
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const inFlight = useRef(false);
  const formRoot = useRef(null);
  useEffect(() => {
    formRoot.current?.closest('.modal-card')?.scrollTo(0, 0);
    formRoot.current?.querySelector('select')?.focus({ preventScroll: true });
  }, []);
  const update = (event) => setForm((current) => ({ ...current, [event.target.name]: event.target.type === 'checkbox' ? event.target.checked : event.target.value, ...(event.target.name === 'date' || event.target.name === 'startTime' || event.target.name === 'endTime' ? { acceptCaution: false } : {}) }));
  const window = { start_at: `${form.date}T${form.startTime}:00+08:00`, end_at: `${form.date}T${form.endTime}:00+08:00` };
  const planting = isPlantingAppointment(request);
  const advice = planting ? assessmentFor(window) : null;
  const conflicts = schedules.filter((schedule) => !['cancelled', 'completed'].includes(schedule.status)
    && Date.parse(schedule.start_at) < Date.parse(window.end_at) && Date.parse(schedule.end_at) > Date.parse(window.start_at));
  const needsCaution = advice?.status === 'unsafe' || conflicts.length > 0;

  const save = async (event) => {
    event?.preventDefault?.();
    if (inFlight.current) return;
    setError('');
    if (!formRoot.current?.reportValidity()) return;
    if (form.action === 'confirmed') {
      if (!form.organization.trim() || !form.date || !form.startTime || !form.endTime || form.endTime <= form.startTime) {
        setError('Enter the organization and a valid agreed date and time window.'); return;
      }
      if (!isFutureTime(window.start_at)) { setError('Choose an agreed date and time in the future.'); return; }
      if (!form.contacted) { setError('Contact the requester and agree on the date and time before confirming.'); return; }
      if (needsCaution && !form.acceptCaution) { setError('Review and acknowledge the schedule warnings before confirming.'); return; }
    } else if (!form.decisionNotes.trim()) { setError('Enter a reason for declining or cancelling this request.'); return; }
    const organization = organizations.find((item) => item.name.toLowerCase() === form.organization.trim().toLowerCase());
    inFlight.current = true;
    setBusy(true);
    try {
      await requestJson(`/api/like-appointments/${request.id}/review`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
          action: form.action, contacted: form.contacted, decision_notes: form.decisionNotes.trim() || null,
          ...(form.action === 'confirmed' ? { email: form.email.trim() } : {}),
          ...(form.action === 'confirmed' ? { ...window, organization: organization?.name || form.organization.trim(), organization_id: organization?.id || null } : {}),
        }),
      });
      onSaved(form.action === 'confirmed' ? 'Appointment confirmed and recorded in Scheduling. The confirmation email is queued for delivery.' : `Website request ${form.action}.`);
    } catch (issue) { setError(issue.message); }
    finally { inFlight.current = false; setBusy(false); }
  };

  return <Modal open title={`Review website request · ${request.reference}`} variant="info" className="modal-card-wide" busy={busy}
    confirmLabel={form.action === 'confirmed' ? 'Confirm and create schedule' : form.action === 'declined' ? 'Decline request' : 'Cancel request'} onConfirm={save} onCancel={onClose}>
    <form className="website-review" ref={formRoot} onSubmit={save}>
      <dl className="website-request-details"><div><dt>Contact person</dt><dd>{request.contact_name}</dd></div><div><dt>Phone</dt><dd><a href={`tel:${request.phone.replace(/[^+\d]/g, '')}`}>{request.phone}</a></dd></div>{request.email ? <div><dt>Email</dt><dd><a href={`mailto:${request.email}`}>{request.email}</a></dd></div> : null}<div><dt>Appointment type</dt><dd>{appointmentTypeLabel(request.appointment_type)}</dd></div><div><dt>Activity</dt><dd>{request.title}</dd></div><div><dt>Participants</dt><dd>{request.participants}</dd></div><div><dt>Requested time</dt><dd>{formatTime(request.start_at)} – {manilaParts(request.end_at).time} (Philippine time)</dd></div></dl>
      {request.notes ? <p className="website-request-notes">{request.notes}</p> : null}
      {error ? <p className="website-request-error" role="alert">{error}</p> : null}
      <label>Decision<select name="action" value={form.action} onChange={update}><option value="confirmed">Confirm appointment</option><option value="declined">Decline request</option><option value="cancelled">Cancel request</option></select></label>
      {form.action === 'confirmed' ? <>
        <label>Organization<input name="organization" value={form.organization} onChange={update} list="website-request-organizations" maxLength={200} required /><datalist id="website-request-organizations">{organizations.map((item) => <option key={item.id} value={item.name} />)}</datalist><small>Select a registered organization or enter the agreed name to register it on approval.</small></label>
        <label>Confirmation email<input name="email" type="email" value={form.email} onChange={update} required maxLength={254} /><small>Confirm this address with the requester. The agreed schedule{planting ? ' and planter access details' : ''} will be sent here.</small></label>
        <div className="website-review-times"><label>Agreed date<input name="date" type="date" value={form.date} onChange={update} required /></label><label>Start time<input name="startTime" type="time" value={form.startTime} onChange={update} required /></label><label>End time<input name="endTime" type="time" value={form.endTime} onChange={update} required /></label></div>
        <div className="website-request-advice">{planting ? <>{renderAdvice(advice)}<p>{advice.reason}</p></> : <p>Check staff availability and site conditions for this {appointmentTypeLabel(request.appointment_type).toLowerCase()}.</p>}{conflicts.length ? <p><strong>{conflicts.length} existing {conflicts.length === 1 ? 'schedule overlaps' : 'schedules overlap'} this time.</strong> Check staff and site availability.</p> : null}</div>
        {needsCaution ? <label className="website-review-check"><input type="checkbox" name="acceptCaution" checked={form.acceptCaution} onChange={update} /><span>I reviewed the {planting ? 'tide and ' : ''}schedule warnings and assessed whether this activity can proceed.</span></label> : null}
        <label className="website-review-check"><input type="checkbox" name="contacted" checked={form.contacted} onChange={update} /><span>I contacted the requester and they agreed to this date and time.</span></label>
      </> : null}
      <label>{form.action === 'confirmed' ? 'Coordination notes (optional)' : 'Reason (required)'}<textarea name="decisionNotes" value={form.decisionNotes} onChange={update} maxLength={2000} rows={3} required={form.action !== 'confirmed'} /></label>
    </form>
  </Modal>;
}

export default function WebsiteRequests({ token, reloadKey, organizations, schedules, assessmentFor, renderAdvice, onRequestsChange, reviewRequest, onReview, onClose, onChanged }) {
  const location = useLocation();
  const [requests, setRequests] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState('');
  const [filter, setFilter] = useState('pending');
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    if (new URLSearchParams(location.search).get('requests') === 'pending') queueMicrotask(() => setFilter('pending'));
  }, [location.key, location.search]);

  useEffect(() => {
    if (!token) return;
    const controller = new AbortController();
    const load = () => requestJson('/api/like-appointments', { signal: controller.signal })
      .then((data) => {
        if (controller.signal.aborted) return;
        const rows = Array.isArray(data.requests) ? data.requests : [];
        setRequests(rows); onRequestsChange(rows); setError(''); setLoading(false);
      }).catch((issue) => {
        if (issue.name !== 'AbortError') { setError(issue.message); setLoading(false); }
      });
    load();
    const timer = globalThis.setInterval(load, 60000);
    return () => { controller.abort(); globalThis.clearInterval(timer); };
  }, [token, reloadKey, refreshKey, onRequestsChange]);
  const pending = requests.filter((request) => request.status === 'pending');
  const visible = filter === 'pending' ? pending : requests;
  const saved = (message) => { setNotice(message); onClose(); setRefreshKey((key) => key + 1); onChanged(); window.dispatchEvent(new Event('mv:data-changed')); };

  return <section className="schedule-panel website-requests" id="website-requests" tabIndex={-1} aria-labelledby="website-requests-title">
    <div className="schedule-panel-head schedule-toolbar"><div><span className="schedule-step">LIKE public website</span><h2 id="website-requests-title">Website requests <span className="website-request-count">{pending.length} pending</span></h2><p>Review the requested time, contact the organization, then confirm. Manual scheduling remains available above.</p></div><a className="website-preview-link" href="/landing-page" target="_blank" rel="noreferrer">Open LIKE website ↗</a></div>
    <div className="website-request-toolbar"><div className="schedule-view-toggle"><button type="button" className={filter === 'pending' ? 'is-active' : ''} onClick={() => setFilter('pending')}>Pending requests</button><button type="button" className={filter === 'all' ? 'is-active' : ''} onClick={() => setFilter('all')}>Recent requests</button></div><button type="button" className="schedule-refresh" onClick={() => setRefreshKey((key) => key + 1)}>Refresh requests</button></div>
    {error ? <p className="website-request-error" role="alert">{error}</p> : null}
    {notice ? <p className="website-request-notice" role="status">{notice}</p> : null}
    {loading ? <p className="website-request-empty" role="status">Loading website requests…</p> : visible.length ? <div className="schedule-table-wrap"><table><caption className="sr-only">LIKE website appointment requests</caption><thead><tr><th>Reference and organization</th><th>Preferred time</th><th>Contact person</th><th>Status</th><th>Action</th></tr></thead><tbody>{visible.map((request) => <tr key={request.id}><td><strong>{request.organization}</strong><small>{request.reference}</small><small>{appointmentTypeLabel(request.appointment_type)} · {request.participants} participants</small></td><td>{formatTime(request.start_at)}<small>Until {manilaParts(request.end_at).time} · Philippine time</small></td><td>{request.contact_name}<small>{request.phone}</small></td><td><span className={`schedule-status is-${request.status === 'pending' ? 'requested' : request.status}`}>{request.status}</span>{request.schedule_id ? <small>Linked schedule #{request.schedule_id}{request.schedule_status ? ` · ${request.schedule_status}` : ''}</small> : null}{request.status === 'confirmed' && !request.schedule_id ? <small>Linked schedule was removed.</small> : null}{request.email_status ? <small>{request.email_error ? 'Email awaiting retry' : `${request.email_kind === 'cancellation' ? 'Cancellation' : request.email_kind === 'update' ? 'Update' : 'Confirmation'} email ${request.email_status === 'sent' ? 'sent' : request.email_status === 'sending' ? 'sending' : 'queued'}`}</small> : null}{request.decision_notes ? <small>{request.decision_notes}</small> : null}</td><td>{request.status === 'pending' ? <button className="website-review-button" type="button" onClick={() => onReview(request)}>Review request</button> : <small>Reviewed {request.reviewed_at ? formatTime(request.reviewed_at) : ''}</small>}</td></tr>)}</tbody></table></div> : <p className="website-request-empty">{filter === 'pending' ? 'No pending website requests. New requests will appear here and on the calendar.' : 'No website requests yet.'}</p>}
    {reviewRequest ? <RequestReview key={reviewRequest.id} request={reviewRequest} organizations={organizations} schedules={schedules} assessmentFor={assessmentFor} renderAdvice={renderAdvice} onClose={onClose} onSaved={saved} /> : null}
  </section>;
}
