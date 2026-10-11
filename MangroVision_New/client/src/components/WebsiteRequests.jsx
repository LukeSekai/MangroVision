import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useLocation } from 'react-router-dom';
import Modal from './Modal';
import useFormFeedback from '../utils/useFormFeedback';
import { formRequest } from '../utils/formValidation';
import { FieldError, FormErrorSummary, FormMessage } from './FormFeedback';
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
const formatEndTime = (value) => new Intl.DateTimeFormat('en-PH', { timeStyle: 'short', timeZone: 'Asia/Manila' }).format(new Date(value));
const isFutureTime = (value) => Date.parse(value) > Date.now();

async function requestJson(path, options = {}) {
  return formRequest(`${API}${path}`, options, 'Could not process the website request. Please check the details and try again.');
}

export function RequestReview({ request, organizations, schedules, assessmentFor, renderAdvice, onClose, onSaved }) {
  const start = manilaParts(request.start_at);
  const end = manilaParts(request.end_at);
  const [form, setForm] = useState({
    date: start.date, startTime: start.time, endTime: end.time,
    contacted: false, action: 'confirmed', decisionNotes: '', acceptCaution: false, email: request.email || '',
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const inFlight = useRef(false);
  const formRoot = useRef(null);
  const feedback = useFormFeedback({
    email: { label: 'Confirmation email', serverTerms: ['email'] },
    date: { label: 'New date' }, startTime: { label: 'Start time', aliases: ['start_at'], serverTerms: ['future'] },
    endTime: { label: 'End time', aliases: ['end_at'] }, contacted: { label: 'Requester agreement' },
    acceptCaution: { label: 'Schedule warnings' }, decisionNotes: { label: 'Reason', aliases: ['decision_notes'] },
  });
  useEffect(() => {
    formRoot.current?.closest('.modal-content')?.scrollTo(0, 0);
    formRoot.current?.querySelector('select')?.focus({ preventScroll: true });
  }, []);
  const update = (event) => {
    const { name, type, checked, value } = event.target;
    if (name === 'action') { feedback.clear(); setError(''); }
    setForm((current) => ({ ...current, [name]: type === 'checkbox' ? checked : value,
      ...(['action', 'date', 'startTime', 'endTime'].includes(name) ? { acceptCaution: false, contacted: false } : {}),
    }));
  };
  const rescheduling = form.action === 'rescheduled';
  const confirming = form.action !== 'declined';
  const window = rescheduling
    ? { start_at: `${form.date}T${form.startTime}:00+08:00`, end_at: `${form.date}T${form.endTime}:00+08:00` }
    : { start_at: request.start_at, end_at: request.end_at };
  const planting = isPlantingAppointment(request);
  const advice = planting ? assessmentFor(window) : null;
  const conflicts = schedules.filter((schedule) => !['cancelled', 'completed'].includes(schedule.status)
    && Date.parse(schedule.start_at) < Date.parse(window.end_at) && Date.parse(schedule.end_at) > Date.parse(window.start_at));
  const needsCaution = advice?.status === 'unsafe' || conflicts.length > 0;

  const save = async (event) => {
    event?.preventDefault?.();
    if (inFlight.current) return;
    setError('');
    if (!feedback.validate()) return;
    if (confirming) {
      if (rescheduling && (!form.date || !form.startTime || !form.endTime || form.endTime <= form.startTime)) {
        feedback.reject({ endTime: 'Choose an end time later than the start time on the same day.' }); return;
      }
      if (!isFutureTime(window.start_at)) {
        if (rescheduling) feedback.reject({ startTime: 'Choose a new date and time in the future.' });
        else setError('The requested time has passed. Choose “Reschedule and confirm” to agree on a new time.');
        return;
      }
      if (!form.contacted) { feedback.reject({ contacted: 'Contact the requester and agree on the date and time before confirming.' }); return; }
      if (needsCaution && !form.acceptCaution) { feedback.reject({ acceptCaution: 'Review and acknowledge the schedule warnings before confirming.' }); return; }
    } else if (!form.decisionNotes.trim()) { feedback.reject({ decisionNotes: 'Enter a reason for declining this request.' }); return; }
    const organization = organizations.find((item) => item.name.toLowerCase() === request.organization.trim().toLowerCase());
    inFlight.current = true;
    setBusy(true);
    try {
      await requestJson(`/api/like-appointments/${request.id}/review`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
          action: confirming ? 'confirmed' : 'declined', contacted: form.contacted, decision_notes: form.decisionNotes.trim() || null,
          ...(confirming && !request.email ? { email: form.email.trim() } : {}),
          ...(confirming ? { organization: organization?.name || request.organization.trim(), organization_id: organization?.id || null } : {}),
          ...(rescheduling ? window : {}),
        }),
      });
      onSaved(confirming ? 'Appointment confirmed and recorded in Scheduling. The confirmation email is queued for delivery.' : 'Website request declined.');
    } catch (issue) { if (!feedback.fromServer(issue)) setError(issue.message); }
    finally { inFlight.current = false; setBusy(false); }
  };

  return createPortal(<Modal open title="Review website request" variant="info" className="modal-card-wide website-review-dialog" busy={busy}
    cancelLabel="Close" confirmLabel={rescheduling ? 'Confirm new schedule' : confirming ? 'Confirm appointment' : 'Decline request'} onConfirm={save} onCancel={onClose}>
    <form className="website-review" noValidate onChangeCapture={feedback.onChange} aria-busy={busy} ref={formRoot} onSubmit={save}>
      <p className="website-review-reference">{request.reference}</p>
      <dl className="website-request-details">
        <div><dt>Organization</dt><dd>{request.organization}</dd></div>
        <div><dt>Contact person</dt><dd>{request.contact_name}</dd></div>
        <div><dt>Phone</dt><dd><a href={`tel:${request.phone.replace(/[^+\d]/g, '')}`}>{request.phone}</a></dd></div>
        <div><dt>Email</dt><dd>{request.email ? <a href={`mailto:${request.email}`}>{request.email}</a> : 'Not supplied'}</dd></div>
        <div><dt>Appointment type</dt><dd>{appointmentTypeLabel(request.appointment_type)}</dd></div>
        <div><dt>Participants</dt><dd>{request.participants}</dd></div>
        <div className="website-request-detail-wide"><dt>Activity</dt><dd>{request.title}</dd></div>
        <div className="website-request-detail-wide"><dt>Requested schedule</dt><dd>{formatTime(request.start_at)} – {formatEndTime(request.end_at)} <small>(Philippine time)</small></dd></div>
      </dl>
      {request.notes ? <div className="website-request-notes"><strong>Requester notes</strong><p>{request.notes}</p></div> : null}
      <FormErrorSummary feedback={feedback} />
      <FormMessage error={error} loading={busy ? 'Saving the decision…' : ''} onRetry={save} busy={busy} />
      <label>Decision<select name="action" disabled={busy} value={form.action} onChange={update}><option value="confirmed">Confirm requested schedule</option><option value="rescheduled">Reschedule and confirm</option><option value="declined">Decline request</option></select></label>
      {confirming ? <>
        {rescheduling ? <fieldset className="website-review-reschedule" disabled={busy}>
          <legend>New schedule</legend>
          <p>Agree on a new time with the requester before confirming.</p>
          <div className="website-review-times"><label>New date<input {...feedback.props('date')} type="date" value={form.date} onChange={update} required /><FieldError feedback={feedback} field="date" /></label><label>Start time<input {...feedback.props('startTime')} type="time" value={form.startTime} onChange={update} required /><FieldError feedback={feedback} field="startTime" /></label><label>End time<input {...feedback.props('endTime')} type="time" value={form.endTime} onChange={update} required /><FieldError feedback={feedback} field="endTime" /></label></div>
        </fieldset> : <p className="website-review-hint">Confirmation keeps the requested date and time.</p>}
        {!request.email ? <label>Confirmation email<input disabled={busy} {...feedback.props('email')} type="email" value={form.email} onChange={update} required maxLength={254} /><small>This older request has no email address. Obtain it from the requester to send their confirmation{planting ? ' and planter codes' : ''}.</small><FieldError feedback={feedback} field="email" /></label> : null}
        <div className="website-request-advice">{planting ? <>{renderAdvice(advice)}<p>{advice.reason}</p></> : <p>Check staff availability and site conditions for this {appointmentTypeLabel(request.appointment_type).toLowerCase()}.</p>}{conflicts.length ? <p><strong>{conflicts.length} existing {conflicts.length === 1 ? 'schedule overlaps' : 'schedules overlap'} this time.</strong> Check staff and site availability.</p> : null}</div>
        {needsCaution ? <label className="website-review-check"><input disabled={busy} type="checkbox" {...feedback.props('acceptCaution')} checked={form.acceptCaution} onChange={update} /><span>I reviewed the {planting ? 'tide and ' : ''}schedule warnings and assessed whether this activity can proceed.<FieldError feedback={feedback} field="acceptCaution" /></span></label> : null}
        <label className="website-review-check"><input disabled={busy} type="checkbox" {...feedback.props('contacted')} checked={form.contacted} onChange={update} /><span>I contacted the requester and they agreed to this date and time.<FieldError feedback={feedback} field="contacted" /></span></label>
      </> : null}
      <label>{confirming ? 'Review notes (optional)' : 'Reason (required)'}<textarea disabled={busy} {...feedback.props('decisionNotes')} value={form.decisionNotes} onChange={update} maxLength={2000} rows={2} required={!confirming} /><FieldError feedback={feedback} field="decisionNotes" /></label>
    </form>
  </Modal>, document.body);
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
    <FormMessage error={error} success={notice} onRetry={() => setRefreshKey((key) => key + 1)} busy={loading} />
    {loading ? <p className="website-request-empty" role="status">Loading website requests…</p> : visible.length ? <div className="schedule-table-wrap"><table><caption className="sr-only">LIKE website appointment requests</caption><thead><tr><th>Reference and organization</th><th>Preferred time</th><th>Contact person</th><th>Status</th><th>Action</th></tr></thead><tbody>{visible.map((request) => <tr key={request.id}><td><strong>{request.organization}</strong><small>{request.reference}</small><small>{appointmentTypeLabel(request.appointment_type)} · {request.participants} participants</small></td><td>{formatTime(request.start_at)}<small>Until {manilaParts(request.end_at).time} · Philippine time</small></td><td>{request.contact_name}<small>{request.phone}</small></td><td><span className={`schedule-status is-${request.status === 'pending' ? 'requested' : request.status}`}>{request.status}</span>{request.schedule_id ? <small>Linked schedule #{request.schedule_id}{request.schedule_status ? ` · ${request.schedule_status}` : ''}</small> : null}{request.status === 'confirmed' && !request.schedule_id ? <small>Linked schedule was removed.</small> : null}{request.email_status ? <small>{request.email_error ? 'Email awaiting retry' : `${request.email_kind === 'cancellation' ? 'Cancellation' : request.email_kind === 'update' ? 'Update' : 'Confirmation'} email ${request.email_status === 'sent' ? 'sent' : request.email_status === 'sending' ? 'sending' : 'queued'}`}</small> : null}{request.decision_notes ? <small>{request.decision_notes}</small> : null}</td><td>{request.status === 'pending' ? <button className="website-review-button" type="button" onClick={() => onReview(request)}>Review request</button> : <small>Reviewed {request.reviewed_at ? formatTime(request.reviewed_at) : ''}</small>}</td></tr>)}</tbody></table></div> : <p className="website-request-empty">{filter === 'pending' ? 'No pending website requests. New requests will appear here and on the calendar.' : 'No website requests yet.'}</p>}
    {reviewRequest ? <RequestReview key={reviewRequest.id} request={reviewRequest} organizations={organizations} schedules={schedules} assessmentFor={assessmentFor} renderAdvice={renderAdvice} onClose={onClose} onSaved={saved} /> : null}
  </section>;
}
