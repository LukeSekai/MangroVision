import { useEffect, useRef, useState } from 'react';
import MapLink from '../components/MapLink';
import { scheduleContactFields } from '../utils/scheduleContact';
import { appointmentTypeLabel } from '../utils/appointmentTypes';
import { scheduleProgressStatus, scheduleTrackingState } from '../utils/scheduleTracking';
import './Organizations.css';

const API = import.meta.env.VITE_API_BASE || '';
const ORGANIZATIONS_PER_PAGE = 4;
const number = (value) => Number(value || 0).toLocaleString();
const dateTime = (value) => {
  if (!value) return 'Not recorded';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Not recorded' : date.toLocaleString('en-PH', {
    timeZone: 'Asia/Manila', month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit',
  });
};

function contactInfo(organization) {
  const parsed = scheduleContactFields(organization);
  return {
    name: parsed.contactName || organization.request_contact_name || '',
    email: parsed.email || organization.request_email || '',
    phone: parsed.phone || organization.request_phone || organization.account_phone || '',
  };
}

async function read(path, signal) {
  const response = await fetch(`${API}${path}`, { signal, cache: 'no-store' });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || 'Could not load organization information. Please retry.');
  return data;
}

function PeopleIcon() {
  return <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><circle cx="9" cy="8" r="3" /><path d="M3 21v-2a6 6 0 0 1 12 0v2M16 5a3 3 0 0 1 0 6M21 21v-2a6 6 0 0 0-4-5.66" /></svg>;
}

function ActivityCode({ organizationId, activity }) {
  const [access, setAccess] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const request = useRef(null);
  useEffect(() => {
    let active = true;
    queueMicrotask(() => { if (active) { setBusy(false); setAccess(null); setCopied(false); } });
    return () => { active = false; request.current?.abort(); };
  }, []);
  const show = async () => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy(true); setError(''); setCopied(false);
    try {
      const data = await read(`/api/planters/organizations/${organizationId}/activities/${activity.id}/access`, controller.signal);
      if (!controller.signal.aborted) {
        setAccess(data.access || null);
        if (!data.access?.code) setError(data.access?.revoked ? 'This code has been revoked.' : 'No organization code has been issued for this activity.');
      }
    } catch (issue) {
      if (!controller.signal.aborted) setError(issue.message);
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(access.code); setCopied(true); setError(''); }
    catch { setError('Could not copy automatically. Select the code and copy it.'); }
  };
  if (activity.appointment_type !== 'tree_planting') return <span className="organization-muted">Not a planting activity</span>;
  if (!activity.has_code || activity.code_revoked) return <span className="organization-muted">{activity.code_revoked ? 'Code revoked' : 'Code not issued'}</span>;
  return <div className="organization-code">
    {access?.code ? <>
      <code>{access.code}</code>
      <button type="button" className="btn btn-secondary btn-sm" onClick={copy}>{copied ? 'Copied' : 'Copy code'}</button>
      <button type="button" className="organization-text-button" onClick={() => { setAccess(null); setCopied(false); setError(''); }}>Hide code</button>
    </> : <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={show}>{busy ? 'Loading code…' : 'Show code'}</button>}
    {error && <p role="alert" className="organization-error">{error}</p>}
  </div>;
}

function Members({ members }) {
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(0);
  const filtered = members.filter((member) => `${member.username || ''} ${member.participant_number} ${member.account_name || ''}`.toLowerCase().includes(query.trim().toLowerCase()));
  const pages = Math.max(1, Math.ceil(filtered.length / 25));
  const current = Math.min(page, pages - 1);
  return <>
    <p className="organization-muted">Members who have joined or have a previously registered participant profile. Unused participant places are not listed.</p>
    <label className="organization-search">Find a member<input type="search" placeholder="Search username or participant number" value={query} onChange={(event) => { setQuery(event.target.value); setPage(0); }} /></label>
    {filtered.length ? <div className="organization-table-wrap"><table className="organization-table">
      <thead><tr><th scope="col">Username</th><th scope="col">Participant</th><th scope="col">First joined</th><th scope="col">Last seen</th></tr></thead>
      <tbody>{filtered.slice(current * 25, (current + 1) * 25).map((member) => <tr key={`${member.planter_id}:${member.participant_number}`}>
        <td><strong>{member.username || 'Name not recorded'}</strong>{!member.username && <small>Existing participant profile</small>}</td>
        <td>#{member.participant_number}<small>{member.account_name}</small></td><td>{dateTime(member.first_joined_at)}</td><td>{dateTime(member.last_seen_at)}</td>
      </tr>)}</tbody>
    </table></div> : <p className="organization-empty">{members.length ? 'No members match your search.' : 'No members have joined yet.'}</p>}
    {filtered.length > 25 && <nav className="organization-pagination" aria-label="Member pages"><button className="btn btn-secondary btn-sm" type="button" disabled={!current} onClick={() => setPage(current - 1)}>Previous</button><span>Page {current + 1} of {pages}</span><button className="btn btn-secondary btn-sm" type="button" disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>Next</button></nav>}
  </>;
}

function OrganizationDetail({ id, revision }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const [tab, setTab] = useState('sites');
  useEffect(() => {
    const controller = new AbortController();
    read(`/api/organizations/${id}`, controller.signal).then((data) => {
      if (!controller.signal.aborted) { setDetail(data); setError(''); }
    }).catch((issue) => { if (!controller.signal.aborted) setError(issue.message); });
    return () => controller.abort();
  }, [id, revision, retry]);
  if (error) return <section className="organization-detail organization-empty"><p role="alert">{error}</p><button className="btn btn-secondary" type="button" onClick={() => { setError(''); setRetry(retry + 1); }}>Retry details</button></section>;
  if (!detail) return <section className="organization-detail organization-empty" role="status">Loading organization details…</section>;
  const organization = detail.organization;
  const contact = contactInfo(organization);
  const tabs = [{ key: 'sites', label: 'Project sites', count: detail.project_sites.length }, { key: 'codes', label: 'Activity codes', count: detail.activities.length }, { key: 'members', label: 'Members', count: detail.members.length }];
  return <section className="organization-detail" aria-label={`${organization.name} details`}>
    <header className="organization-detail-heading"><span className="organization-avatar"><PeopleIcon /></span><div><span className="organization-eyebrow">Organization profile</span><h2>{organization.name}</h2></div></header>
    <div className="organization-profile-body">
    <dl className="organization-contact-grid" data-guide="organization-contact">
      <div><dt>Contact person</dt><dd>{contact.name || 'Not recorded'}</dd></div>
      <div><dt>Email</dt><dd>{contact.email ? <a href={`mailto:${contact.email}`}>{contact.email}</a> : 'Not recorded'}</dd></div>
      <div><dt>Phone number</dt><dd>{contact.phone ? <a href={`tel:${contact.phone.replace(/[^+\d]/g, '')}`}>{contact.phone}</a> : 'Not recorded'}</dd></div>
      <div><dt>Expected participants</dt><dd>{organization.scheduled_participants == null ? 'Not recorded' : number(organization.scheduled_participants)}</dd></div>
      <div><dt>Registered members</dt><dd>{number(organization.member_count)}</dd></div>
      <div><dt>Days between plant checks</dt><dd>Every {organization.inspection_interval_days} days</dd></div>
    </dl>
    <p className="organization-contact-source organization-muted">{organization.contact_activity_title ? `Latest saved activity: ${organization.contact_activity_title}. Expected participants refers to this activity.` : 'No activity contact information has been recorded yet.'}</p>
    </div>
    <div className="organization-tabs" role="tablist" aria-label="Organization information">
      {tabs.map((item, index) => <button key={item.key} type="button" role="tab" id={`organization-tab-${item.key}`} aria-selected={tab === item.key} tabIndex={tab === item.key ? 0 : -1} aria-controls={`organization-panel-${item.key}`} onClick={() => setTab(item.key)} onKeyDown={(event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
        setTab(tabs[next].key);
        event.currentTarget.parentElement.children[next].focus();
      }}>{item.label}<span>{number(item.count)}</span></button>)}
    </div>
    <div className="organization-tab-content" role="tabpanel" id={`organization-panel-${tab}`} aria-labelledby={`organization-tab-${tab}`}>
      {tab === 'sites' && (detail.project_sites.length ? <div className="organization-sites">{detail.project_sites.map((site) => <article key={site.id}>
        <div><h3>{site.name}</h3><p>{site.notes || 'No site notes recorded.'}</p><span className="organization-muted">{site.deleted_at ? 'Archived project site' : `Plant checks every ${site.inspection_interval_days || organization.inspection_interval_days} days`}</span></div>
        {!site.deleted_at && <MapLink className="btn btn-secondary btn-sm" to={`/map?project_site_id=${site.id}&focus=site_points`}>View on map</MapLink>}
      </article>)}</div> : <p className="organization-empty">This organization has no project sites yet.</p>)}
      {tab === 'codes' && <>
        <p className="organization-muted">One organization code is shared by the participants in each planting activity. Access follows the approved activity’s start and finish times. All times are Philippine time.</p>
        {detail.activities.length ? <div className="organization-activities">{detail.activities.map((activity) => {
          const tracking = scheduleTrackingState(activity);
          return <article key={activity.id}><div className="organization-activity-heading"><div><h3>{activity.title}</h3><p>{appointmentTypeLabel(activity.appointment_type)} · {number(activity.expected_planters)} participants</p></div><div className="organization-status"><span className={`is-${scheduleProgressStatus(activity.status)}`}>{scheduleProgressStatus(activity.status)}</span>{tracking.approved && tracking.pending && <span className="is-approved">Approved</span>}</div></div>
            <p>{dateTime(activity.start_at)} – {dateTime(activity.end_at)}</p><p className="organization-muted">{activity.project_site_name || 'No project site selected'}</p>
            <ActivityCode key={`${activity.id}:${revision}`} organizationId={id} activity={activity} />
          </article>;
        })}</div> : <p className="organization-empty">No activities have been scheduled for this organization.</p>}
      </>}
      {tab === 'members' && <Members members={detail.members} />}
    </div>
  </section>;
}

export default function Organizations() {
  const [organizations, setOrganizations] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const refresh = () => setRevision((value) => value + 1);
    window.addEventListener('mv:data-changed', refresh);
    return () => window.removeEventListener('mv:data-changed', refresh);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    read('/api/organizations', controller.signal).then((data) => {
      if (controller.signal.aborted) return;
      const items = data.organizations || [];
      setOrganizations(items);
      setSelectedId((current) => items.some((item) => String(item.id) === current) ? current : String(items[0]?.id || ''));
      setError(''); setLoading(false);
    }).catch((issue) => { if (!controller.signal.aborted) { setError(issue.message); setLoading(false); } });
    return () => controller.abort();
  }, [revision]);
  const visible = organizations.filter((item) => {
    const contact = contactInfo(item);
    return `${item.name} ${contact.name} ${contact.email} ${contact.phone}`.toLowerCase().includes(query.trim().toLowerCase());
  });
  const unavailable = loading || Boolean(error);
  const pageCount = Math.max(1, Math.ceil(visible.length / ORGANIZATIONS_PER_PAGE));
  const currentPage = Math.min(page, pageCount - 1);
  const start = currentPage * ORGANIZATIONS_PER_PAGE;
  const pageOrganizations = visible.slice(start, start + ORGANIZATIONS_PER_PAGE);
  return <main className="organizations-page"><div className="organizations-content">
    <header className="organizations-header"><div><span className="organization-eyebrow">People and planting partnerships</span><h1>Organizations</h1><p>Find contact details, planting sites, activity codes, and the members behind each organization.</p></div></header>
    {error && <div className="organization-load-error" role="alert">{error} <button type="button" className="organization-text-button" onClick={() => { setLoading(true); setRevision((value) => value + 1); }}>Retry</button></div>}
    <div className="organization-totals"><article><span>Organizations</span><strong>{unavailable ? '—' : number(organizations.length)}</strong><small>Saved organization profiles</small></article><article><span>Current project sites</span><strong>{unavailable ? '—' : number(organizations.reduce((sum, item) => sum + Number(item.project_site_count), 0))}</strong><small>Current organization planting areas</small></article><article><span>Registered members</span><strong>{unavailable ? '—' : number(organizations.reduce((sum, item) => sum + Number(item.member_count), 0))}</strong><small>Participants with a saved profile</small></article></div>
    <div className="organization-layout">
      <section className="organization-list" aria-label="Organization directory" data-guide="organization-list">
        <header className="organization-list-heading"><div><h2>Find an organization</h2><p>Select a profile to view its information.</p></div><span>{unavailable ? '—' : number(organizations.length)}</span></header>
        <div className="organization-list-body">
        <label className="organization-search">Search by name or contact<input type="search" value={query} placeholder="Name, email, or phone" onChange={(event) => { setQuery(event.target.value); setPage(0); }} /></label>
        <p className="organization-result-count" role="status" aria-live="polite">{loading ? 'Loading organizations…' : error ? 'Organization list unavailable.' : visible.length ? `Showing ${start + 1}–${Math.min(start + ORGANIZATIONS_PER_PAGE, visible.length)} of ${visible.length} organizations` : '0 organizations found'}</p>
        <div className="organization-list-results">
        {pageOrganizations.map((item) => <button key={item.id} className="organization-list-item" type="button" aria-pressed={String(item.id) === selectedId} onClick={() => setSelectedId(String(item.id))}><span className="organization-avatar"><PeopleIcon /></span><span><strong>{item.name}</strong><span>{contactInfo(item).email || 'Email not recorded'}</span><small>{number(item.project_site_count)} sites · {number(item.member_count)} members</small></span><span className="organization-list-arrow" aria-hidden="true">›</span></button>)}
        </div>
        {!unavailable && !visible.length && <p className="organization-empty">{organizations.length ? 'No organizations match your search.' : 'No organizations have been registered yet.'}</p>}
        </div>
        {!unavailable && pageCount > 1 && <nav className="organization-finder-pagination" aria-label="Organization pages">
          <button type="button" className="btn btn-secondary btn-sm" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>Previous</button>
          <span>Page {currentPage + 1} of {pageCount}</span>
          <button type="button" className="btn btn-secondary btn-sm" disabled={currentPage + 1 >= pageCount} onClick={() => setPage(currentPage + 1)}>Next</button>
        </nav>}
      </section>
      {selectedId && organizations.some((item) => String(item.id) === selectedId) ? <OrganizationDetail key={selectedId} id={selectedId} revision={revision} /> : <section className="organization-detail organization-empty">{loading ? 'Loading organizations…' : 'Select an organization to view its information.'}</section>}
    </div>
  </div></main>;
}
