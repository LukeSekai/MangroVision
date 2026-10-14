import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Modal from './Modal';
import SiteMapPreview from './SiteMapPreview';
import './SitesToMonitor.css';
import './ProjectSiteActions.css';

const API = import.meta.env.VITE_API_BASE || '';
const errorMessage = (data, fallback) => typeof data?.detail === 'string' ? data.detail : fallback;

export default function ProjectSiteActions({ site, open, onToggle, locked, onComplete }) {
  const id = site.id ?? site.properties?.id;
  const name = site.properties?.name || 'project site';
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const [action, setAction] = useState(null);
  const [busy, setBusy] = useState(false);
  const trigger = useRef(null);
  const menu = useRef(null);
  const request = useRef(null);
  const submitting = useRef(false);

  const load = async () => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true); setError(''); setPlan(null);
    try {
      const response = await fetch(`${API}/api/project-sites/${id}/actions`, { signal: controller.signal, cache: 'no-store' });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(data, 'Could not check this site. Try again.'));
      if (!controller.signal.aborted) setPlan(data);
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure instanceof TypeError ? 'Could not check this site. Check your connection and try again.' : failure.message || 'Could not check this site. Try again.');
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  };

  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    if (!open) return undefined;
    const close = () => { onToggle(false); trigger.current?.focus({ preventScroll: true }); };
    const outside = (event) => {
      if (!menu.current?.contains(event.target) && !trigger.current?.contains(event.target)) onToggle(false);
    };
    const keyboard = (event) => {
      if (event.key === 'Tab') { onToggle(false); return; }
      if (event.key === 'Escape') { event.preventDefault(); close(); }
      if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const items = [...(menu.current?.querySelectorAll('[role="menuitem"]') || [])];
        if (!items.length) return;
        const index = items.indexOf(document.activeElement);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
          : (index + (event.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length;
        items[next]?.focus();
      }
    };
    menu.current?.querySelector('[role="menuitem"]')?.focus();
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', keyboard);
    window.addEventListener('resize', close);
    window.addEventListener('scroll', close, true);
    return () => {
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('keydown', keyboard);
      window.removeEventListener('resize', close);
      window.removeEventListener('scroll', close, true);
    };
  }, [open, onToggle]);

  const submit = async () => {
    if (submitting.current || locked || !plan?.[`can_${action}`]) return;
    submitting.current = true; setBusy(true); setError('');
    try {
      const response = await fetch(action === 'delete'
        ? `${API}/api/project-sites/${id}?version=${encodeURIComponent(plan.version)}`
        : `${API}/api/project-sites/${id}/release`, {
        method: action === 'delete' ? 'DELETE' : 'POST',
        ...(action === 'release' ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ version: plan.version }) } : {}),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setPlan(null);
        throw new Error(errorMessage(data, 'Could not complete this action. Review the site and try again.'));
      }
      setAction(null); onToggle(false);
      await onComplete(action, data);
    } catch (failure) {
      setError(failure instanceof TypeError ? 'Could not complete this action. Check your connection and try again.' : failure.message || 'Could not complete this action. Try again.');
    } finally {
      submitting.current = false; setBusy(false);
    }
  };

  const points = plan?.points?.map((point) => ({ ...point, map_status: point.status })) || [];
  const closeDialog = () => { if (!busy) { setAction(null); setError(''); trigger.current?.focus({ preventScroll: true }); } };
  return <>
    <button ref={trigger} type="button" className="btn btn-ghost btn-sm btn-icon site-actions-trigger"
      aria-label={`Actions for ${name}`} aria-haspopup="menu" aria-expanded={open} disabled={locked || busy}
      onClick={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        setPosition({ top: Math.max(8, Math.min(bounds.bottom + 6, window.innerHeight - 280)),
          left: Math.max(8, Math.min(bounds.right - 280, window.innerWidth - 288)) });
        onToggle(!open);
        if (!open) void load();
      }}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <circle cx="5" cy="12" r="2" /><circle cx="12" cy="12" r="2" /><circle cx="19" cy="12" r="2" />
      </svg>
    </button>
    {open && createPortal(<div ref={menu} role="menu" className="site-actions-menu" aria-label={`Actions for ${name}`} style={position}>
      {loading && <p role="status">Checking site…</p>}
      {error && <p role="alert">{error}</p>}
      {['release', 'delete'].map((choice) => <div key={choice} className="site-actions-choice">
        <button type="button" role="menuitem" aria-disabled={locked || !plan?.[`can_${choice}`]} className={choice === 'delete' ? 'is-danger' : ''}
          onClick={() => {
            if (locked || !plan?.[`can_${choice}`]) return;
            onToggle(false); setAction(choice); setError('');
          }}>{choice === 'release' ? 'Release area' : 'Delete site'}</button>
        {plan && !plan[`can_${choice}`] && <small>{plan[`${choice}_reason`]}</small>}
      </div>)}
      {error && <button type="button" className="btn btn-secondary btn-sm" onClick={load}>Try again</button>}
    </div>, document.body)}
    {action !== null && createPortal(<Modal open title={action === 'delete' ? 'Delete site' : 'Release area'}
      className="site-action-modal" variant={action === 'delete' ? 'danger' : 'warning'} busy={busy}
      confirmLabel={action === 'delete' ? 'Delete site' : 'Release area'} cancelLabel="Cancel"
      onConfirm={plan?.[`can_${action}`] && !locked ? submit : undefined} onCancel={closeDialog}>
      <p className="site-action-name">{name}</p>
      <p className="site-action-description">{action === 'release'
        ? 'Make unplanted and dead locations available to other organizations. Keep living seedlings in this site.'
        : plan?.can_delete ? 'All seedlings are recorded dead. Remove this site from the active map.' : 'Review this site before deleting it.'}</p>
      {plan && <>
        <dl className="site-action-counts">
          <div className="site-action-kept"><dt>Living kept</dt><dd>{plan.counts.living}</dd></div>
          <div><dt>Unplanted released</dt><dd>{plan.counts.unplanted}</dd></div>
          <div><dt>Dead locations released</dt><dd>{plan.counts.dead}</dd></div>
        </dl>
        {action === 'release' && <div className="site-action-boundaries">
          <figure><figcaption>Current area</figcaption><SiteMapPreview site={{ geometry: plan.before_geometry }} name={name} points={points} /></figure>
          <figure><figcaption>After release {plan.after_geometry && <span>Living seedlings only</span>}</figcaption>{plan.after_geometry
            ? <SiteMapPreview site={{ geometry: plan.after_geometry }} name={`${name}, area kept`} points={points.filter((point) => point.status === 'planted')} frameGeometry={plan.before_geometry} />
            : <p className="site-action-empty">No living seedlings remain. The full area will be released.</p>}</figure>
        </div>}
      </>}
      <div className="site-action-history">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />
        </svg>
        <div><strong>Site history stays saved</strong>
          <p>Planting records, deaths and survival rate are kept for this site.</p>
          {plan && <p>{plan.counts.historical_planted} planted · {plan.counts.historical_dead} recorded deaths</p>}
        </div>
      </div>
      {loading && <p role="status">Checking site…</p>}
      {plan && !plan[`can_${action}`] && <p role="alert">{plan[`${action}_reason`]}</p>}
      {error && <div><p role="alert">{error}</p><button type="button" className="btn btn-secondary btn-sm" onClick={load}>Review updated area</button></div>}
      {locked && <p role="alert">Site editing is locked while image processing is running.</p>}
    </Modal>, document.body)}
  </>;
}
