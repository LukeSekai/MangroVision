import { lazy, Suspense, useCallback, useRef, useState } from 'react';
import { useInRouterContext, useLocation, useNavigate } from 'react-router-dom';
import Modal from './Modal';
import { fullWorkflow, toursForMode } from '../guides/tours';
import './GuideButton.css';

const GuideTour = lazy(() => import('./GuideTour'));

function RoutedGuideButton(props) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  return <GuideMenu {...props} pathname={pathname} navigate={navigate} />;
}

export default function GuideButton(props) {
  const inRouter = useInRouterContext();
  return inRouter ? <RoutedGuideButton {...props} /> : <GuideMenu {...props} />;
}

function GuideMenu({ mode = 'staff', pathname = '/', navigate }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(null);
  const buttonRef = useRef(null);
  const tours = toursForMode(mode);
  const current = tours.find((tour) => tour.path === pathname) || tours[0];
  const finish = useCallback(() => {
    setActive(null);
    window.requestAnimationFrame(() => buttonRef.current?.focus());
  }, []);
  const start = (definitions) => {
    setOpen(false);
    setActive(definitions);
  };

  return <>
    <button ref={buttonRef} type="button" className="guide-trigger" data-guide="help"
      aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(true)}>
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
        <circle cx="12" cy="12" r="9" /><path d="M9.4 9a2.7 2.7 0 0 1 5.2 1c0 1.8-2.6 2-2.6 3.5" /><path d="M12 17h.01" />
      </svg>
      <span>Help / Guide</span>
    </button>
    <Modal open={open} title="How to use MangroVision" cancelLabel="Close" onCancel={() => setOpen(false)} className="guide-menu-modal">
      <p className="guide-menu-intro">Choose a step-by-step guide. You can go back, end it at any time, and replay it here.</p>
      <button type="button" className="guide-featured" onClick={() => start(current.steps)}>
        <span className="guide-eyebrow">{mode === 'staff' ? 'THIS SCREEN' : 'GET STARTED'}</span>
        <strong>{current.title}</strong>
        <span>{current.description}</span>
        <span className="guide-featured-action">Start guide · {current.steps.length} steps →</span>
      </button>
      {mode === 'staff' && <>
        <button type="button" className="guide-workflow-button" onClick={() => start(fullWorkflow(tours))}>
          <span><strong>Full system workflow</strong><small>From site preparation to planting, monitoring and reports</small></span>
          <span aria-hidden="true">→</span>
        </button>
        <h3 className="guide-menu-label">Guides by task</h3>
        <div className="guide-topic-list">
          {tours.map((tour) => <button type="button" key={tour.id} className="guide-topic" onClick={() => start(tour.steps)}>
            <span><strong>{tour.title}</strong><small>{tour.description}</small></span>
            <span className="guide-topic-count">{tour.steps.length} steps</span>
          </button>)}
        </div>
      </>}
    </Modal>
    {active && <Suspense fallback={<div className="guide-loading" role="status">Loading your guide… <button type="button" onClick={finish}>Cancel</button></div>}>
      <GuideTour definitions={active} onDone={finish} pathname={pathname} navigate={navigate} />
    </Suspense>}
  </>;
}
