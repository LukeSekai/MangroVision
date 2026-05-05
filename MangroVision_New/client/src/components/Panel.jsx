import { useState, useRef, useEffect } from 'react';
import './Panel.css';

/**
 * Floating side panel with hide/show toggle and collapsible sections.
 */
export function Panel({ title, subtitle, children, defaultHidden = false }) {
  const [hidden, setHidden] = useState(defaultHidden);

  return (
    <>
      <button
        className={`panel-toggle ${hidden ? 'panel-toggle-hidden' : ''}`}
        onClick={() => setHidden((h) => !h)}
        title={hidden ? 'Show panel' : 'Hide panel'}
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transform: hidden ? 'rotate(180deg)' : 'none' }}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        <span>{hidden ? 'Show' : 'Hide'}</span>
      </button>

      <div className={`floating-panel ${hidden ? 'floating-panel-hidden' : ''}`}>
        <div className="floating-panel-header">
          <div>
            <h2 className="floating-panel-title">{title}</h2>
            {subtitle && <span className="floating-panel-subtitle">{subtitle}</span>}
          </div>
        </div>
        <div className="floating-panel-body">
          {children}
        </div>
      </div>
    </>
  );
}

/**
 * Collapsible card section inside a Panel — with smooth height animation.
 */
export function PanelCard({ icon, title, badge, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyRef = useRef(null);
  const [height, setHeight] = useState(defaultOpen ? 'auto' : '0px');
  const [overflow, setOverflow] = useState(defaultOpen ? 'visible' : 'hidden');
  const isFirstRender = useRef(true);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }

    const el = bodyRef.current;
    if (!el) return;

    if (open) {
      // Opening: measure scrollHeight, animate to it
      const scrollH = el.scrollHeight;
      setHeight(`${scrollH}px`);
      setOverflow('hidden');
      const timer = setTimeout(() => {
        setHeight('auto');
        setOverflow('visible');
      }, 280);
      return () => clearTimeout(timer);
    } else {
      // Closing: set explicit height first, then collapse
      const scrollH = el.scrollHeight;
      setHeight(`${scrollH}px`);
      setOverflow('hidden');
      // Force reflow before collapsing
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setHeight('0px');
        });
      });
    }
  }, [open]);

  return (
    <div className={`panel-card ${open ? 'panel-card-open' : ''}`}>
      <button className="panel-card-header" onClick={() => setOpen((o) => !o)}>
        <div className="panel-card-header-left">
          {icon && <span className="panel-card-icon">{icon}</span>}
          <span className="panel-card-title">{title}</span>
          {badge !== undefined && <span className="panel-card-badge">{badge}</span>}
        </div>
        <svg
          className={`panel-card-chevron ${open ? 'panel-card-chevron-open' : ''}`}
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      <div
        ref={bodyRef}
        className="panel-card-body-wrapper"
        style={{ height, overflow }}
      >
        <div className="panel-card-body">{children}</div>
      </div>
    </div>
  );
}
