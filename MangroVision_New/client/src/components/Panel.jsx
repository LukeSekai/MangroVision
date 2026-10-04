import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from 'react';
import './Panel.css';

const PanelAccordionContext = createContext(null);

/**
 * Floating side panel with hide/show toggle and collapsible sections.
 */
export function Panel({
  title,
  subtitle,
  children,
  defaultHidden = false,
  accordion = true,
  initialOpenKeys = [],
  className = '',
  toggleClassName = '',
}) {
  const [hidden, setHidden] = useState(defaultHidden);
  const [openKeys, setOpenKeys] = useState(() => (
    Array.isArray(initialOpenKeys) ? initialOpenKeys : []
  ));

  const registerCard = useCallback((key, defaultOpen) => {
    if (!defaultOpen) return;
    setOpenKeys((currentKeys) => {
      if (currentKeys.length > 0) return currentKeys;
      return [key];
    });
  }, []);

  const setCardOpen = useCallback((key, nextOpen) => {
    setOpenKeys((currentKeys) => {
      if (nextOpen) return [key];
      return currentKeys.filter((openKey) => openKey !== key);
    });
  }, []);

  const accordionContext = useMemo(() => (
    accordion
      ? {
          openKeys,
          registerCard,
          setCardOpen,
          isCardOpen: (key) => openKeys.includes(key),
        }
      : null
  ), [accordion, openKeys, registerCard, setCardOpen]);

  return (
    <>
      <button
        className={`panel-toggle ${toggleClassName} ${hidden ? 'panel-toggle-hidden' : ''}`.trim()}
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

      <div className={`floating-panel ${className} ${hidden ? 'floating-panel-hidden' : ''}`.trim()}>
        <div className="floating-panel-header">
          <div>
            <h2 className="floating-panel-title">{title}</h2>
            {subtitle && <span className="floating-panel-subtitle">{subtitle}</span>}
          </div>
        </div>
        <PanelAccordionContext.Provider value={accordionContext}>
          <div className="floating-panel-body">
            {children}
          </div>
        </PanelAccordionContext.Provider>
      </div>
    </>
  );
}

/**
 * Collapsible card section inside a Panel — with smooth height animation.
 */
export function PanelCard({
  icon,
  title,
  badge,
  children,
  defaultOpen = true,
  open: controlledOpen,
  onOpenChange,
  panelKey,
  headerRef,
  className = '',
}) {
  const generatedKey = useId();
  const cardKey = panelKey || generatedKey;
  const accordionContext = useContext(PanelAccordionContext);
  const registerPanelCard = accordionContext?.registerCard;
  const setPanelCardOpen = accordionContext?.setCardOpen;
  const isPanelCardOpen = accordionContext?.isCardOpen;
  const isControlled = controlledOpen !== undefined;
  const usesPanelAccordion = Boolean(accordionContext) && !isControlled;
  const [internalOpen, setInternalOpen] = useState(defaultOpen);
  const open = isControlled
    ? Boolean(controlledOpen)
    : usesPanelAccordion
      ? isPanelCardOpen(cardKey)
      : internalOpen;
  const initialOpen = open;
  const bodyRef = useRef(null);
  const [height, setHeight] = useState(initialOpen ? 'auto' : '0px');
  const [overflow, setOverflow] = useState(initialOpen ? 'visible' : 'hidden');
  const isFirstRender = useRef(true);

  const toggleOpen = () => {
    const nextOpen = !open;
    if (usesPanelAccordion) {
      setPanelCardOpen(cardKey, nextOpen);
    } else if (!isControlled) {
      setInternalOpen(nextOpen);
    }
    onOpenChange?.(nextOpen);
  };

  useEffect(() => {
    if (usesPanelAccordion) {
      registerPanelCard(cardKey, defaultOpen);
    }
  }, [cardKey, defaultOpen, registerPanelCard, usesPanelAccordion]);

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
    <div className={`panel-card ${className} ${open ? 'panel-card-open' : ''}`.trim()}>
      <button ref={headerRef} className="panel-card-header" aria-expanded={open} onClick={toggleOpen}>
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
