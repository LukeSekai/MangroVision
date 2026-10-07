import { useEffect, useRef, useState } from 'react';
import './LikeMapDialog.css';

export default function LikeMapDialog({ onClose }) {
  const dialogRef = useRef(null);
  const containerRef = useRef(null);
  const viewerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const dialog = dialogRef.current;
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    dialog.showModal();
    document.body.style.overflow = 'hidden';
    return () => {
      dialog.close();
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus({ preventScroll: true });
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    // Load the mapping library only when a visitor opens the map. This viewer
    // never imports staff stores, points, zones, filters, or account data.
    import('./publicMap.js').then(({ createPublicMap }) => {
      if (cancelled) return;
      viewerRef.current = createPublicMap(containerRef.current, {
        onLoading: (value) => { if (!cancelled) setLoading(value); },
        onError: () => { if (!cancelled) setError('Some map imagery could not load. Please try again.'); },
      });
      setReady(true);
    }).catch(() => {
      if (!cancelled) { setLoading(false); setError('The map could not load. Please try again.'); }
    });
    return () => {
      cancelled = true;
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, [attempt]);

  const retry = () => {
    setError('');
    setLoading(true);
    if (viewerRef.current) viewerRef.current.retry();
    else setAttempt((value) => value + 1);
  };

  const keepFocusInMap = (event) => {
    if (event.key !== 'Tab') return;
    const controls = [...dialogRef.current.querySelectorAll('button, a[href], [tabindex]')]
      .filter((element) => element.tabIndex >= 0 && !element.disabled && element.getClientRects().length);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first?.focus();
    }
  };

  return <dialog className="like-map-dialog" ref={dialogRef} aria-labelledby="like-map-title" aria-describedby="like-map-description"
    onKeyDown={keepFocusInMap}
    onCancel={(event) => { event.preventDefault(); onClose(); }}
    onClick={(event) => { if (event.target === dialogRef.current) onClose(); }}>
    <div className="like-map-shell">
      <header className="like-map-header">
        <div><span className="like-eyebrow">A CLOSER LOOK AT LIKE</span><h2 id="like-map-title" className="like-map-heading">LIKE ecopark map</h2><p id="like-map-description">Discover the mangroves and coast through our aerial map.</p></div>
        <button className="like-map-close" type="button" aria-label="Close map" onClick={onClose}><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="m6 6 12 12M6 18 18 6" /></svg></button>
      </header>
      <div className="like-map-stage">
        <div ref={containerRef} className="like-map-canvas" role="region" aria-label="Interactive LIKE ecopark map" />
        {loading || error ? <div className="like-map-status" role="status"><span>{error || 'Loading map imagery…'}</span>{error ? <button type="button" onClick={retry}>Retry</button> : null}</div> : null}
      </div>
      <footer className="like-map-footer"><p>Drag to explore. Use + and − to zoom.</p><button className="like-button is-secondary is-small" type="button" disabled={!ready} onClick={() => viewerRef.current?.reset()}>Reset view</button></footer>
    </div>
  </dialog>;
}
