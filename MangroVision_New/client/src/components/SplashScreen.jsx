import { useEffect, useState } from 'react';
import Logo from './Logo';
import './SplashScreen.css';

/**
 * Brand splash shown on first entry to the system.
 *
 * Renders for ~1.6 s on the very first page load of a tab session, then
 * fades away. We gate it on sessionStorage so route changes inside the SPA
 * never re-trigger it; closing the tab and reopening counts as a new entry.
 *
 * Usage:
 *   <SplashScreen onDone={() => setSplashShown(true)} />
 *
 * Props:
 *   - onDone:  called once the splash finishes its fade-out
 *   - skipKey: sessionStorage key used to detect "already shown this tab"
 */
const SHOW_MS = 1600;
const FADE_MS = 320;

export default function SplashScreen({ onDone, skipKey = 'mv_splash_seen' }) {
  const [phase, setPhase] = useState('show');

  useEffect(() => {
    const showTimer = window.setTimeout(() => setPhase('fade'), SHOW_MS);
    const doneTimer = window.setTimeout(() => {
      try { window.sessionStorage.setItem(skipKey, '1'); } catch { /* noop */ }
      onDone?.();
    }, SHOW_MS + FADE_MS);
    return () => {
      window.clearTimeout(showTimer);
      window.clearTimeout(doneTimer);
    };
  }, [onDone, skipKey]);

  return (
    <div
      className={`mv-splash mv-splash-${phase}`}
      role="status"
      aria-live="polite"
      aria-label="MangroVision is loading"
    >
      <div className="mv-splash-inner">
        <Logo variant="lockup" size={68} alt="MangroVision" />
        <div className="mv-splash-tagline">
          AI-powered mangrove planting zone analysis
        </div>
        <div className="mv-splash-bar" aria-hidden="true">
          <div className="mv-splash-bar-fill" />
        </div>
      </div>
    </div>
  );
}
