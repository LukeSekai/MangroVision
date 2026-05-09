import { useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useProcessingStore } from '../stores/processingStore';
import './ProcessingIndicator.css';

/**
 * Floating status badge shown at the bottom-center while an image analysis
 * is in progress. Stays visible across page navigations because it reads
 * from the global processingStore.
 *
 * Two behaviours live here in addition to rendering the badge:
 *   1. Click navigation — clicking the pill returns to /processing.
 *   2. Auto-redirect — when a processing run finishes successfully on a
 *      different page, route the user back to /processing so the result
 *      modal opens automatically. Failed runs do NOT auto-redirect; the
 *      user can keep what they were doing and revisit /processing manually.
 *
 * Renders nothing when there is no active processing job, so it is safe to
 * mount unconditionally inside AppShell.
 */
export default function ProcessingIndicator() {
  const processing = useProcessingStore((s) => s.processing);
  const stage = useProcessingStore((s) => s.stage);
  const progress = useProcessingStore((s) => s.progress);
  const fileName = useProcessingStore((s) => s.fileName);
  const saving = useProcessingStore((s) => s.saving);
  const result = useProcessingStore((s) => s.result);
  const error = useProcessingStore((s) => s.error);

  const navigate = useNavigate();
  const location = useLocation();

  const prevProcessingRef = useRef(processing);

  // Auto-redirect to /processing when a run finishes (transition from
  // processing=true to processing=false WITH a result and no error).
  useEffect(() => {
    const wasProcessing = prevProcessingRef.current;
    const justFinished = wasProcessing && !processing;
    if (justFinished && result && !error) {
      if (!location.pathname.startsWith('/processing')) {
        navigate('/processing');
      }
    }
    prevProcessingRef.current = processing;
  }, [processing, result, error, location.pathname, navigate]);

  // Hide on the Image Processing page itself — the full progress card is
  // already on screen, the floating badge would just duplicate the info.
  if (location.pathname.startsWith('/processing')) return null;

  // Show the badge while either processing or saving so the user sees that
  // their click was received and the long-running save is happening.
  if (!processing && !saving) return null;

  const label = processing ? (stage || 'Processing image…') : 'Saving analysis…';
  const pct = processing && typeof progress === 'number' ? Math.max(0, Math.min(100, progress)) : null;

  const handleClick = () => {
    navigate('/processing');
  };

  return (
    <button
      type="button"
      className="processing-indicator"
      onClick={handleClick}
      aria-label={`${label}. Click to open the Image Processing page.`}
      title="Open Image Processing"
    >
      <span className="processing-indicator__spinner" aria-hidden="true" />
      <span className="processing-indicator__body">
        <span className="processing-indicator__label">{label}</span>
        {fileName && processing && (
          <span className="processing-indicator__file">{fileName}</span>
        )}
        {pct !== null && (
          <span className="processing-indicator__bar" aria-hidden="true">
            <span
              className="processing-indicator__bar-fill"
              style={{ width: `${pct}%` }}
            />
          </span>
        )}
      </span>
      {pct !== null && (
        <span className="processing-indicator__pct" aria-hidden="true">{pct}%</span>
      )}
    </button>
  );
}
