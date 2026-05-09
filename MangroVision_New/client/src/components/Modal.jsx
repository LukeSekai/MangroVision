import { useEffect, useRef } from 'react';
import './Modal.css';

const VARIANT_ICONS = {
  default: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  danger: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  ),
  warning: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  ),
  success: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  info: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  ),
};

const VARIANT_BUTTON_CLASS = {
  default: 'btn btn-primary',
  danger: 'btn btn-danger',
  warning: 'btn btn-primary',
  success: 'btn btn-primary',
  info: 'btn btn-primary',
};

export default function Modal({
  open,
  title,
  children,
  confirmLabel = 'OK',
  cancelLabel = 'Cancel',
  variant = 'default',
  busy = false,
  onConfirm,
  onCancel,
}) {
  const cardRef = useRef(null);
  const previouslyFocusedRef = useRef(null);
  const confirmButtonRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;

    previouslyFocusedRef.current = document.activeElement;
    confirmButtonRef.current?.focus();

    const handleKeyDown = (event) => {
      if (busy) return;
      if (event.key === 'Escape' && typeof onCancel === 'function') {
        event.preventDefault();
        onCancel();
        return;
      }
      if (event.key === 'Enter' && typeof onConfirm === 'function' && event.target?.tagName !== 'TEXTAREA') {
        event.preventDefault();
        onConfirm();
        return;
      }
      if (event.key !== 'Tab') return;

      const card = cardRef.current;
      if (!card) return;
      const focusables = card.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = previousOverflow;
      const previous = previouslyFocusedRef.current;
      if (previous && typeof previous.focus === 'function') {
        previous.focus();
      }
    };
  }, [open, busy, onCancel, onConfirm]);

  if (!open) return null;

  const confirmClass = VARIANT_BUTTON_CLASS[variant] || VARIANT_BUTTON_CLASS.default;
  const showCancel = typeof onCancel === 'function' && cancelLabel;
  const showConfirm = typeof onConfirm === 'function';
  const variantIcon = VARIANT_ICONS[variant] || VARIANT_ICONS.default;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={busy ? undefined : onCancel}>
      <div
        ref={cardRef}
        className={`modal-card modal-card-${variant}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="app-modal-title"
        aria-describedby="app-modal-body"
        onMouseDown={(event) => event.stopPropagation()}
      >
        {typeof onCancel === 'function' && (
          <button
            type="button"
            className="modal-close"
            onClick={onCancel}
            aria-label="Close modal"
            disabled={busy}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        )}

        <div className="modal-content">
          <div className={`modal-icon modal-icon-${variant}`} aria-hidden="true">
            {variantIcon}
          </div>

          <h2 id="app-modal-title" className="modal-title">{title}</h2>

          <div id="app-modal-body" className="modal-body">{children}</div>
        </div>

        <div className={`modal-actions${showCancel && showConfirm ? ' modal-actions-split' : ''}`}>
          {showCancel && (
            <button
              type="button"
              className="btn btn-secondary modal-action-btn"
              onClick={onCancel}
              disabled={busy}
            >
              {cancelLabel}
            </button>
          )}
          {showConfirm && (
            <button
              ref={confirmButtonRef}
              type="button"
              className={`${confirmClass} modal-action-btn${busy ? ' is-busy' : ''}`}
              onClick={onConfirm}
              disabled={busy}
            >
              {busy ? (
                <>
                  <span className="modal-spinner" aria-hidden="true" />
                  <span>Working…</span>
                </>
              ) : (
                confirmLabel
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
