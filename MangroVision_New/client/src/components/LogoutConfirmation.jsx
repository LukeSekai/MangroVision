import { useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Modal from './Modal';

export default function LogoutConfirmation({ open, onConfirm, onCancel }) {
  const inFlight = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const confirm = async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError('');
    try {
      await onConfirm();
    } catch {
      setError('Could not sign out. Please try again.');
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  };
  const cancel = () => {
    if (inFlight.current) return;
    setError('');
    onCancel();
  };

  if (!open) return null;

  return createPortal(<Modal open={open} title="Sign out of MangroVision?"
    confirmLabel="Sign out" cancelLabel="Stay signed in" initialFocus="cancel"
    busy={busy} onConfirm={confirm} onCancel={cancel}
    icon={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <path d="m16 17 5-5-5-5M21 12H9" />
    </svg>}>
    <p>You’ll return to the sign-in screen. Your saved work will be kept.</p>
    {error && <p className="form-message is-error" role="alert">{error}</p>}
  </Modal>, document.body);
}
