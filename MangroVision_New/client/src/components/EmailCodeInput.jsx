import { useEffect, useState } from 'react';
import { FieldError } from './FormFeedback';

export default function EmailCodeInput({ code, onChange, readyAt, onResend, busy, feedback }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const remaining = Math.max(0, Math.ceil((readyAt - now) / 1000));
  return (
    <>
      <div className="form-group">
        <label className="form-label" htmlFor="email-code">Email verification code</label>
        <input {...feedback?.props('code')} id="email-code" className="form-input email-code-input" type="text"
          inputMode="numeric" autoComplete="one-time-code" pattern="[0-9]{6}" maxLength={6}
          placeholder="000000" value={code} onChange={(event) => onChange(event.target.value.replace(/\D/g, '').slice(0, 6))}
          title="Enter the six-digit code from your email." required autoFocus disabled={busy} />
        {feedback && <FieldError feedback={feedback} field="code" />}
      </div>
      <div className="email-code-help">
        <span>The code expires in 10 minutes. Check spam too.</span>
        <button type="button" className="auth-text-button" onClick={onResend} disabled={busy || remaining > 0}>
          {remaining > 0 ? `Resend in ${remaining}s` : 'Resend code'}
        </button>
      </div>
    </>
  );
}
