import { useEffect, useRef, useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { staffAuthRequest } from '../utils/staffAuth';
import EmailCodeInput from '../components/EmailCodeInput';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary } from '../components/FormFeedback';
import '../components/LoginScreen.css';
import './AccountSettings.css';

export default function AccountSettings() {
  const [account, setAccount] = useState(null);
  const accountLoaded = useRef(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [challenge, setChallenge] = useState(null);
  const [code, setCode] = useState('');
  const [readyAt, setReadyAt] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const clearSession = useAuthStore((state) => state.clearSession);
  const feedback = useFormFeedback({
    currentPassword: { label: 'Current password', aliases: ['current_password'], serverTerms: ['current password'] },
    password: { label: 'New password', aliases: ['new_password'], serverTerms: ['password must'] },
    confirmation: { label: 'Confirm new password', validate: (value) => value && value !== password ? 'Enter the same password as the new password above.' : '' },
    code: { label: 'Email verification code', serverTerms: ['invalid code', 'code expired', 'code is invalid', 'code has expired', 'verification code'] },
  });

  useEffect(() => {
    if (accountLoaded.current) return undefined;
    let active = true;
    staffAuthRequest('account').then((data) => {
      if (active) {
        accountLoaded.current = true;
        setAccount(data);
      }
    }).catch((err) => {
      if (active) { accountLoaded.current = true; setError(err.message); }
    });
    return () => { active = false; };
  }, []);

  const acceptChallenge = (data) => {
    feedback.clear();
    setChallenge(data);
    setCode('');
    setReadyAt(Date.now() + data.resend_after * 1000);
    // Only password hashes are pending on the server; clear browser inputs.
    setCurrentPassword('');
    setPassword('');
    setConfirmation('');
  };

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    if (!feedback.validate()) return;
    setBusy(true);
    try {
      if (challenge) {
        const data = await staffAuthRequest('account/verify', { code });
        clearSession(data.message);
      } else {
        acceptChallenge(await staffAuthRequest('account/change', {
          current_password: currentPassword, new_password: password,
        }));
      }
    } catch (err) {
      if (!feedback.fromServer(err)) setError(err.message);
      if (err.retryAfter) setReadyAt(Date.now() + err.retryAfter * 1000);
    } finally {
      setBusy(false);
    }
  };

  const resend = async () => {
    setBusy(true);
    setError('');
    try { acceptChallenge(await staffAuthRequest('account/resend', {})); }
    catch (err) {
      setError(err.message);
      if (err.retryAfter) setReadyAt(Date.now() + err.retryAfter * 1000);
    } finally { setBusy(false); }
  };

  return (
    <section className="account-page">
      <div className="account-card">
        <p className="account-eyebrow">Your staff account</p>
        <h1>Account settings</h1>
        <p className="credential-hint">Change your password. We’ll verify the change using your registered email.</p>
        {account && <div className="account-email"><span>Verification email</span><strong>{account.email}</strong></div>}
        {!account && !error && <p role="status">Loading your account…</p>}
        {account && <form className="login-form" noValidate onChangeCapture={feedback.onChange} onSubmit={submit}>
          <FormErrorSummary feedback={feedback} />
          {challenge ? <>
            <p className="credential-hint">Enter the code sent to {challenge.email_hint} to save your changes. You’ll then sign in again.</p>
            <EmailCodeInput code={code} onChange={setCode} readyAt={readyAt} onResend={resend} busy={busy} feedback={feedback} />
          </> : <>
            <div className="form-group">
              <label className="form-label" htmlFor="account-current-password">Current password</label>
              <input {...feedback.props('currentPassword')} className="form-input" id="account-current-password" autoComplete="current-password" type="password"
                required value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} disabled={busy} />
              <FieldError feedback={feedback} field="currentPassword" />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="account-new-password">New password</label>
              <input {...feedback.props('password')} className="form-input" id="account-new-password" autoComplete="new-password" type="password" minLength={12} maxLength={128}
                placeholder="At least 12 characters" required value={password} onChange={(event) => setPassword(event.target.value)} disabled={busy} />
              <FieldError feedback={feedback} field="password" />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="account-confirm-password">Confirm new password</label>
              <input {...feedback.props('confirmation')} className="form-input" id="account-confirm-password" autoComplete="new-password" type="password" required
                value={confirmation} onChange={(event) => setConfirmation(event.target.value)} disabled={busy} />
              <FieldError feedback={feedback} field="confirmation" />
            </div>
          </>}
          {error && <div className="login-error" role="alert">{error}</div>}
          <p className="credential-hint">Saving a change signs out all sessions on this account.</p>
          <button type="submit" className="btn btn-primary login-submit" disabled={busy}>
            {busy ? 'Please wait…' : challenge ? 'Verify and change password' : 'Send verification code'}
          </button>
          {challenge && <button type="button" className="auth-text-button" disabled={busy} onClick={() => { setChallenge(null); setCode(''); setError(''); feedback.clear(); }}>Cancel changes</button>}
        </form>}
        {!account && error && <div className="login-error" role="alert">{error}</div>}
      </div>
    </section>
  );
}
