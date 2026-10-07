import { useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { staffAuthRequest } from '../utils/staffAuth';
import EmailCodeInput from './EmailCodeInput';
import Logo from './Logo';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary } from './FormFeedback';
import './LoginScreen.css';

export default function LoginScreen() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState('login');
  const [challenge, setChallenge] = useState(null);
  const [readyAt, setReadyAt] = useState(0);
  const [code, setCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [notice, setNotice] = useState(() => sessionStorage.getItem('mv_auth_notice') || '');
  const login = useAuthStore((s) => s.login);
  const verifyLogin = useAuthStore((s) => s.verifyLogin);
  const feedback = useFormFeedback({
    username: { label: 'Username' },
    password: { label: 'Password' },
    code: { label: 'Email verification code', serverTerms: ['invalid code', 'code expired', 'code is invalid', 'code has expired', 'verification code'] },
    newPassword: { label: 'New password', aliases: ['new_password'], serverTerms: ['password must'] },
    confirmation: { label: 'Confirm new password', validate: (value) => value && value !== newPassword ? 'Enter the same password as the new password above.' : '' },
  });

  const acceptChallenge = (data, nextStep) => {
    feedback.clear();
    setChallenge(data);
    setReadyAt(Date.now() + data.resend_after * 1000);
    setCode('');
    setStep(nextStep);
    setPassword('');
  };

  const reset = (nextStep = 'login') => {
    feedback.clear();
    setStep(nextStep);
    setError('');
    setChallenge(null);
    setCode('');
    setPassword('');
    setNewPassword('');
    setConfirmation('');
    sessionStorage.removeItem('mv_auth_notice');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!feedback.validate()) return;
    setLoading(true);
    try {
      if (step === 'login') {
        acceptChallenge(await login(username, password), 'login-code');
      } else if (step === 'login-code') {
        await verifyLogin(code);
        sessionStorage.removeItem('mv_auth_notice');
      } else if (step === 'recovery') {
        acceptChallenge(await staffAuthRequest('recovery/request', {}), 'recovery-code');
      } else {
        const data = await staffAuthRequest('recovery/complete', {
          code, new_password: newPassword,
        });
        reset();
        setNotice(data.message);
      }
    } catch (err) {
      if (!feedback.fromServer(err)) setError(err.message || 'Authentication failed. Check your connection and try again.');
      if (err.retryAfter) setReadyAt(Date.now() + err.retryAfter * 1000);
    } finally {
      setLoading(false);
    }
  };

  const resend = async () => {
    setLoading(true);
    setError('');
    try {
      const data = step === 'login-code'
        ? await staffAuthRequest('login/resend', {})
        : await staffAuthRequest('recovery/request', {});
      acceptChallenge(data, step);
    } catch (err) {
      setError(err.message);
      if (err.retryAfter) setReadyAt(Date.now() + err.retryAfter * 1000);
    } finally {
      setLoading(false);
    }
  };

  const checkingCode = step.endsWith('-code');
  const recovering = step.startsWith('recovery');
  const title = checkingCode ? (recovering ? 'Set a new password' : 'Check your email') : (recovering ? 'Reset your password' : 'Sign In');
  const subtitle = step === 'login-code'
    ? `Enter the code sent to ${challenge?.email_hint}.`
    : step === 'recovery-code'
      ? 'Enter the recovery code sent to mangrovision.lgu@gmail.com and your new password.'
      : recovering ? 'We’ll send a recovery code to mangrovision.lgu@gmail.com to reset your password.'
        : 'Enter your credentials. We’ll email you a code to verify your sign-in.';

  return (
    <div className="login-screen">
      <div className="login-card">
        <div className="login-hero">
          <img src="/mangrove.jpg" alt="Mangrove forest" className="login-hero-img" />
          <div className="login-hero-overlay" />
          <div className="login-hero-content">
            <h2 className="login-hero-title">MangroVision</h2>
            <p className="login-hero-text">
              AI-powered mangrove planting zone analysis for coastal restoration.
            </p>
          </div>
        </div>
        <div className="login-form-side">
          <div className="login-form-inner">
            <div className="login-form-header">
              <div className="login-logo">
                <Logo variant="lockup" size={120} alt="MangroVision" />
              </div>
              <h1 className="login-title">{title}</h1>
              <p className="login-subtitle">{subtitle}</p>
            </div>

            <form className="login-form" noValidate onChangeCapture={feedback.onChange} onSubmit={handleSubmit}>
              <FormErrorSummary feedback={feedback} />
              {step === 'login' && <>
              <div className="form-group">
                <label className="form-label" htmlFor="login-username">Username</label>
                <input {...feedback.props('username')}
                  id="login-username"
                  className="form-input"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  autoComplete="username"
                  disabled={loading}
                  onChange={(e) => setUsername(e.target.value)}
                  autoFocus
                  required
                />
              <FieldError feedback={feedback} field="username" />
              </div>
              <div className="form-group">
                <label className="form-label" htmlFor="login-password">Password</label>
                <input {...feedback.props('password')}
                  id="login-password"
                  className="form-input"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  autoComplete="current-password"
                  disabled={loading}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              <FieldError feedback={feedback} field="password" />
              </div>
              </>}

              {checkingCode && <EmailCodeInput code={code} onChange={setCode} readyAt={readyAt} onResend={resend} busy={loading} feedback={feedback} />}

              {step === 'recovery-code' && <>
                <div className="form-group">
                  <label className="form-label" htmlFor="recovery-password">New password</label>
                  <input {...feedback.props('newPassword')} id="recovery-password" className="form-input" type="password" autoComplete="new-password"
                    minLength={12} maxLength={128} placeholder="At least 12 characters" value={newPassword}
                    onChange={(event) => setNewPassword(event.target.value)} required disabled={loading} />
              <FieldError feedback={feedback} field="newPassword" />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="recovery-confirm">Confirm new password</label>
                  <input {...feedback.props('confirmation')} id="recovery-confirm" className="form-input" type="password" autoComplete="new-password"
                    value={confirmation} onChange={(event) => setConfirmation(event.target.value)} required disabled={loading} />
              <FieldError feedback={feedback} field="confirmation" />
                </div>
              </>}

              {error && <div className="login-error" role="alert">{error}</div>}
              {notice && <div className="auth-notice" role="status">{notice}</div>}

              <button
                type="submit"
                className="btn btn-primary btn-lg login-submit"
                disabled={loading}
              >
                {loading ? 'Please wait…' : step === 'login' ? 'Send sign-in code'
                  : step === 'login-code' ? 'Verify and sign in' : step === 'recovery' ? 'Send recovery code' : 'Verify and reset password'}
              </button>
              <button type="button" className="auth-text-button" disabled={loading} onClick={() => {
                reset(step === 'login' ? 'recovery' : 'login');
                setNotice('');
              }}>
                {step === 'login' ? 'Forgot password?' : 'Back to sign in'}
              </button>
            </form>

            <p className="login-footer">
              Use the account provisioned by your administrator.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
