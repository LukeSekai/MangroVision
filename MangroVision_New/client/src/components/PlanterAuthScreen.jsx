import { useEffect, useRef, useState } from 'react';
import Logo from './Logo';
import GuideButton from './GuideButton';
import { FieldError } from './FormFeedback';
import { SIGN_IN_GUIDES_ENABLED } from '../guides/config';
import useFormFeedback from '../utils/useFormFeedback';
import { isPlanterCode } from '../utils/planterCode';
import { rememberedUsername, usePlanterAuthStore } from '../stores/planterAuthStore';

export default function PlanterAuthScreen() {
  const login = usePlanterAuthStore((state) => state.login);
  const checkOrganizationCode = usePlanterAuthStore((state) => state.checkOrganizationCode);
  const storeStatus = usePlanterAuthStore((state) => state.status);
  const storeError = usePlanterAuthStore((state) => state.error);
  const [localError, setLocalError] = useState('');
  const [mode, setMode] = useState(() => rememberedUsername() ? 'return' : 'first');
  const [activity, setActivity] = useState(null);
  const [switchPrompt, setSwitchPrompt] = useState(null);
  const heading = useRef(null);
  const busy = useRef(false);
  const feedback = useFormFeedback({
    access_code: { label: 'Organization code', validate: (value) => isPlanterCode(value) ? '' : 'Enter the eight-character organization code from your group leader.' },
    username: { label: 'Username', validate: (value) => /^[a-z0-9][a-z0-9._-]{2,31}$/i.test(value.trim()) ? '' : 'Use 3–32 letters or numbers. Dots, underscores and hyphens are allowed.' },
  });
  const [form, setForm] = useState({ access_code: '', username: rememberedUsername() });
  const submitting = storeStatus === 'loading';
  useEffect(() => { if (activity) heading.current?.focus(); }, [activity]);
  const clearErrors = () => {
    setLocalError(''); setSwitchPrompt(null); feedback.clear();
    usePlanterAuthStore.setState({ error: '' });
  };
  const update = (field) => (event) => {
    setLocalError(''); setSwitchPrompt(null);
    usePlanterAuthStore.setState({ error: '' });
    setForm((current) => ({ ...current, [field]: event.target.value }));
  };
  const submit = async (switchToken) => {
    if (busy.current || submitting) return;
    setLocalError('');
    usePlanterAuthStore.setState({ error: '' });
    if (!feedback.validate()) return;
    busy.current = true;
    try {
      if (mode === 'first' && !activity) {
        const verified = await checkOrganizationCode(form.access_code);
        if (verified.permanent_access) {
          setMode('return'); setActivity(null); feedback.clear();
          setForm((current) => ({ ...current, username: verified.username }));
          return;
        }
        setForm((current) => ({ ...current, username: '' }));
        feedback.clear();
        setActivity(verified);
      } else {
        await login(form.access_code, form.username, { mode, switchToken });
      }
    } catch (error) {
      if (error.kind === 'session_in_use') { setSwitchPrompt(error.switchToken); return; }
      if (error.kind === 'already_joined') {
        setActivity(null); setMode('return'); feedback.clear();
        setForm((current) => ({ ...current, username: error.participantUsername }));
        setLocalError('This browser has already joined. Sign in below to return to the same participant.');
        return;
      }
      if (error.kind === 'existing_participant') {
        usePlanterAuthStore.setState({ error: '' });
        setLocalError('No unused participant place is available. If you have planted before, ask LGU staff to help restore your existing participant and saved progress.');
        return;
      }
      // An expired/replaced code belongs on the code screen, where it can be corrected.
      const codeError = Array.isArray(error.detail) && error.detail.some((issue) => issue.loc?.includes('access_code'));
      if (activity && codeError) {
        setActivity(null); feedback.clear();
        usePlanterAuthStore.setState({ error: '' });
        setLocalError(error.message);
      } else if (feedback.fromServer(error)) usePlanterAuthStore.setState({ error: '' });
      else setLocalError(error.message || 'Could not sign in. Check your connection and try again.');
    } finally { busy.current = false; }
  };
  const chooseMode = (next) => { setMode(next); setActivity(null); clearErrors(); };
  const formContent = <form className="field-form" noValidate onChangeCapture={feedback.onChange} onSubmit={(event) => { event.preventDefault(); void submit(); }}>
    <fieldset className="field-auth-fields" disabled={submitting}>
      {!activity && <label className="field-label">Organization code
        <input {...feedback.props('access_code')} className="field-input field-access-code" type="text" value={form.access_code}
          onChange={update('access_code')} autoComplete="off" autoCapitalize="characters" spellCheck={false}
          placeholder="Organization code" maxLength={32} required />
        <FieldError feedback={feedback} field="access_code" />
      </label>}
      {(activity || mode === 'return') && <>
        <label className="field-label">{activity ? 'Choose your username' : 'Your username'}
          <input {...feedback.props('username')} className="field-input" value={form.username} onChange={update('username')}
            autoComplete={activity ? 'off' : 'username'} autoCapitalize="none" spellCheck={false} maxLength={32} required placeholder="Example: ana.cruz" />
          <FieldError feedback={feedback} field="username" />
        </label>
        <p className="field-label-help">{activity
          ? 'Choose a unique username for your organization. Use 3–32 letters or numbers; dots, underscores and hyphens are allowed.'
          : 'Use your username to return to your assigned points and saved progress. No password or PIN is needed.'}</p>
      </>}
    </fieldset>
    {(localError || storeError) && <div className="field-error" role="alert">{localError || storeError}</div>}
    {switchPrompt ? <div className="field-switch-prompt" role="alert">
      <p>This participant is already signed in. Switch access to this browser? The other login will end. Your points and progress stay the same.</p>
      <button className="field-submit" type="button" disabled={submitting} onClick={() => void submit(switchPrompt)}>{submitting ? 'Switching...' : 'Switch here'}</button>
      <button className="field-btn field-btn-outline" type="button" disabled={submitting} onClick={() => setSwitchPrompt(null)}>Cancel</button>
    </div> : <button className="field-submit" type="submit" disabled={submitting}>{submitting
      ? activity ? 'Saving username...' : mode === 'first' ? 'Checking code...' : 'Signing in...'
      : activity ? 'Save username and start' : mode === 'first' ? 'Continue' : 'Sign in'}</button>}
    {activity && <button className="field-btn field-btn-outline" type="button" disabled={submitting} onClick={() => { setActivity(null); clearErrors(); }}>Back to sign in</button>}
  </form>;

  if (activity) return <div className="field-shell field-onboarding-shell">
    <header className="field-header"><div className="field-header-brand"><Logo variant="lockup" size={36} alt="MangroVision" /><span className="field-header-brand-name">Field</span></div></header>
    <main className="field-onboarding-main">
      <section className="field-auth-card field-onboarding-card" aria-labelledby="field-username-heading">
        <p className="field-onboarding-eyebrow">Welcome to MangroVision Field</p>
        <h1 className="field-auth-title" id="field-username-heading" tabIndex={-1} ref={heading}>Choose your username</h1>
        <p className="field-onboarding-activity"><strong>{activity.organization_name}</strong><span>{activity.title}</span></p>
        <p className="field-label-help">Before you start, give your planter profile a username. It will keep your assigned points and planting progress together.</p>
        <p className="field-onboarding-reminder">Remember your username. Next time, choose <strong>Returning planter</strong> and enter it with your organization code.</p>
        {formContent}
      </section>
    </main>
  </div>;

  return <div className="field-auth-screen"><div className="field-auth-card">
    {SIGN_IN_GUIDES_ENABLED && <div className="login-guide"><GuideButton mode="field-login" /></div>}
    <div className="field-auth-brand"><Logo variant="lockup" size={44} alt="MangroVision" /><span className="field-auth-brand-suffix">Field</span></div>
    <h1 className="field-auth-subtitle">Planter sign in</h1>
    <div className="field-auth-mode" role="group" aria-label="Joining or returning">
      <button type="button" aria-pressed={mode === 'first'} onClick={() => chooseMode('first')} disabled={submitting}>First time</button>
      <button type="button" aria-pressed={mode === 'return'} onClick={() => chooseMode('return')} disabled={submitting}>Returning planter</button>
    </div>
    <p className="field-label-help">{mode === 'first'
      ? 'Enter the organization code shared by your group leader. You’ll choose your own username after you enter.'
      : 'Enter your organization code and the username you chose on your first visit.'}</p>
    {formContent}
  </div></div>;
}
