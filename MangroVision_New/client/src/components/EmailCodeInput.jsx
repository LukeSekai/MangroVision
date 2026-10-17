import { useEffect, useId, useRef, useState } from 'react';
import { FieldError } from './FormFeedback';
import './EmailCodeInput.css';

const CODE_LENGTH = 6;

export default function EmailCodeInput({ code, onChange, readyAt, onResend, busy, feedback }) {
  const id = useId();
  const inputs = useRef([]);
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!busy && !code) inputs.current[0]?.focus();
  }, [busy, code]);

  const digits = Array.from({ length: CODE_LENGTH }, (_, index) => /[0-9]/.test(code[index] || '') ? code[index] : '');
  const field = feedback?.props('code');
  const complete = /^[0-9]{6}$/.test(code);
  const focus = (index) => inputs.current[Math.max(0, Math.min(index, CODE_LENGTH - 1))]?.focus();
  const update = (next) => {
    // Keep an empty position when correcting a middle digit. Only a complete
    // six-digit value passes the form's existing verification-code validation.
    onChange(next.map((digit) => digit || ' ').join('').trimEnd());
    feedback?.onChange({ target: { name: 'code' } });
  };
  const fill = (index, value) => {
    const incoming = value.replace(/\D/g, '').slice(0, CODE_LENGTH);
    if (!incoming) return;
    const start = incoming.length === CODE_LENGTH ? 0 : index;
    const next = [...digits];
    [...incoming.slice(0, CODE_LENGTH - start)].forEach((digit, offset) => { next[start + offset] = digit; });
    update(next);
    focus(start + incoming.length);
  };
  const clear = (index) => {
    const next = [...digits];
    next[index] = '';
    update(next);
  };
  const handleKeyDown = (event, index) => {
    if (event.key === 'Backspace') {
      event.preventDefault();
      const target = digits[index] || index === 0 ? index : index - 1;
      clear(target);
      focus(target);
    } else if (event.key === 'Delete') {
      event.preventDefault();
      clear(index);
    } else if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      focus(event.key === 'Home' ? 0 : event.key === 'End' ? CODE_LENGTH - 1 : index + (event.key === 'ArrowLeft' ? -1 : 1));
    }
  };
  const remaining = Math.max(0, Math.ceil((readyAt - now) / 1000));
  return (
    <>
      <div className="form-group">
        <label id={`${id}-label`} className="form-label" htmlFor={`${id}-digit-0`}>Email verification code</label>
        <div className="email-code-fields" role="group" aria-labelledby={`${id}-label`}>
          {digits.map((digit, index) => <input
            key={index}
            {...field}
            ref={(node) => {
              inputs.current[index] = node;
              if (index === 0) {
                field?.ref(node);
                node?.setCustomValidity(complete ? '' : 'Enter all six digits from your email.');
              }
            }}
            id={`${id}-digit-${index}`}
            className={`form-input email-code-digit${digit ? ' is-filled' : ''}`}
            type="text"
            name="code"
            aria-label={`Digit ${index + 1} of ${CODE_LENGTH}`}
            inputMode="numeric"
            autoComplete={index === 0 ? 'one-time-code' : 'off'}
            pattern="[0-9]"
            maxLength={CODE_LENGTH}
            value={digit}
            onFocus={(event) => event.target.select()}
            onKeyDown={(event) => handleKeyDown(event, index)}
            onPaste={(event) => {
              event.preventDefault();
              fill(index, event.clipboardData.getData('text'));
            }}
            onChange={(event) => {
              if (!event.target.value) clear(index);
              else fill(index, event.nativeEvent.inputType === 'insertText' && event.nativeEvent.data
                ? event.nativeEvent.data : event.target.value);
            }}
            title="Enter the six-digit code from your email."
            required
            disabled={busy}
          />)}
        </div>
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
