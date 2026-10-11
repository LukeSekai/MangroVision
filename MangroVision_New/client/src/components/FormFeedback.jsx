import './FormFeedback.css';

export function FieldError({ feedback, field }) {
  const message = feedback.errors[field];
  return message ? <span className="form-field-error" id={feedback.errorId(field)}>{message}</span> : null;
}

export function FormErrorSummary({ feedback }) {
  const count = Object.keys(feedback.errors).length;
  return count ? <p className="form-error-summary" role="alert">
    {count === 1 ? 'Correct the highlighted field' : `Correct the ${count} highlighted fields`} and try again.
  </p> : null;
}

export function FormMessage({ error, success, loading, onRetry, busy = false }) {
  if (!error && !success && !loading) return null;
  return <div className={`form-message ${error ? 'is-error' : loading ? 'is-loading' : 'is-success'}`}
    role={error ? 'alert' : 'status'} aria-live={error ? 'assertive' : 'polite'}>
    <span>{error || loading || success}</span>
    {error && onRetry ? <button type="button" onClick={onRetry} disabled={busy}>Retry</button> : null}
  </div>;
}
