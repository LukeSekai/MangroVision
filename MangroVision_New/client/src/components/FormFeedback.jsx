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
