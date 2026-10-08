export function controlError(control, label) {
  if (control.disabled || control.readOnly) return '';
  const validity = control.validity;
  if (validity.valueMissing || (control.required && !String(control.value).trim())) {
    return `${control.tagName === 'SELECT' ? 'Choose' : 'Enter'} ${label.toLowerCase()}.`;
  }
  if (validity.badInput) return `Enter a number for ${label.toLowerCase()}.`;
  if (validity.rangeUnderflow) return `Enter ${control.min} or more for ${label.toLowerCase()}.`;
  if (validity.rangeOverflow) return `Enter ${control.max} or less for ${label.toLowerCase()}.`;
  if (validity.stepMismatch) return control.step === '1'
    ? `Enter a whole number for ${label.toLowerCase()}.`
    : `Use increments of ${control.step} for ${label.toLowerCase()}.`;
  if (validity.typeMismatch && control.type === 'email') return 'Enter an email address such as name@example.com.';
  if (validity.tooShort || (control.value.length > 0 && control.minLength > 0 && control.value.length < control.minLength)) return `Use at least ${control.minLength} characters for ${label.toLowerCase()}.`;
  if (validity.tooLong || (control.maxLength > 0 && control.value.length > control.maxLength)) return `Use no more than ${control.maxLength} characters for ${label.toLowerCase()}.`;
  if (validity.patternMismatch) return control.title || `Enter ${label.toLowerCase()} in the requested format.`;
  if (!validity.valid) return control.validationMessage || `Check ${label.toLowerCase()}.`;
  return '';
}

export function submissionError(detail, fallback = 'Could not save. Please try again.') {
  const error = new Error(typeof detail === 'string' ? detail : fallback);
  error.detail = detail;
  return error;
}

export function serverFieldErrors(error, fields) {
  const detail = error?.detail;
  const issues = Array.isArray(detail) ? detail : [];
  const result = {};
  for (const issue of issues) {
    const location = issue.loc?.filter((part) => typeof part === 'string') || [];
    const field = Object.keys(fields).find((name) => [name, ...(fields[name].aliases || [])].some((alias) => location.includes(alias)));
    if (field) {
      const label = fields[field].label.toLowerCase();
      const ctx = issue.ctx || {};
      if (issue.type === 'missing') result[field] = `Enter ${label}.`;
      else if (issue.type === 'credential_error') result[field] = String(issue.msg || `Check ${label}.`);
      else if (ctx.ge !== undefined) result[field] = `Enter ${ctx.ge} or more for ${label}.`;
      else if (ctx.le !== undefined) result[field] = `Enter ${ctx.le} or less for ${label}.`;
      else if (ctx.min_length !== undefined) result[field] = `Use at least ${ctx.min_length} characters for ${label}.`;
      else if (ctx.max_length !== undefined) result[field] = `Use no more than ${ctx.max_length} characters for ${label}.`;
      else if (issue.type === 'int_parsing' || issue.type === 'int_from_float') result[field] = `Enter a whole number for ${label}.`;
      else result[field] = `${fields[field].label}: ${String(issue.msg || 'check this value').replace(/\.$/, '')}. Correct this value and try again.`;
    }
  }
  if (!issues.length) {
    const message = String(error?.message || error || '');
    // Match explicit server field names; connection/session failures stay at form level.
    const field = Object.keys(fields).find((name) => (fields[name].serverTerms || [])
      .some((term) => message.toLowerCase().includes(term.toLowerCase())));
    if (field) result[field] = message;
  }
  return result;
}
