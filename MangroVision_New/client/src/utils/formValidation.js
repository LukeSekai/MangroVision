export function controlError(control, label) {
  if (control.disabled || control.readOnly) return '';
  const validity = control.validity;
  if (validity.valueMissing || (control.required && !String(control.value).trim())) {
    return `${control.tagName === 'SELECT' ? 'Choose' : 'Enter'} ${label.toLowerCase()}.`;
  }
  if (validity.badInput) return `Enter a number for ${label.toLowerCase()}.`;
  if (validity.rangeUnderflow) return control.type === 'date' ? `Choose ${control.min} or later for ${label.toLowerCase()}.` : `Enter ${control.min} or more for ${label.toLowerCase()}.`;
  if (validity.rangeOverflow) return control.type === 'date' ? `Choose ${control.max} or earlier for ${label.toLowerCase()}.` : `Enter ${control.max} or less for ${label.toLowerCase()}.`;
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

export function submissionError(detail, fallback = 'Could not save. Please try again.', status) {
  const statusMessage = status === 503 && detail?.kind === 'shared_access_setup_required'
    ? 'The server needs a database update before planting requests can be approved. Finish the shared planter-code setup, then try again. Your entries are kept.'
    : status === 401 && !Array.isArray(detail) && (!detail || /session|expired/i.test(detail)) ? 'Your session has expired. Sign in again, then retry.'
    : status === 429 ? 'Too many requests. Please wait before trying again.'
      : status >= 500 ? 'The server could not complete this request. Your entries are kept; please retry.' : '';
  const error = new Error(statusMessage || (typeof detail === 'string' ? detail : fallback));
  error.detail = detail;
  error.status = status;
  return error;
}

// Use the same response handling for forms, retaining structured field errors.
export async function formRequest(url, options = {}, fallback) {
  let response;
  try { response = await fetch(url, options); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw Object.assign(new Error('Could not connect. Check your connection and retry. Your entries are kept.'), {
      uncertain: !['GET', 'HEAD'].includes(String(options.method || 'GET').toUpperCase()),
    });
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw submissionError(payload?.detail, fallback, response.status);
  if (payload === null) throw Object.assign(new Error('Could not verify the result. Refresh before submitting again.'), { uncertain: true });
  return payload;
}

export function serverFieldErrors(error, fields) {
  if ((error?.status === 401 && !Array.isArray(error.detail)) || error?.status === 429 || error?.status >= 500) return {};
  if (error?.fieldErrors) return Object.fromEntries(Object.entries(error.fieldErrors).filter(([field]) => fields[field]));
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
