import { useId, useRef, useState } from 'react';
import { controlError, serverFieldErrors } from './formValidation';

export default function useFormFeedback(fields) {
  const prefix = useId();
  const controls = useRef({});
  const [errors, setErrors] = useState({});
  const errorId = (field) => `${prefix}-${field}-error`;
  const reject = (nextErrors) => {
    setErrors(nextErrors);
    const first = Object.keys(fields).find((field) => nextErrors[field]);
    if (first) requestAnimationFrame(() => controls.current[first]?.focus());
    return false;
  };
  const clear = () => setErrors({});
  const onChange = (event) => {
    const field = event.target.name;
    setErrors((current) => {
      if (!current[field]) return current;
      const next = { ...current };
      delete next[field];
      return next;
    });
  };
  const validate = () => {
    const next = {};
    for (const [field, config] of Object.entries(fields)) {
      const control = controls.current[field];
      if (!control) continue;
      const message = controlError(control, config.label) || (!control.disabled && config.validate?.(control.value)) || '';
      if (message) next[field] = message;
    }
    if (Object.keys(next).length) return reject(next);
    clear();
    return true;
  };
  const fromServer = (error) => {
    // Authentication steps and conditional fields may be unmounted. Keep those
    // failures at form level so a correction is never hidden beside a missing input.
    const next = Object.fromEntries(Object.entries(serverFieldErrors(error, fields))
      .filter(([field]) => controls.current[field]));
    if (!Object.keys(next).length) return false;
    reject(next);
    return true;
  };
  return {
    errors, errorId, reject, clear, validate, fromServer, onChange,
    props: (field) => ({
      name: field,
      ref: (node) => { controls.current[field] = node; },
      'aria-label': fields[field].label,
      'aria-invalid': Boolean(errors[field]),
      'aria-describedby': errors[field] ? errorId(field) : undefined,
      // Validate on submission; losing focus can mean switching pages or tabs.
    }),
  };
}
