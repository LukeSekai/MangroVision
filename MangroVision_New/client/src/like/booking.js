import { APPOINTMENT_TYPES, appointmentTypeLabel } from '../utils/appointmentTypes.js';
import { formRequest } from '../utils/formValidation.js';

export function manilaToday(now = new Date()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Manila', year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(now);
  const values = Object.fromEntries(parts.map(({ type, value }) => [type, value]));
  return `${values.year}-${values.month}-${values.day}`;
}

export function bookingErrors(form, step, now = new Date()) {
  const errors = {};
  if (!step || step === 1) {
    for (const [field, label] of Object.entries({ organization: 'your organization', contact_name: 'a contact person', phone: 'a phone number', email: 'an email address' })) {
      if (!form[field].trim()) errors[field] = `Enter ${label}.`;
    }
    if (form.phone.trim() && (form.phone.replace(/\D/g, '').length < 7 || !/^[+()\d\s.-]+$/.test(form.phone.trim()))) errors.phone = 'Enter a valid phone number with at least 7 digits.';
    if (form.email.trim() && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) errors.email = 'Enter an email address you can access.';
    if (!APPOINTMENT_TYPES.some((type) => type.value === form.appointment_type)) errors.appointment_type = 'Choose an appointment type.';
  }
  if (!step || step === 2) {
    if (!form.date) errors.date = 'Choose your preferred date.';
    if (!form.start_time) errors.start_time = 'Choose a start time.';
    if (!form.end_time) errors.end_time = 'Choose an end time.';
    if (form.date && form.date < manilaToday(now)) errors.date = 'Choose a date on or after today.';
    if (form.date && form.date >= manilaToday(now) && form.start_time && new Date(`${form.date}T${form.start_time}:00+08:00`) <= now) {
      errors.start_time = 'Choose a date and time in the future.';
    }
    if (form.end_time && form.start_time && form.end_time <= form.start_time) errors.end_time = 'End time must be later than start time.';
    if (!Number.isInteger(Number(form.participants)) || Number(form.participants) < 1 || Number(form.participants) > 10000) {
      errors.participants = 'Enter a participant count between 1 and 10,000.';
    }
    if (!form.consent) errors.consent = 'Please acknowledge how your contact details will be used.';
  }
  return errors;
}

export function bookingPayload(form, submissionKey) {
  const fieldErrors = bookingErrors(form);
  if (Object.keys(fieldErrors).length) throw Object.assign(new Error(Object.values(fieldErrors)[0]), { fieldErrors });
  return {
    organization: form.organization.trim(), contact_name: form.contact_name.trim(),
    phone: form.phone.trim(), email: form.email.trim(), appointment_type: form.appointment_type,
    title: appointmentTypeLabel(form.appointment_type),
    start_at: `${form.date}T${form.start_time}:00+08:00`,
    end_at: `${form.date}T${form.end_time}:00+08:00`,
    participants: Number(form.participants), notes: form.notes.trim() || null,
    consent: true, website: form.website, submission_key: submissionKey,
  };
}

export async function submitBooking(payload) {
  const result = await formRequest(`${import.meta.env.VITE_API_BASE || ''}/api/public/like/appointments`, {
    method: 'POST', credentials: 'omit', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 'Check your contact details and appointment times, then try again.');
  if (!result.reference) throw new Error('We could not verify your request. Please try again.');
  return result;
}
