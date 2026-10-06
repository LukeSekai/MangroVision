export function manilaToday(now = new Date()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Manila', year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(now);
  const values = Object.fromEntries(parts.map(({ type, value }) => [type, value]));
  return `${values.year}-${values.month}-${values.day}`;
}

export function bookingPayload(form, submissionKey) {
  if (!form.organization.trim() || !form.contact_name.trim() || !form.phone.trim()) {
    throw new Error('Enter your organization, contact person, and phone number.');
  }
  if (!form.date || !form.start_time || !form.end_time) {
    throw new Error('Choose your preferred date, start time, and end time.');
  }
  if (form.date < manilaToday() || new Date(`${form.date}T${form.start_time}:00+08:00`) <= new Date()) {
    throw new Error('Choose a date and time in the future.');
  }
  if (form.end_time <= form.start_time) throw new Error('End time must be later than start time.');
  if (!Number.isInteger(Number(form.participants)) || Number(form.participants) < 1 || Number(form.participants) > 10000) {
    throw new Error('Enter a participant count between 1 and 10,000.');
  }
  if (!form.consent) throw new Error('Please acknowledge how your contact details will be used.');
  return {
    organization: form.organization.trim(), contact_name: form.contact_name.trim(),
    phone: form.phone.trim(), email: form.email.trim() || null,
    title: form.title.trim() || 'Mangrove planting activity',
    start_at: `${form.date}T${form.start_time}:00+08:00`,
    end_at: `${form.date}T${form.end_time}:00+08:00`,
    participants: Number(form.participants), notes: form.notes.trim() || null,
    consent: true, website: form.website, submission_key: submissionKey,
  };
}

export async function submitBooking(payload) {
  const response = await fetch(`${import.meta.env.VITE_API_BASE || ''}/api/public/like/appointments`, {
    method: 'POST', credentials: 'omit', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = result.detail;
    throw new Error(typeof detail === 'string' ? detail : response.status === 422
      ? 'Check your contact details and appointment times, then try again.'
      : 'We could not record your request. Please try again shortly.');
  }
  if (!result.reference) throw new Error('We could not verify your request. Please try again.');
  return result;
}
