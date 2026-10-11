// The existing schedule stores a contact summary and a separate email address.
// Parse the known name/phone/email formats without changing untouched legacy text.
export function scheduleContactFields(schedule = {}) {
  const parts = String(schedule.contact || '').split(',').map((part) => part.trim());
  let email = schedule.notification_email || '';
  if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(parts.at(-1) || '')) {
    const embeddedEmail = parts.pop();
    email ||= embeddedEmail;
  }
  const last = parts.at(-1) || '';
  const phone = last.replace(/\D/g, '').length >= 7 && /^[+()\d\s.-]+$/.test(last) ? parts.pop() : '';
  return { contactName: parts.join(', '), phone, email };
}

export const scheduleContactSummary = (name, phone) => [name.trim(), phone.trim()].filter(Boolean).join(', ') || null;
