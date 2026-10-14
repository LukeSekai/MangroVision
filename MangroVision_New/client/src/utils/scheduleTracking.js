const TIMEZONE = 'Asia/Manila';

export const SCHEDULE_TRACKING_GROUPS = [
  { key: 'completed', label: 'Completed', description: 'Activities marked completed', empty: 'No completed schedules' },
  { key: 'pending', label: 'Pending', description: 'Activities awaiting approval or completion', empty: 'No pending schedules' },
  { key: 'cancelled', label: 'Cancelled', description: 'Activities that were cancelled', empty: 'No cancelled schedules' },
];

export function scheduleProgressStatus(status) {
  const value = String(status || '').trim().toLowerCase();
  return ['completed', 'cancelled'].includes(value) ? value : 'pending';
}

// Progress and approval are separate: editing a pending activity must retain
// its existing approval and eligibility for planting.
export function scheduleStatusForProgress(progress, currentStatus = 'requested') {
  const current = String(currentStatus || 'requested').trim().toLowerCase();
  if (current === 'completed') return 'completed';
  if (progress !== 'pending') return progress;
  return current === 'cancelled' ? 'requested' : current;
}

function manilaDate(value) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: TIMEZONE, year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(new Date(value));
  const date = Object.fromEntries(parts.map(({ type, value: part }) => [type, part]));
  return `${date.year}-${date.month}-${date.day}`;
}

function hasEnded(schedule, now) {
  if (schedule.end_at) {
    const end = Date.parse(schedule.end_at);
    return Number.isFinite(end) && end < now;
  }
  const day = schedule.scheduled_date || schedule.date;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day || '')) return false;
  const midnight = Date.parse(`${day}T00:00:00+08:00`);
  if (!Number.isFinite(midnight) || manilaDate(midnight) !== day) return false;
  if (schedule.end_time) {
    const end = Date.parse(`${day}T${schedule.end_time}+08:00`);
    return Number.isFinite(end) && end < now;
  }
  // Without an end time, only a previous Philippine date is known to have ended.
  return day < manilaDate(now);
}

export function scheduleTrackingState(schedule, now = Date.now()) {
  const status = String(schedule?.status || '').trim().toLowerCase();
  const progress = scheduleProgressStatus(status);
  const currentTime = new Date(now).getTime();
  return {
    completed: progress === 'completed',
    pending: progress === 'pending',
    cancelled: progress === 'cancelled',
    needs_confirmation: ['requested', 'tentative'].includes(status),
    approved: ['confirmed', 'in_progress', 'completed'].includes(status),
    missed: ['requested', 'tentative', 'confirmed', 'in_progress'].includes(status)
      && Number.isFinite(currentTime) && hasEnded(schedule, currentTime),
  };
}
