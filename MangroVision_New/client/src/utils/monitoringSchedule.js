const TIMEZONE = 'Asia/Manila';

export function manilaDateKey(value = new Date()) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: TIMEZONE, year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map(({ type, value }) => [type, value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function dayNumber(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value || '')) return null;
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime()) || date.toISOString().slice(0, 10) !== value) return null;
  return date.getTime() / 86_400_000;
}

export function monitoringSchedule(organization, today = manilaDateKey()) {
  const total = Number(organization?.total_planted ?? organization?.current_planted_points ?? 0);
  const dueDay = dayNumber(organization?.next_monitoring_date);
  const currentDay = dayNumber(today);
  if (!Number.isFinite(total) || total <= 0 || dueDay === null || currentDay === null) {
    return { status: 'unscheduled', daysOverdue: 0 };
  }
  if (dueDay > currentDay) return { status: 'upcoming', daysOverdue: 0 };
  // The server's availability flag remains the authority for recording visits.
  if (organization.monitoring_available !== true) return { status: 'unavailable', daysOverdue: 0 };
  return dueDay === currentDay
    ? { status: 'today', daysOverdue: 0 }
    : { status: 'overdue', daysOverdue: currentDay - dueDay };
}

export function monitoringScheduleLabel(schedule) {
  if (schedule.status === 'today') return 'Due today';
  if (schedule.status === 'overdue') return `${schedule.daysOverdue} day${schedule.daysOverdue === 1 ? '' : 's'} overdue`;
  if (schedule.status === 'upcoming') return 'Upcoming';
  if (schedule.status === 'unavailable') return 'Not available';
  return 'Not scheduled';
}

