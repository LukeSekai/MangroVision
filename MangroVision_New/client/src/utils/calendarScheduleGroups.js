function normalizedText(value) {
  return String(value ?? '').trim().replace(/\s+/g, ' ').toLowerCase();
}

// Combine matching calendar entries without merging their organization records.
export function groupCalendarSchedules(schedules = []) {
  const groups = new Map();
  schedules.forEach((schedule, index) => {
    const date = schedule.scheduled_date;
    const title = normalizedText(schedule.title);
    const start = Date.parse(schedule.start_at || `${date}T${schedule.start_time}+08:00`);
    const end = Date.parse(schedule.end_at || `${date}T${schedule.end_time}+08:00`);
    const key = date && title && Number.isFinite(start) && Number.isFinite(end)
      ? JSON.stringify([date, title, start, end, normalizedText(schedule.status)])
      : JSON.stringify(['schedule', schedule.id ?? index]);
    if (!groups.has(key)) groups.set(key, { key, schedule, schedules: [] });
    groups.get(key).schedules.push(schedule);
  });
  return [...groups.values()].map((group) => {
    const organizations = new Set(group.schedules.map((schedule) => (
      schedule.organization_id !== null && schedule.organization_id !== undefined
        ? `id:${schedule.organization_id}`
        : normalizedText(schedule.organization_name)
    )).filter(Boolean));
    return { ...group, organizationCount: organizations.size };
  });
}
