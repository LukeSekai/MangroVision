// Presentation aliases only. API and database status values remain unchanged.
export const POINT_STATUS_LABELS = {
  planned: 'Planned', not_assigned: 'Planned',
  assigned: 'Assigned', pending: 'Assigned',
  planted: 'Planted', completed: 'Planted',
  dead: 'Dead', skipped: 'Skipped',
  unavailable: 'Unavailable', eroded_unavailable: 'Unavailable',
  deleted: 'Removed',
};

export const POINT_STATUS_COLORS = {
  planned: '#16a34a', assigned: '#2563eb', pending: '#2563eb',
  planted: '#eab308', completed: '#eab308', dead: '#7f1d1d',
  skipped: '#9ca3af', unavailable: '#f97316', eroded_unavailable: '#f97316',
};

export function pointStatusLabel(status) {
  return POINT_STATUS_LABELS[status] || 'Status unknown';
}

export const ASSIGNMENT_STATUS_LABELS = {
  active: 'Active', completed: 'Completed', archived: 'Archived',
};
