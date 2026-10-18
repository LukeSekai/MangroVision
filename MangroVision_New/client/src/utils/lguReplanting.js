export const REPLANTING_STAGES = [
  ['awaiting_review', 'Awaiting review'], ['ready_lgu', 'LGU queue'], ['completed', 'Replacements planted'],
];

export function replantingProgress(rows) {
  const retained = rows.filter((row) => !row.point_deleted_at);
  return {
    awaiting_review: retained.filter((row) => row.replanting_status === 'awaiting_review').length,
    ready_lgu: retained.filter((row) => row.replanting_status === 'ready_lgu' && row.can_record_replanting).length,
    replacements_completed: rows.filter((row) => row.replanting_status === 'completed').length,
  };
}

export function replantingRows(rows, { stage = 'ready_lgu', site = '', search = '' } = {}) {
  const query = search.trim().toLowerCase();
  return rows.filter((row) => row.replanting_status === stage
    && (stage === 'completed' || !row.point_deleted_at)
    && (!site || String(row.project_site_id) === String(site))
    && (!query || `${row.analysis_name} ${row.point_num} ${row.project_site_name} ${row.organization_name}`.toLowerCase().includes(query)));
}

export function replantingVersions(rows) {
  return Object.fromEntries(rows.map((row) => [row.planting_event_id, row.version]));
}

export function replantingStatusLabel(row) {
  if (row.unavailable_reason && row.replanting_status !== 'completed') return row.unavailable_reason;
  return { awaiting_review: 'Awaiting review', ready_lgu: 'Assigned — LGU replanting',
    assigned: 'Assigned to organization', completed: 'Replacement planted' }[row.replanting_status] || 'Status unavailable';
}
