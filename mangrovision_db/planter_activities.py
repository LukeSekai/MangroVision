"""Organization-scoped activity details for signed-in participant devices."""

import json

from .compat import get_connection


def list_planter_activities(organization_id):
    if organization_id is None:
        return []
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT s.id, s.organization_id, s.title, s.appointment_type, s.start_at, s.end_at,
                   s.status, s.expected_planters, s.updated_at, sz.name AS project_site_name,
                   change.details AS change_details, change.created_at AS changed_at
            FROM planting_schedules s
            LEFT JOIN site_zones sz ON sz.id = s.project_site_id
            LEFT JOIN LATERAL (
                SELECT details, created_at FROM activity_logs
                WHERE action = 'schedule.updated' AND organization_id = s.organization_id
                  AND details->>'schedule_id' = CAST(s.id AS text)
                  AND jsonb_exists_any(details->'changed_fields', ARRAY['start_at', 'end_at', 'status', 'project_site_id'])
                ORDER BY created_at DESC, id DESC LIMIT 1
            ) change ON TRUE
            WHERE s.organization_id = ? AND s.status IN ('confirmed', 'in_progress', 'cancelled')
              AND (s.end_at >= CURRENT_TIMESTAMP OR s.updated_at >= CURRENT_TIMESTAMP - INTERVAL '30 days')
            ORDER BY CASE WHEN s.end_at >= CURRENT_TIMESTAMP THEN 0 ELSE 1 END, s.start_at, s.id
            LIMIT 50
        """, (int(organization_id),)).fetchall()
        activities = []
        for row in rows:
            activity = {key: row[key] for key in (
                'id', 'title', 'appointment_type', 'start_at', 'end_at', 'status',
                'expected_planters', 'updated_at', 'project_site_name',
            )}
            details = row['change_details']
            if isinstance(details, str):
                details = json.loads(details)
            fields = (details or {}).get('changed_fields') or []
            before = (details or {}).get('before') or {}
            activity['change_notice'] = {
                'changed_at': row['changed_at'],
                'rescheduled': bool({'start_at', 'end_at'} & set(fields)),
                'previous_start_at': before.get('start_at'), 'previous_end_at': before.get('end_at'),
            } if details else None
            activities.append(activity)
        return activities
    finally:
        conn.close()
