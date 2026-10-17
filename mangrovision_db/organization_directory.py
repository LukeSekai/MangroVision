"""Read-only staff directory, assembled from existing organization records."""

from .compat import get_connection


# A reserved participant place is not a member. Include named participants and
# occupied legacy places, even when their session has expired or was revoked.
_MEMBER = """(op.participant_username IS NOT NULL OR op.device_key_hash IS NOT NULL
    OR EXISTS (SELECT 1 FROM auth_sessions ses WHERE ses.subject_type = 'planter'
        AND ses.subject_id = p.id AND ses.participant_slot = op.slot))"""


def _organizations(conn, organization_id=None):
    where = 'WHERE o.id = ?' if organization_id is not None else ''
    rows = conn.execute(f"""
        SELECT o.id, o.name, o.inspection_interval_days,
               latest.contact, latest.notification_email,
               latest.expected_planters AS scheduled_participants,
               latest.id AS contact_activity_id, latest.title AS contact_activity_title,
               request.contact_name AS request_contact_name,
               request.phone AS request_phone, request.email AS request_email,
               COALESCE(accounts.participant_capacity, 0) AS participant_capacity,
               accounts.phone AS account_phone,
               COALESCE(accounts.account_count, 0) AS account_count,
               COALESCE(members.member_count, 0) AS member_count,
               (SELECT COUNT(*) FROM project_sites ps
                WHERE ps.organization_id = o.id AND ps.deleted_at IS NULL) AS project_site_count,
               (SELECT COUNT(*) FROM planting_schedules s
                WHERE s.organization_id = o.id) AS activity_count
        FROM organizations o
        LEFT JOIN LATERAL (
            SELECT s.id, s.contact, s.notification_email, s.expected_planters, s.title
            FROM planting_schedules s WHERE s.organization_id = o.id
            ORDER BY (s.status = 'cancelled'), COALESCE(s.updated_at, s.created_at) DESC, s.id DESC
            LIMIT 1
        ) latest ON TRUE
        LEFT JOIN like_appointment_requests request ON request.schedule_id = latest.id
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS account_count, SUM(p.participant_count) AS participant_capacity,
                   MAX(NULLIF(BTRIM(p.phone), '')) AS phone
            FROM planters p WHERE p.organization_id = o.id AND p.merged_into_planter_id IS NULL
        ) accounts ON TRUE
        LEFT JOIN LATERAL (
            SELECT COUNT(*) AS member_count FROM organization_participants op
            JOIN planters p ON p.id = op.planter_id
            WHERE p.organization_id = o.id AND p.merged_into_planter_id IS NULL AND {_MEMBER}
        ) members ON TRUE
        {where} ORDER BY LOWER(o.name), o.id
    """, (organization_id,) if organization_id is not None else ()).fetchall()
    return [dict(row) for row in rows]


def list_directory():
    conn = get_connection()
    try:
        return _organizations(conn)
    finally:
        conn.close()


def organization_detail(organization_id):
    conn = get_connection()
    try:
        rows = _organizations(conn, organization_id)
        if not rows:
            return None
        sites = conn.execute('''SELECT id, name, notes, inspection_interval_days, deleted_at
            FROM project_sites WHERE organization_id = ?
            ORDER BY (deleted_at IS NOT NULL), LOWER(name), id''', (organization_id,)).fetchall()
        activities = conn.execute('''SELECT s.id, s.title, s.appointment_type, s.start_at, s.end_at,
                s.status, s.expected_planters, s.project_site_id, ps.name AS project_site_name,
                ac.schedule_id IS NOT NULL AS has_code, ac.revoked_at IS NOT NULL AS code_revoked
            FROM planting_schedules s
            LEFT JOIN project_sites ps ON ps.id = s.project_site_id
            LEFT JOIN activity_access_codes ac ON ac.schedule_id = s.id
            WHERE s.organization_id = ? ORDER BY s.start_at DESC, s.id DESC''', (organization_id,)).fetchall()
        members = conn.execute(f'''SELECT p.id AS planter_id, op.slot AS participant_number,
                op.participant_username AS username, p.full_name AS account_name,
                sessions.first_joined_at, sessions.last_seen_at
            FROM organization_participants op JOIN planters p ON p.id = op.planter_id
            LEFT JOIN LATERAL (
                SELECT MIN(s.created_at) AS first_joined_at,
                       MAX(COALESCE(s.last_seen_at, s.created_at)) AS last_seen_at
                FROM auth_sessions s WHERE s.subject_type = 'planter'
                    AND s.subject_id = p.id AND s.participant_slot = op.slot
            ) sessions ON TRUE
            WHERE p.organization_id = ? AND p.merged_into_planter_id IS NULL AND {_MEMBER}
            ORDER BY LOWER(COALESCE(op.participant_username, '')), p.id, op.slot''', (organization_id,)).fetchall()
        return {'organization': rows[0], 'project_sites': [dict(row) for row in sites],
                'activities': [dict(row) for row in activities], 'members': [dict(row) for row in members]}
    finally:
        conn.close()
