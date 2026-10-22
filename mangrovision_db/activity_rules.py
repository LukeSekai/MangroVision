"""Activity protection follows its assignments and recorded planting cycles."""

from datetime import datetime


class ActivityConflict(ValueError):
    """A planning change conflicts with recorded work or an active delivery."""


def complete_fully_planted_activity(conn, schedule_id):
    """Close one activity atomically with its final planted point, never a GET."""
    if schedule_id is None:
        return None
    current = conn.execute('''SELECT id, organization_id, project_site_id, status, appointment_type
        FROM planting_schedules WHERE id = ? FOR UPDATE''', (int(schedule_id),)).fetchone()
    if not current or current['appointment_type'] != 'tree_planting' or current['status'] not in {'confirmed', 'in_progress'}:
        return None
    work = conn.execute('''SELECT COUNT(p.id) AS total_points,
            COUNT(p.id) FILTER (WHERE p.status = 'completed') AS planted_points
        FROM planter_assignments a
        JOIN planter_assignment_points p ON p.assignment_id = a.id
        WHERE a.planting_schedule_id = ?''', (int(schedule_id),)).fetchone()
    if not work['total_points'] or work['planted_points'] != work['total_points']:
        return None
    conn.execute('''UPDATE planting_schedules SET status = 'completed',
        updated_at = CURRENT_TIMESTAMP, updated_by_user_id = NULL WHERE id = ?''', (int(schedule_id),))
    from .activity import append_activity
    append_activity(conn, action='schedule.updated', actor_type='system',
        organization_id=current['organization_id'], project_site_id=current['project_site_id'],
        summary='Completed planting activity: all assigned points are planted.',
        details={'schedule_id': int(schedule_id), 'changed_fields': ['status'],
                 'before': {'status': current['status']}, 'after': {'status': 'completed'},
                 'source': 'all_assigned_points_planted', 'planted_points': int(work['planted_points'])})
    return int(schedule_id)


def activity_work(conn, schedule_id):
    row = conn.execute("""
        SELECT COUNT(DISTINCT pa.id) AS assignments, COUNT(pe.id) AS plantings
        FROM planter_assignments pa
        LEFT JOIN planting_events pe ON pe.assignment_id = pa.id
        WHERE pa.planting_schedule_id = ?
    """, (schedule_id,)).fetchone()
    return {'assignments': int(row['assignments']), 'plantings': int(row['plantings'])}


def _comparable(key, value):
    if key in {'start_at', 'end_at'} and value:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return value


def changed_fields(current, updated):
    return [key for key in (
        'organization_id', 'project_site_id', 'title', 'start_at', 'end_at',
        'expected_planters', 'expected_seedlings', 'inspection_interval_days',
        'status', 'contact', 'notification_email', 'notes',
    ) if _comparable(key, current.get(key)) != _comparable(key, updated.get(key))]


def ensure_activity_editable(current):
    if str(current.get('status') or '').strip().lower() == 'completed':
        raise ActivityConflict('This activity is completed and read-only. It cannot be edited, reopened, cancelled, or deleted.')


def validate_activity_change(conn, current, updated):
    ensure_activity_editable(current)
    work = activity_work(conn, int(current['id']))
    changed = changed_fields(current, updated)
    if work['plantings']:
        frozen = set(changed) - {'contact', 'notification_email', 'notes', 'status'}
        allowed_statuses = {current['status'], 'in_progress', 'completed'}
        if frozen or updated['status'] not in allowed_statuses:
            raise ActivityConflict(
                'Planting has already been recorded for this activity. Its organization, '
                'area, title, dates, and expected counts are locked. You can correct contact '
                'details and notes, or mark the activity completed.')
    elif work['assignments'] and {'organization_id', 'project_site_id'}.intersection(changed):
        raise ActivityConflict('Remove this activity\'s unplanted assignments before changing its organization or planting area.')
    return changed, work


def resolve_assignment_activity(conn, organization_id, site_id, assignment_date, schedule_id=None, *, actor_user_id=None):
    """Explicit selection wins; auto-link only one activity on the assignment date."""
    if schedule_id is None:
        rows = conn.execute("""
            SELECT id FROM planting_schedules
            WHERE organization_id = ? AND (project_site_id = ? OR project_site_id IS NULL)
              AND appointment_type = 'tree_planting' AND status IN ('confirmed', 'in_progress')
              AND (start_at AT TIME ZONE 'Asia/Manila')::date = CAST(? AS date)
            ORDER BY id
        """, (organization_id, site_id, assignment_date)).fetchall()
        if len(rows) > 1:
            raise ActivityConflict('Choose the planting activity: this site has more than one activity on that date.')
        if not rows:
            available = conn.execute("""SELECT 1 FROM planting_schedules
                WHERE organization_id = ? AND (project_site_id = ? OR project_site_id IS NULL) AND appointment_type = 'tree_planting'
                  AND status IN ('confirmed', 'in_progress') LIMIT 1""", (organization_id, site_id)).fetchone()
            if available:
                raise ActivityConflict('Choose the planting activity receiving these points; its date differs from the assignment date.')
            raise ActivityConflict('Schedule and confirm a tree-planting activity for this organization before assigning points.')
        schedule_id = int(rows[0]['id'])
    row = conn.execute('SELECT * FROM planting_schedules WHERE id = ? FOR UPDATE', (int(schedule_id),)).fetchone()
    if not row or row['organization_id'] != organization_id:
        raise ActivityConflict('Choose a planting activity owned by this organization and project site.')
    if row['appointment_type'] != 'tree_planting' or row['status'] not in {'confirmed', 'in_progress'}:
        raise ActivityConflict('Choose a confirmed or ongoing tree-planting activity.')
    if row['project_site_id'] is None:
        site = conn.execute('''SELECT id, organization_id FROM project_sites
            WHERE id = ? AND deleted_at IS NULL AND area_released_at IS NULL FOR SHARE''', (site_id,)).fetchone()
        if not site or site['organization_id'] != organization_id:
            raise ActivityConflict('Choose a planting area owned by this organization.')
        validate_activity_change(conn, dict(row), {**dict(row), 'project_site_id': site_id})
        conn.execute('''UPDATE planting_schedules SET project_site_id = ?,
            updated_by_user_id = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?''',
            (site_id, actor_user_id, int(row['id'])))
        from .activity import append_activity
        append_activity(conn, action='schedule.updated', actor_type='staff' if actor_user_id else 'system',
            actor_user_id=actor_user_id, organization_id=organization_id, project_site_id=site_id,
            summary='Linked planting activity to its assigned project site.',
            details={'schedule_id': int(row['id']), 'changed_fields': ['project_site_id'],
                     'before': {'project_site_id': None}, 'after': {'project_site_id': site_id},
                     'source': 'point_assignment'})
    elif row['project_site_id'] != site_id:
        raise ActivityConflict('Choose a planting activity owned by this organization and project site.')
    from zoneinfo import ZoneInfo
    day = datetime.fromisoformat(row['start_at']).astimezone(ZoneInfo('Asia/Manila')).date().isoformat()
    return int(row['id']), day


def lock_assignment_activity(conn, assignment_id):
    row = conn.execute("""
        SELECT ps.status FROM planting_schedules ps
        JOIN planter_assignments pa ON pa.planting_schedule_id = ps.id
        WHERE pa.id = ? FOR UPDATE OF ps
    """, (assignment_id,)).fetchone()
    if row and row['status'] not in {'confirmed', 'in_progress', 'completed'}:
        raise ActivityConflict('This activity is not confirmed. Contact LGU staff before recording planting.')
    return row['status'] if row else None
