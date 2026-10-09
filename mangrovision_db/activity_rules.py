"""Activity protection follows its assignments and recorded planting cycles."""

from datetime import datetime


class ActivityConflict(ValueError):
    """A planning change conflicts with recorded work or an active delivery."""


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
        'status', 'contact', 'notes',
    ) if _comparable(key, current.get(key)) != _comparable(key, updated.get(key))]


def validate_activity_change(conn, current, updated):
    work = activity_work(conn, int(current['id']))
    changed = changed_fields(current, updated)
    if work['plantings']:
        frozen = set(changed) - {'contact', 'notes', 'status'}
        allowed_statuses = {current['status'], 'in_progress', 'completed'}
        if current['status'] == 'completed':
            allowed_statuses = {'completed'}
        if frozen or updated['status'] not in allowed_statuses:
            raise ActivityConflict(
                'Planting has already been recorded for this activity. Its organization, '
                'area, title, dates, and expected counts are locked. You can correct contact '
                'details and notes, or mark the activity completed.')
    elif work['assignments'] and {'organization_id', 'project_site_id'}.intersection(changed):
        raise ActivityConflict('Remove this activity\'s unplanted assignments before changing its organization or planting area.')
    return changed, work


def resolve_assignment_activity(conn, organization_id, site_id, assignment_date, schedule_id=None):
    """Explicit selection wins; auto-link only one activity on the assignment date."""
    if schedule_id is None:
        rows = conn.execute("""
            SELECT id FROM planting_schedules
            WHERE organization_id = ? AND project_site_id = ?
              AND appointment_type = 'tree_planting' AND status IN ('confirmed', 'in_progress')
              AND (start_at AT TIME ZONE 'Asia/Manila')::date = CAST(? AS date)
            ORDER BY id
        """, (organization_id, site_id, assignment_date)).fetchall()
        if len(rows) > 1:
            raise ActivityConflict('Choose the planting activity: this site has more than one activity on that date.')
        if not rows:
            available = conn.execute("""SELECT 1 FROM planting_schedules
                WHERE organization_id = ? AND project_site_id = ? AND appointment_type = 'tree_planting'
                  AND status IN ('confirmed', 'in_progress') LIMIT 1""", (organization_id, site_id)).fetchone()
            if available:
                raise ActivityConflict('Choose the planting activity receiving these points; its date differs from the assignment date.')
            return None, assignment_date
        schedule_id = int(rows[0]['id'])
    row = conn.execute('SELECT * FROM planting_schedules WHERE id = ? FOR UPDATE', (int(schedule_id),)).fetchone()
    if not row or row['organization_id'] != organization_id or row['project_site_id'] != site_id:
        raise ActivityConflict('Choose a planting activity owned by this organization and project site.')
    if row['appointment_type'] != 'tree_planting' or row['status'] not in {'confirmed', 'in_progress'}:
        raise ActivityConflict('Choose a confirmed or ongoing tree-planting activity.')
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
