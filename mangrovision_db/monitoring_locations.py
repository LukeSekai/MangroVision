"""Visit-to-seedling reconciliation and release of dead locations."""
import json
from datetime import datetime, timedelta, time


def _db():
    import planting_database
    return planting_database


def location_summary(conn, record):
    links = conn.execute('''SELECT planting_event_id FROM monitoring_death_locations
        WHERE record_id = ? AND revoked_at IS NULL ORDER BY planting_event_id''', (record['id'],)).fetchall()
    return location_summary_from_ids(record, [row['planting_event_id'] for row in links])


def location_summary_from_ids(record, event_ids):
    reported = record.get('location_death_count')
    return {'reported_dead_count': reported, 'located_dead_count': len(event_ids),
            'unlocated_dead_count': max(0, reported - len(event_ids)) if reported is not None else None,
            'location_review_required': reported is None,
            'dead_planting_event_ids': event_ids}


def _record(conn, record_id, lock=False):
    row = conn.execute('SELECT * FROM organization_monitoring_records WHERE id = ?' +
                       (' FOR UPDATE' if lock else ''), (record_id,)).fetchone()
    if not row:
        raise ValueError('Monitoring visit was not found.')
    return dict(row)


def _points(conn, organization_id, monitored_at, record_id=None):
    db = _db()
    observed = db._parse_dashboard_datetime(monitored_at) if monitored_at else db._manila_now()
    if not observed or observed.date() > db._manila_now().date():
        raise ValueError('Choose a valid inspection date that is not in the future.')
    cutoff = datetime.combine(observed.date() + timedelta(days=1), time.min, tzinfo=observed.tzinfo)
    # Snapshot coordinates and species identify the plant observed, including
    # older cycles when reconciling an earlier visit after replacement.
    rows = conn.execute('''SELECT pe.id AS planting_event_id, pe.planting_point_id,
        pe.assignment_id, pe.site_zone_id AS project_site_id, pe.species, pe.planted_at,
        pe.latitude, pe.longitude, pe.point_num, pe.closed_at, pe.closure_reason,
        a.image_name AS analysis_name, a.id AS analysis_id, s.name AS project_site_name,
        pa.title AS assignment_title, pp.deleted_at,
        pdr.id AS death_record_id, pdr.death_at,
        mdl.record_id AS linked_record_id,
        CASE WHEN pp.deleted_at IS NULL AND (mdl.record_id IS NULL OR mdl.record_id = ?)
            AND NOT EXISTS (SELECT 1 FROM planting_events newer
                WHERE newer.planting_point_id = pe.planting_point_id
                AND (newer.planted_at, newer.id) > (pe.planted_at, pe.id)
                AND newer.planted_at < ? AND COALESCE(newer.closure_reason, '') <> 'completion_reversed')
            THEN TRUE ELSE FALSE END AS selectable
        FROM planting_events pe JOIN planters pl ON pl.id = pe.planter_id
        JOIN planting_points pp ON pp.id = pe.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        LEFT JOIN site_zones s ON s.id = pe.site_zone_id
        LEFT JOIN planter_assignments pa ON pa.id = pe.assignment_id
        LEFT JOIN point_death_records pdr ON pdr.planting_event_id = pe.id
        LEFT JOIN monitoring_death_locations mdl ON mdl.planting_event_id = pe.id AND mdl.revoked_at IS NULL
        WHERE pl.organization_id = ? AND pe.planted_at < ?
            AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
            AND NOT EXISTS (SELECT 1 FROM planting_events newer
                WHERE newer.planting_point_id = pe.planting_point_id
                AND (newer.planted_at, newer.id) > (pe.planted_at, pe.id)
                AND newer.planted_at < ? AND COALESCE(newer.closure_reason, '') <> 'completion_reversed')
        ORDER BY a.analysis_number, pe.point_num, pe.planted_at''',
        (record_id, cutoff.isoformat(), organization_id, min(cutoff, db._manila_now()).isoformat(), cutoff.isoformat())).fetchall()
    points = [dict(row) for row in rows]
    for point in points:
        point['reference'] = f"{point['analysis_name']} · Point {point['point_num']}"
    previous = conn.execute('''SELECT monitored_at FROM organization_monitoring_records
        WHERE organization_id = ? AND monitored_at < ? ORDER BY monitored_at DESC, id DESC LIMIT 1''',
        (organization_id, observed.isoformat())).fetchone()
    if previous:
        previous_date = db._parse_dashboard_datetime(previous['monitored_at']).date()
        for point in points:
            if point['death_at'] and (not point['linked_record_id'] or point['linked_record_id'] != record_id) and db._parse_dashboard_datetime(point['death_at']).date() <= previous_date:
                point['selectable'] = False
    return points


def monitoring_points(organization_id, monitored_at=None, record_id=None):
    conn = _db()._get_connection()
    try:
        if record_id:
            record = _record(conn, record_id)
            if record['organization_id'] != organization_id:
                raise ValueError('This visit belongs to a different organization.')
            monitored_at = record['monitored_at']
        return _points(conn, organization_id, monitored_at, record_id)
    finally:
        conn.close()


def clean_event_ids(values):
    try:
        if any(isinstance(value, bool) or str(int(value)) != str(value).strip() for value in (values or [])):
            raise ValueError('Planting event IDs must be whole numbers.')
        ids = sorted(set(int(value) for value in (values or [])))
    except (TypeError, ValueError) as error:
        raise ValueError('Planting event IDs must be whole numbers.') from error
    if any(value <= 0 for value in ids):
        raise ValueError('Planting event IDs must be positive.')
    return ids


def reconcile_on_connection(conn, record, event_ids, user_id, correction_note=''):
    """Caller locks the organization then visit; all edits use this transaction."""
    db = _db()
    requested = set(clean_event_ids(event_ids))
    if record.get('location_death_count') is None:
        raise ValueError('This older visit has inconsistent death counts and needs review first.')
    if len(requested) > record['location_death_count']:
        raise ValueError('Located deaths cannot exceed the deaths reported in this visit.')
    snapshot = db._monitoring_json(record.get('count_snapshot')) or {}
    cause = snapshot.get('death_reason_category') or 'unknown'
    cause_label = db.DEATH_REASON_CATEGORIES.get(cause, 'Not determined')
    cause_notes = snapshot.get('death_reason_notes') or ''
    if snapshot.get('event_ids') is not None and not requested.issubset(set(snapshot['event_ids'])):
        raise ValueError('These seedlings were not included in the saved visit. Reload the planting locations for its inspection date.')
    existing = {int(row['planting_event_id']): dict(row) for row in conn.execute('''
        SELECT * FROM monitoring_death_locations WHERE record_id = ? AND revoked_at IS NULL
        FOR UPDATE''', (record['id'],)).fetchall()}
    removed, added = set(existing) - requested, requested - set(existing)
    if removed and not correction_note.strip():
        raise ValueError('Explain the location correction before removing a link.')
    now = db._manila_now().isoformat()
    changed_ids = sorted(removed | added)
    if changed_ids:
        marks = ','.join('?' for _ in changed_ids)
        conn.execute(f'''SELECT id FROM planting_points WHERE id IN (
            SELECT planting_point_id FROM planting_events WHERE id IN ({marks})) ORDER BY id FOR UPDATE''', changed_ids).fetchall()
    for event_id in sorted(removed | added):
        conn.execute('SELECT id FROM planting_events WHERE id = ? FOR UPDATE', (event_id,)).fetchone()
    candidates = {row['planting_event_id']: row for row in _points(
        conn, record['organization_id'], record['monitored_at'], record['id'])}
    for event_id in removed:
        link = existing[event_id]
        if conn.execute('SELECT 1 FROM replanting_requests WHERE planting_event_id = ?', (event_id,)).fetchone():
            raise ValueError('A released location prevents changing this death record.')
        if conn.execute('SELECT 1 FROM planting_events WHERE replaces_event_id = ?', (event_id,)).fetchone():
            raise ValueError('A completed replacement prevents changing this location.')
        if link['created_death']:
            if conn.execute('SELECT 1 FROM monitoring_observations WHERE planting_event_id = ?', (event_id,)).fetchone():
                raise ValueError('Point inspection history prevents changing this death location.')
            conn.execute('DELETE FROM point_death_records WHERE id = ?', (link['death_record_id'],))
            snapshot = db._monitoring_json(link['snapshot'])
            conn.execute('UPDATE planting_events SET closed_at = ?, closure_reason = ? WHERE id = ?',
                         (snapshot.get('closed_at'), snapshot.get('closure_reason'), event_id))
            conn.execute('''UPDATE planting_points SET death_at = NULL, death_reason = NULL,
                death_reason_category = NULL, death_notes = NULL
                WHERE id = ? AND NOT EXISTS (SELECT 1 FROM planting_events WHERE planting_point_id = ? AND id > ?)''',
                (snapshot['planting_point_id'], snapshot['planting_point_id'], event_id))
        conn.execute('''UPDATE monitoring_death_locations SET revoked_at = ?, revoked_by = ?, correction_note = ?
            WHERE id = ?''', (now, user_id, correction_note.strip(), link['id']))
    for event_id in sorted(added):
        point = candidates.get(event_id)
        if not point or not point['selectable']:
            raise ValueError('Choose a planted seedling from this organization and inspection date, not an already counted death.')
        # Recheck after locking: another visit may have linked the event.
        if conn.execute('SELECT 1 FROM monitoring_death_locations WHERE planting_event_id = ? AND revoked_at IS NULL',
                        (event_id,)).fetchone():
            raise ValueError('This death location was already linked to another visit. Reload the form.')
        death = conn.execute('SELECT * FROM point_death_records WHERE planting_event_id = ?', (event_id,)).fetchone()
        created = death is None
        if created:
            death_at = max(db._parse_dashboard_datetime(record['monitored_at']),
                           db._parse_dashboard_datetime(point['planted_at'])).isoformat()
            cur = conn.execute('''INSERT INTO point_death_records (planting_point_id, assignment_id,
                planting_event_id, death_at, reason_category, reason_label, notes, planter_id, species)
                SELECT planting_point_id, assignment_id, id, ?, ?, ?, ?, planter_id, species
                FROM planting_events WHERE id = ?''',
                (death_at, cause, cause_label, cause_notes, event_id))
            death_id = cur.lastrowid
            conn.execute("UPDATE planting_events SET closed_at = ?, closure_reason = 'recorded_death' WHERE id = ?",
                         (death_at, event_id))
            conn.execute('''UPDATE planting_points SET death_at = ?, death_reason = ?,
                death_reason_category = ?, death_notes = ? WHERE id = ? AND NOT EXISTS (
                    SELECT 1 FROM planting_events WHERE planting_point_id = ? AND id > ?)''',
                (death_at, cause_label, cause, cause_notes, point['planting_point_id'], point['planting_point_id'], event_id))
        else:
            death_id = death['id']
        conn.execute('''INSERT INTO monitoring_death_locations (record_id, planting_event_id, death_record_id,
            created_death, snapshot, created_by, correction_note) VALUES (?, ?, ?, ?, CAST(? AS jsonb), ?, ?)''',
            (record['id'], event_id, death_id, created, json.dumps(point), user_id, correction_note.strip() or None))
    if added or removed:
        conn.execute('UPDATE organization_monitoring_records SET location_version = location_version + 1 WHERE id = ?',
                     (record['id'],))


def reconcile_locations(record_id, event_ids, expected_version, user_id, correction_note=''):
    db = _db()
    conn = db._get_connection()
    try:
        record = _record(conn, record_id)
        conn.execute('SELECT id FROM organizations WHERE id = ? FOR UPDATE', (record['organization_id'],)).fetchone()
        record = _record(conn, record_id, lock=True)
        if record['location_version'] != expected_version:
            raise ValueError('Locations changed. Reload this visit before saving.')
        reconcile_on_connection(conn, record, event_ids, user_id, correction_note)
        result = db._organization_monitoring_record_row(conn, record_id)
        conn.commit()
        return result
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_release_candidates(organization_id=None, project_site_id=None):
    conn = _db()._get_connection()
    try:
        rows = conn.execute('''SELECT pe.id AS planting_event_id, pe.planting_point_id,
            pe.latitude, pe.longitude, pe.point_num, pe.species, pe.site_zone_id AS project_site_id,
            pe.planted_at, pl.organization_id,
            COALESCE(o.name, CASE WHEN pe.planted_by_user_id IS NOT NULL THEN 'LGU' END) AS organization_name,
            s.name AS project_site_name,
            a.image_name AS analysis_name, pdr.death_at, r.id AS request_id, COALESCE(r.version, 0) AS version,
            pp.deleted_at AS point_deleted_at, pp.death_at AS point_death_at, pp.status AS point_status,
            r.approved_at, approver.full_name AS approved_by_name,
            replacement_event.planted_at AS replacement_planted_at,
            replacement_staff.full_name AS replacement_staff_name,
            EXISTS (SELECT 1 FROM map_zones mz WHERE mz.zone_type = 'eroded' AND mz.deleted_at IS NULL
                AND extensions.ST_Covers(mz.geometry, pp.location)) AS eroded_unavailable,
            r.assignment_id, r.replacement_event_id, pa.status AS assignment_status,
            CASE WHEN r.replacement_event_id IS NOT NULL OR EXISTS (SELECT 1 FROM planting_events n
                WHERE n.planting_point_id = pe.planting_point_id AND (n.planted_at, n.id) > (pe.planted_at, pe.id)
                AND COALESCE(n.closure_reason, '') <> 'completion_reversed') THEN 'completed'
                WHEN r.assignment_id IS NOT NULL THEN 'assigned'
                WHEN r.id IS NOT NULL THEN 'released' ELSE 'awaiting_release' END AS release_status
            FROM point_death_records pdr JOIN planting_events pe ON pe.id = pdr.planting_event_id
            JOIN planting_points pp ON pp.id = pe.planting_point_id JOIN analyses a ON a.id = pp.analysis_id
            LEFT JOIN planters pl ON pl.id = pe.planter_id LEFT JOIN organizations o ON o.id = pl.organization_id
            LEFT JOIN site_zones s ON s.id = pe.site_zone_id
            LEFT JOIN replanting_requests r ON r.planting_event_id = pe.id
            LEFT JOIN planter_assignments pa ON pa.id = r.assignment_id
            LEFT JOIN users approver ON approver.id = r.approved_by
            LEFT JOIN planting_events replacement_event ON replacement_event.id = r.replacement_event_id
            LEFT JOIN users replacement_staff ON replacement_staff.id = replacement_event.planted_by_user_id
            WHERE (CAST(? AS BIGINT) IS NULL OR pl.organization_id = ?)
              AND (CAST(? AS BIGINT) IS NULL OR pe.site_zone_id = ?)
            ORDER BY pdr.death_at DESC, pe.id''',
            (organization_id, organization_id, project_site_id, project_site_id)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            blocked = ('Location was removed' if item['point_deleted_at'] else
                       'Currently unavailable because of erosion' if item['eroded_unavailable'] else '')
            item['can_release'] = (not blocked and item['release_status'] == 'awaiting_release'
                                   and bool(item['point_death_at']))
            item['unavailable_reason'] = blocked
            result.append(item)
        return result
    finally:
        conn.close()


def release_dead_locations_batch(event_ids, versions, user_id, project_site_id):
    from .site_actions import release_dead_locations
    if isinstance(project_site_id, bool) or not isinstance(project_site_id, int) or project_site_id <= 0:
        raise ValueError('Choose a project site before releasing locations.')
    return release_dead_locations(event_ids, versions, user_id, project_site_id=project_site_id)
