"""Visit-to-seedling reconciliation and reviewed replacement planting."""
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
            raise ValueError('An approved or assigned replacement prevents changing this location.')
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


def list_replanting(organization_id=None):
    conn = _db()._get_connection()
    try:
        rows = conn.execute('''SELECT pe.id AS planting_event_id, pe.planting_point_id,
            pe.latitude, pe.longitude, pe.point_num, pe.species, pe.site_zone_id AS project_site_id,
            pe.planted_at, pl.organization_id,
            COALESCE(o.name, CASE WHEN pe.planted_by_user_id IS NOT NULL THEN 'LGU' END) AS organization_name,
            s.name AS project_site_name,
            a.image_name AS analysis_name, pdr.death_at, r.id AS request_id, COALESCE(r.version, 0) AS version,
            r.assignment_id, r.replacement_event_id, pa.status AS assignment_status,
            CASE WHEN r.replacement_event_id IS NOT NULL OR EXISTS (SELECT 1 FROM planting_events n
                WHERE n.planting_point_id = pe.planting_point_id AND n.id > pe.id
                AND COALESCE(n.closure_reason, '') <> 'completion_reversed') THEN 'completed'
                WHEN r.assignment_id IS NOT NULL THEN 'assigned'
                WHEN r.id IS NOT NULL THEN 'approved' ELSE 'awaiting_review' END AS replanting_status
            FROM point_death_records pdr JOIN planting_events pe ON pe.id = pdr.planting_event_id
            JOIN planting_points pp ON pp.id = pe.planting_point_id JOIN analyses a ON a.id = pp.analysis_id
            LEFT JOIN planters pl ON pl.id = pe.planter_id LEFT JOIN organizations o ON o.id = pl.organization_id
            LEFT JOIN site_zones s ON s.id = pe.site_zone_id
            LEFT JOIN replanting_requests r ON r.planting_event_id = pe.id
            LEFT JOIN planter_assignments pa ON pa.id = r.assignment_id
            WHERE (CAST(? AS BIGINT) IS NULL OR pl.organization_id = ?) ORDER BY pdr.death_at DESC, pe.id''',
            (organization_id, organization_id)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _safe_point(conn, point_id):
    db = _db()
    if db._first_deleted_point(conn, [point_id]):
        raise ValueError('This planting location was removed and cannot be replanted.')
    if db._first_eroded_unavailable_point(conn, [point_id]):
        raise ValueError('This location is currently unavailable because of erosion.')


def approve_replanting(event_id, expected_version, user_id, connection=None):
    db = _db()
    conn = connection or db._get_connection()
    try:
        conn.execute('''SELECT id FROM planting_points WHERE id =
            (SELECT planting_point_id FROM planting_events WHERE id = ?) FOR UPDATE''', (event_id,)).fetchone()
        event = conn.execute('SELECT * FROM planting_events WHERE id = ? FOR UPDATE', (event_id,)).fetchone()
        if not event:
            raise ValueError('Planting event was not found.')
        request = conn.execute('SELECT * FROM replanting_requests WHERE planting_event_id = ? FOR UPDATE', (event_id,)).fetchone()
        if expected_version != (request['version'] if request else 0):
            raise ValueError('Replacement work changed. Reload before approving.')
        if request:
            return dict(request)
        point_id = event['planting_point_id']
        point = conn.execute('SELECT * FROM planting_points WHERE id = ? FOR UPDATE', (point_id,)).fetchone()
        if not point or not point['death_at'] or not conn.execute(
            'SELECT 1 FROM point_death_records WHERE planting_event_id = ?', (event_id,)).fetchone():
            raise ValueError('Only a confirmed dead seedling can be prepared for replanting.')
        if conn.execute('SELECT 1 FROM planting_events WHERE planting_point_id = ? AND id > ?', (point_id, event_id)).fetchone():
            raise ValueError('A newer planting cycle already exists at this location.')
        _safe_point(conn, point_id)
        conn.execute('INSERT INTO replanting_requests (planting_event_id, approved_by) VALUES (?, ?)', (event_id, user_id))
        conn.execute('UPDATE planter_assignment_points SET released_at = ?, released_by = ? WHERE planting_point_id = ? AND released_at IS NULL',
                     (db._manila_now().isoformat(), user_id, point_id))
        conn.execute("""UPDATE planting_points SET status = 'planned', planted_at = NULL, planted_date = NULL,
            death_at = NULL, death_reason = NULL, death_reason_category = NULL, death_notes = NULL WHERE id = ?""", (point_id,))
        conn.commit()
        return {'planting_event_id': event_id, 'status': 'approved', 'version': 1}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


class _BorrowedConnection:
    def __init__(self, conn):
        self.conn = conn

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def commit(self):
        pass

    def close(self):
        pass


def approve_replanting_batch(event_ids, versions, user_id):
    """Release the selected locations atomically, keeping every death in history."""
    ids = clean_event_ids(event_ids)
    if not ids or len(ids) > 500:
        raise ValueError('Select between 1 and 500 dead seedlings for approval.')
    if any(str(event_id) not in versions for event_id in ids):
        raise ValueError('Reload the selected locations before approving.')
    conn = _db()._get_connection()
    try:
        marks = ','.join('?' for _ in ids)
        # All approval paths lock physical locations before their planting events.
        conn.execute(f'''SELECT id FROM planting_points WHERE id IN
            (SELECT planting_point_id FROM planting_events WHERE id IN ({marks}))
            ORDER BY id FOR UPDATE''', ids).fetchall()
        results = [approve_replanting(event_id, versions[str(event_id)], user_id,
                                     connection=_BorrowedConnection(conn)) for event_id in ids]
        conn.commit()
        return {'approved_count': len(results), 'planting_event_ids': ids}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def assign_replanting(event_ids, versions, organization_id, assignment_date, user_id):
    db = _db()
    ids = clean_event_ids(event_ids)
    if not ids or not organization_id:
        raise ValueError('Select replacement locations and an organization.')
    try:
        datetime.strptime(assignment_date, '%Y-%m-%d')
    except (TypeError, ValueError) as error:
        raise ValueError('Choose a valid planting work date.') from error
    conn = db._get_connection()
    try:
        planter = conn.execute("""SELECT * FROM planters WHERE organization_id = ? AND status = 'active'
            AND merged_into_planter_id IS NULL ORDER BY id LIMIT 1 FOR UPDATE""", (organization_id,)).fetchone()
        if not planter:
            raise ValueError('Choose an organization with an active field account.')
        marks = ','.join('?' for _ in ids)
        conn.execute(f'''SELECT id FROM planting_points WHERE id IN
            (SELECT planting_point_id FROM planting_events WHERE id IN ({marks})) ORDER BY id FOR UPDATE''', ids).fetchall()
        groups = {}
        for event_id in ids:
            request = conn.execute('''SELECT r.*, pe.planting_point_id, pe.site_zone_id, pe.species
                FROM replanting_requests r JOIN planting_events pe ON pe.id = r.planting_event_id
                WHERE r.planting_event_id = ? FOR UPDATE OF r''', (event_id,)).fetchone()
            if not request or request['assignment_id'] or request['replacement_event_id']:
                raise ValueError('Only approved, unassigned locations can be assigned.')
            if request['version'] != versions.get(str(event_id)):
                raise ValueError('Replacement work changed. Reload before assigning.')
            _safe_point(conn, request['planting_point_id'])
            groups.setdefault((request['site_zone_id'], request['species']), []).append(dict(request))
        assignments = []
        for (site_id, species), requests in groups.items():
            assignment_id = db.create_planter_assignment(planter['id'], [r['planting_point_id'] for r in requests],
                assigned_by_user_id=user_id, title='Replacement planting', assignment_date=assignment_date,
                species=species or '', site_zone_id=site_id, connection=_BorrowedConnection(conn))
            assignments.append(assignment_id)
        conn.commit()
        return {'assignment_ids': assignments}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
