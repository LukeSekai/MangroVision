"""LGU replacement work reserved by an approved replanting request."""
from uuid import uuid4

from .activity import append_activity
from .seedlings import _details, _planting_timestamp


def pending_lgu_requests(conn, point_ids=None):
    params = []
    selected = ''
    if point_ids is not None:
        ids = list(dict.fromkeys(int(value) for value in point_ids))
        if not ids:
            return []
        selected = ' AND pe.planting_point_id IN (' + ','.join('?' for _ in ids) + ')'
        params = ids
    return [dict(row) for row in conn.execute('''SELECT
        r.id AS replanting_request_id, r.planting_event_id AS replanting_event_id,
        r.approved_at AS replanting_approved_at, r.version AS replanting_version,
        pe.planting_point_id, pe.site_zone_id AS replanting_site_id,
        pe.species AS replanting_species
        FROM replanting_requests r JOIN planting_events pe ON pe.id = r.planting_event_id
        JOIN planting_points pp ON pp.id = pe.planting_point_id
        WHERE r.assignment_id IS NULL AND r.replacement_event_id IS NULL
          AND pp.deleted_at IS NULL AND pp.status = 'planned' AND pp.death_at IS NULL
          AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
          AND EXISTS (SELECT 1 FROM point_death_records death WHERE death.planting_event_id = pe.id)
          AND NOT EXISTS (SELECT 1 FROM planting_events newer
              WHERE newer.planting_point_id = pe.planting_point_id
                AND (newer.planted_at, newer.id) > (pe.planted_at, pe.id)
                AND COALESCE(newer.closure_reason, '') <> 'completion_reversed')
        ''' + selected, params).fetchall()]


def annotate_lgu_replanting(conn, points):
    """Keep the physical Planned value; expose its reserved LGU assignment."""
    by_point = {row['planting_point_id']: row for row in pending_lgu_requests(conn)}
    for point in points:
        reservation = by_point.get(point['id'])
        if reservation:
            point.update(reservation, lgu_replanting_pending=True, assigned_team_name='LGU',
                         assignment_title='LGU replanting', assigned_at=reservation['replanting_approved_at'])
            # Dashboard dimension filters need the original site/species while
            # the location has no organization allocation. Map ownership is
            # still resolved independently from current project boundaries.
            if 'site_id' in point:
                point['site_id'] = reservation['replanting_site_id']
                point['species'] = reservation['replanting_species']
    return points


def lgu_work_history(conn):
    """Task dates for operations reporting; no fake organization allocations."""
    return [dict(row) for row in conn.execute('''SELECT r.planting_event_id,
        r.approved_at AS assigned_at, replacement.planted_at AS completed_at,
        pe.planting_point_id, pe.site_zone_id AS site_id, pe.species, pe.assignment_id, pe.planter_id,
        pp.deleted_at, replacement.id AS replacement_event_id
        FROM replanting_requests r JOIN planting_events pe ON pe.id = r.planting_event_id
        JOIN planting_points pp ON pp.id = pe.planting_point_id
        LEFT JOIN planting_events replacement ON replacement.id = r.replacement_event_id
            AND COALESCE(replacement.closure_reason, '') <> 'completion_reversed'
        WHERE r.assignment_id IS NULL AND pp.deleted_at IS NULL
          AND (r.replacement_event_id IS NULL OR replacement.planted_by_user_id IS NOT NULL)''').fetchall()]


def record_lgu_replanting(event_ids, versions, *, project_site_id, planted_date,
                          species, staff_user_id, initial_height_cm=None,
                          initial_condition=None, initial_notes=None):
    import planting_database as db
    from .monitoring_locations import _safe_point, clean_event_ids

    ids = clean_event_ids(event_ids)
    if not ids or len(ids) > 500 or any(str(value) not in versions for value in ids):
        raise ValueError('Select between 1 and 500 ready locations and reload their latest records.')
    planted_at = _planting_timestamp(planted_date)
    canonical_species = db._canonical_assignment_species(species)
    if not canonical_species:
        raise ValueError('Choose a valid mangrove species.')
    height, condition, notes = _details(initial_height_cm, initial_condition, initial_notes)
    marks = ','.join('?' for _ in ids)
    conn = db._get_connection()
    try:
        site = conn.execute('''SELECT id, name FROM project_sites
            WHERE id = ? AND deleted_at IS NULL AND area_released_at IS NULL FOR SHARE''',
            (project_site_id,)).fetchone()
        if not site:
            raise ValueError('Choose an active project site for the replacement planting.')
        conn.execute(f'''SELECT id FROM planting_points WHERE id IN
            (SELECT planting_point_id FROM planting_events WHERE id IN ({marks}))
            ORDER BY id FOR UPDATE''', ids).fetchall()
        events = [dict(row) for row in conn.execute(f'''SELECT pe.*,
            death.death_at AS recorded_death_at, pp.point_num, pp.latitude, pp.longitude,
            pp.status AS point_status, pp.death_at AS point_death_at, pp.deleted_at,
            pe.site_zone_id AS source_site_id
            FROM planting_events pe JOIN planting_points pp ON pp.id = pe.planting_point_id
            JOIN point_death_records death ON death.planting_event_id = pe.id
            WHERE pe.id IN ({marks}) ORDER BY pe.id FOR UPDATE OF pe''', ids).fetchall()]
        requests = {row['planting_event_id']: dict(row) for row in conn.execute(f'''
            SELECT * FROM replanting_requests WHERE planting_event_id IN ({marks})
            ORDER BY planting_event_id FOR UPDATE''', ids).fetchall()}
        if len(events) != len(ids):
            raise ValueError('A selected dead seedling record was not found. Reload the queue.')
        resolved = db._resolve_point_project_sites(conn, [
            {**event, 'id': event['planting_point_id']} for event in events])
        for event, location in zip(events, resolved):
            request = requests.get(event['id'])
            if not request or request['assignment_id'] is not None or request['replacement_event_id'] is not None:
                raise ValueError('Only locations ready for LGU replanting can be recorded here.')
            if request['version'] != versions[str(event['id'])]:
                raise ValueError('Replanting work changed. Reload before recording planting.')
            if event['point_status'] != 'planned' or event['point_death_at'] or event['deleted_at']:
                raise ValueError('A selected location is no longer ready for LGU replanting.')
            if location['source_site_id'] != int(project_site_id):
                raise ValueError(f"Point #{event['point_num']} is not inside the selected project site.")
            _safe_point(conn, event['planting_point_id'])
            if conn.execute('''SELECT 1 FROM planting_events WHERE planting_point_id = ?
                AND (planted_at, id) > (?, ?) AND COALESCE(closure_reason, '') <> 'completion_reversed' ''',
                (event['planting_point_id'], event['planted_at'], event['id'])).fetchone():
                raise ValueError('A newer planting cycle already exists. Reload the queue.')
            if conn.execute('''SELECT 1 FROM planter_assignment_points
                WHERE planting_point_id = ? AND released_at IS NULL''', (event['planting_point_id'],)).fetchone():
                raise ValueError('A selected location has an organization allocation. Reload the queue.')
            earliest = max(db._parse_dashboard_datetime(request['approved_at']),
                           db._parse_dashboard_datetime(event['recorded_death_at']))
            if planted_at.date() < earliest.date():
                raise ValueError('Planting date cannot be before the death or replanting approval date.')

        replacements = []
        for event in events:
            request = requests[event['id']]
            recorded_at = max(planted_at, db._parse_dashboard_datetime(request['approved_at']),
                              db._parse_dashboard_datetime(event['recorded_death_at']))
            conn.execute('''UPDATE planting_events SET closed_at = COALESCE(closed_at, ?),
                closure_reason = COALESCE(closure_reason, 'recorded_death') WHERE id = ?''',
                (event['recorded_death_at'], event['id']))
            cursor = conn.execute('''INSERT INTO planting_events (
                source_key, planting_point_id, project_site_id, species, planted_at,
                point_num, latitude, longitude, source, site_attribution_source,
                inspection_interval_days, planted_by_user_id,
                initial_height_cm, initial_condition, initial_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'lgu_direct', 'lgu_direct', 14, ?, ?, ?, ?)''',
                (f'lgu-replanting:{uuid4().hex}', event['planting_point_id'], project_site_id,
                 canonical_species, recorded_at.isoformat(), event['point_num'], event['latitude'],
                 event['longitude'], staff_user_id, height, condition, notes))
            replacement_id = cursor.lastrowid
            # The legacy INSERT trigger links organization assignments. LGU
            # tasks have no planter assignment; link both records explicitly
            # under the same row locks and transaction instead.
            conn.execute('UPDATE planting_events SET replaces_event_id = ? WHERE id = ?',
                         (event['id'], replacement_id))
            conn.execute('''UPDATE replanting_requests SET replacement_event_id = ?,
                version = version + 1 WHERE planting_event_id = ?''', (replacement_id, event['id']))
            conn.execute('''UPDATE planting_points SET status = 'planted', planted_at = ?, planted_date = ?,
                death_at = NULL, death_reason = NULL, death_reason_category = NULL, death_notes = NULL
                WHERE id = ?''', (recorded_at.isoformat(), planted_at.date().isoformat(), event['planting_point_id']))
            append_activity(conn, action='seedling.replanted', actor_type='staff', actor_user_id=staff_user_id,
                project_site_id=int(project_site_id), planting_point_id=event['planting_point_id'],
                planting_event_id=replacement_id,
                summary=f"LGU recorded replacement planting at point #{event['point_num']} in {site['name']}.",
                details={'replaces_event_id': event['id'], 'species': canonical_species,
                         'planted_date': planted_at.date().isoformat()})
            replacements.append(replacement_id)
        conn.commit()
        return {'replanted_points': len(replacements), 'replacement_event_ids': replacements,
                'project_site_id': int(project_site_id), 'planted_date': planted_at.date().isoformat()}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
