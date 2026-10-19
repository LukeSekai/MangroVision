"""Release physical locations while preserving each site's planting cohorts."""
from __future__ import annotations

import hashlib
import json

from pyproj import Geod, Transformer
from shapely.geometry import MultiPolygon, Point, box, mapping, shape
from shapely.ops import transform, unary_union

from .activity import append_activity


def _db():
    import planting_database
    return planting_database


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _rows(conn, site_id):
    db = _db()
    return [dict(row) for row in conn.execute(f"""
        SELECT pp.id, pp.point_num, pp.latitude, pp.longitude, pp.status,
               pp.planted_at, pp.planted_date, pp.death_at, a.species,
               event.id AS planting_event_id, event.project_site_id AS event_site_id,
               event.closed_at, event.closure_reason,
               EXISTS(SELECT 1 FROM point_death_records d WHERE d.planting_event_id = event.id) AS death_recorded,
               allocation.assignment_id, allocation.assignment_status, allocation.assignment_site_id
        FROM planting_points pp JOIN analyses a ON a.id = pp.analysis_id
        LEFT JOIN LATERAL (
            SELECT {db._point_project_site_owner_sql('a.site_zone_id')} AS site_id
            FROM project_sites ps WHERE extensions.ST_Covers(ps.geometry, pp.location)
        ) owner ON true
        LEFT JOIN LATERAL (
            SELECT pe.id, pe.project_site_id, pe.closed_at, pe.closure_reason
            FROM planting_events pe WHERE pe.planting_point_id = pp.id
              AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
            ORDER BY pe.planted_at DESC, pe.id DESC LIMIT 1
        ) event ON true
        LEFT JOIN LATERAL (
            SELECT pa.id AS assignment_id, pap.status AS assignment_status,
                   pa.project_site_id AS assignment_site_id
            FROM planter_assignment_points pap JOIN planter_assignments pa ON pa.id = pap.assignment_id
            WHERE pap.planting_point_id = pp.id AND pap.released_at IS NULL
              AND pa.status IN ('active', 'completed', 'archived')
            ORDER BY pap.id DESC LIMIT 1
        ) allocation ON true
        WHERE pp.deleted_at IS NULL AND (owner.site_id = ? OR allocation.assignment_site_id = ?)
        ORDER BY pp.id
    """, (site_id, site_id)).fetchall()]


def retained_boundary(original, retained, released):
    """Keep living planting cells; holes exclude released locations between them."""
    if not retained:
        return None
    forward = Transformer.from_crs('EPSG:4326', 'EPSG:32651', always_xy=True).transform
    backward = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True).transform
    boundary = transform(forward, shape(original))
    live_locations = [Point(*forward(row['longitude'], row['latitude'])) for row in retained]
    if any(not boundary.covers(point) for point in live_locations):
        raise ValueError('Some living seedlings are outside this site boundary. Review the area before releasing it.')
    cells = []
    for row, point in zip(retained, live_locations):
        radius = .6 if str(row.get('species', '')).lower() in {'bungalon', 'api-api'} else 1.1
        cells.append(box(point.x - radius, point.y - radius, point.x + radius, point.y + radius))
    kept = boundary.intersection(unary_union(cells))
    if released:
        clear_locations = [Point(*forward(row['longitude'], row['latitude'])).buffer(.12) for row in released]
        kept = kept.difference(unary_union(clear_locations))
    if kept.is_empty or not kept.is_valid or any(not kept.covers(point) for point in live_locations):
        raise ValueError('Some locations overlap living seedlings. Review duplicated points before releasing the area.')
    polygons = [kept] if kept.geom_type == 'Polygon' else list(kept.geoms)
    return mapping(transform(backward, MultiPolygon([part for part in polygons if part.geom_type == 'Polygon'])))


def _plan(conn, site):
    site_id = int(site['id'])
    rows = _rows(conn, site_id)
    living, dead, unplanted = [], [], []
    for row in rows:
        if row['death_at']:
            dead.append(row)
        elif row['status'] == 'planted' or row['assignment_status'] == 'completed':
            living.append(row)
        else:
            unplanted.append(row)
    history = dict(conn.execute("""
        SELECT COUNT(*) AS planted, COUNT(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM point_death_records d WHERE d.planting_event_id = pe.id
        )) AS dead
        FROM planting_events pe WHERE pe.project_site_id = ?
          AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
    """, (site_id,)).fetchone())
    released = dead + unplanted
    reason = ''
    if any(not row['planting_event_id'] or not row['death_recorded'] for row in dead):
        reason = 'Some dead seedlings have incomplete planting history. Review their records before releasing them.'
    if any(row['assignment_site_id'] is not None and int(row['assignment_site_id']) != site_id for row in rows):
        reason = 'Some points are assigned to another site. Review those assignments before releasing the area.'
    after = None
    if not reason:
        try:
            after = retained_boundary(_json(site['polygon_geojson']), living, released)
        except ValueError as error:
            reason = str(error)
    can_delete = (not living and not unplanted and (bool(dead) or history['dead'] > 0)
                  and history['planted'] == history['dead'] and not reason)
    before = _json(site['polygon_geojson'])
    released_geometry = mapping(shape(before).difference(shape(after))) if after else before
    fingerprint = json.dumps({'site': dict(site), 'points': rows, 'history': history}, sort_keys=True, default=str)
    return {
        'site_id': site_id, 'name': site['name'],
        'version': hashlib.sha256(fingerprint.encode()).hexdigest(),
        'counts': {'living': len(living), 'dead': len(dead), 'unplanted': len(unplanted),
                   'release': len(released), 'historical_planted': history['planted'], 'historical_dead': history['dead']},
        'can_release': bool(released) and not reason,
        'can_delete': can_delete,
        'release_reason': reason or ('No unplanted or dead locations remain to release.' if not released else ''),
        'delete_reason': '' if can_delete else reason or 'Delete is available only when all seedlings are recorded dead and no unplanted points remain.',
        'before_geometry': before, 'after_geometry': after, 'released_geometry': released_geometry,
        'points': [{key: row[key] for key in ('id', 'point_num', 'latitude', 'longitude')} | {
            'status': 'dead' if row['death_at'] else 'planted' if row in living else 'planned',
        } for row in rows],
        '_released': released,
    }


def _site(conn, site_id, lock=False):
    row = conn.execute('SELECT * FROM project_sites WHERE id = ? AND deleted_at IS NULL'
                       + (' FOR UPDATE' if lock else ''), (int(site_id),)).fetchone()
    if not row:
        raise KeyError(site_id)
    return row


def preview_site_actions(site_id):
    conn = _db()._get_connection()
    try:
        result = _plan(conn, _site(conn, site_id))
        result.pop('_released')
        return result
    finally:
        conn.close()


def _save_boundary_release(conn, site, after, released, user_id, action='release'):
    """Persist current land ownership and its audit, never historical events."""
    before = _json(site['polygon_geojson'])
    released_geometry = mapping(shape(before).difference(shape(after))) if after else before
    site_id = int(site['id'])
    conn.execute("""UPDATE project_sites SET original_polygon_geojson = COALESCE(original_polygon_geojson, polygon_geojson),
        polygon_geojson = COALESCE(CAST(? AS jsonb), polygon_geojson), area_released_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END,
        deleted_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE deleted_at END,
        tide_calibration = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = ?""",
        (json.dumps(after) if after else None, after is None, action == 'delete', site_id))
    if after is None or action == 'delete':
        conn.execute('UPDATE planting_schedules SET project_site_id = NULL WHERE project_site_id = ?', (site_id,))
    conn.execute("""INSERT INTO project_site_area_releases
        (project_site_id, action, before_polygon, after_polygon, released_polygon, released_points, actor_user_id)
        VALUES (?, ?, CAST(? AS jsonb), CAST(? AS jsonb), CAST(? AS jsonb), CAST(? AS jsonb), ?)""",
        (site_id, action, json.dumps(before), json.dumps(after) if after else None,
         json.dumps(released_geometry), json.dumps(released, default=str), user_id))


def _selected_release_boundary(original, retained, released):
    """Release selected planting cells without changing the remaining land."""
    if not retained:
        return None
    forward = Transformer.from_crs('EPSG:4326', 'EPSG:32651', always_xy=True).transform
    backward = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True).transform
    boundary = transform(forward, shape(original))
    cells = []
    for row in released:
        x, y = forward(row['longitude'], row['latitude'])
        radius = .6 if str(row.get('species', '')).lower() in {'bungalon', 'api-api'} else 1.1
        cells.append(box(x - radius, y - radius, x + radius, y + radius))
    kept = boundary.difference(unary_union(cells))
    if (kept.is_empty or not kept.is_valid
            or any(not kept.covers(Point(*forward(row['longitude'], row['latitude']))) for row in retained)
            or any(kept.covers(Point(*forward(row['longitude'], row['latitude']))) for row in released)):
        raise ValueError('Selected locations overlap other seedlings. Review the site before releasing them.')
    polygons = [kept] if kept.geom_type == 'Polygon' else list(kept.geoms)
    return mapping(transform(backward, MultiPolygon([part for part in polygons if part.geom_type == 'Polygon'])))


def release_dead_locations(event_ids, versions, user_id, *, prepared=False):
    """Release selected dead planting cells and allocations in one transaction.

    The private prepared option converts old, unplanted LGU queue reservations.
    The public action always requires the location to still be recorded dead.
    """
    from .monitoring_locations import clean_event_ids

    ids = clean_event_ids(event_ids)
    if not ids or len(ids) > 500 or any(str(value) not in versions for value in ids):
        raise ValueError('Select between 1 and 500 dead locations and reload their latest records.')
    if user_id is None:
        raise ValueError('An LGU staff member must release dead locations.')
    db = _db()
    conn = db._get_connection()
    marks = ','.join('?' for _ in ids)
    try:
        conn.execute("SET LOCAL lock_timeout = '5s'")
        initial = [dict(row) for row in conn.execute(f'''SELECT pe.id AS planting_event_id,
            pe.planting_point_id AS id, pp.latitude, pp.longitude, pe.project_site_id AS source_site_id
            FROM planting_events pe JOIN planting_points pp ON pp.id = pe.planting_point_id
            WHERE pe.id IN ({marks})''', ids).fetchall()]
        if len(initial) != len(ids):
            raise ValueError('A selected planting record was not found. Reload the map.')
        resolved = db._resolve_point_project_sites(conn, initial)
        site_ids = sorted({row['source_site_id'] for row in resolved if row['source_site_id'] is not None})
        if len([row for row in resolved if row['source_site_id'] is not None]) != len(ids):
            raise ValueError('A selected location no longer belongs to an active project site. Reload the map.')
        sites = {site_id: dict(_site(conn, site_id, lock=True)) for site_id in site_ids}
        site_rows = {site_id: _rows(conn, site_id) for site_id in site_ids}
        point_ids = sorted({row['id'] for rows in site_rows.values() for row in rows})
        point_marks = ','.join('?' for _ in point_ids)
        if not point_ids:
            raise ValueError('The selected site changed. Reload the map.')
        # Match area release ordering: sites, physical points, allocations, events.
        conn.execute(f'SELECT id FROM planting_points WHERE id IN ({point_marks}) ORDER BY id FOR UPDATE', point_ids).fetchall()
        conn.execute(f'SELECT id FROM planter_assignment_points WHERE planting_point_id IN ({point_marks}) ORDER BY id FOR UPDATE', point_ids).fetchall()
        conn.execute(f'SELECT id FROM planting_events WHERE planting_point_id IN ({point_marks}) ORDER BY id FOR UPDATE', point_ids).fetchall()
        current = db._resolve_point_project_sites(conn, initial)
        if {row['id']: row['source_site_id'] for row in current} != {row['id']: row['source_site_id'] for row in resolved}:
            raise ValueError('Site ownership changed. Reload the map before releasing locations.')
        owner_by_point = {row['id']: row['source_site_id'] for row in current}
        rows_by_point = {row['id']: (site_id, row) for site_id in site_ids for row in _rows(conn, site_id)
                         if owner_by_point.get(row['id']) == site_id}
        groups = {}
        for event_id in ids:
            event = conn.execute('SELECT * FROM planting_events WHERE id = ?', (event_id,)).fetchone()
            request = conn.execute('SELECT * FROM replanting_requests WHERE planting_event_id = ? FOR UPDATE', (event_id,)).fetchone()
            if versions[str(event_id)] != (request['version'] if request else 0):
                raise ValueError('Location changed. Reload the map before releasing it.')
            entry = rows_by_point.get(event['planting_point_id'])
            if not entry or entry[1]['planting_event_id'] != event_id:
                raise ValueError('A newer planting cycle or site change exists. Reload the map.')
            site_id, point = entry
            is_prepared = (prepared and request and request['assignment_id'] is None
                           and request['replacement_event_id'] is None and point['status'] == 'planned'
                           and point['assignment_id'] is None)
            if not point['death_recorded'] or (not point['death_at'] and not is_prepared):
                raise ValueError('Only a confirmed dead seedling can be released from its project site.')
            if request and (request['assignment_id'] is not None or request['replacement_event_id'] is not None):
                raise ValueError('This location has already been assigned or planted again. Reload the map.')
            if point['assignment_site_id'] is not None and point['assignment_site_id'] != site_id:
                raise ValueError('This location is assigned to another site. Reload the map.')
            if db._first_deleted_point(conn, [point['id']]):
                raise ValueError('Removed locations cannot be released.')
            if db._first_eroded_unavailable_point(conn, [point['id']]):
                raise ValueError('This location is currently unavailable because of erosion.')
            groups.setdefault(site_id, []).append(point)

        # Validate every boundary before applying any release.
        plans = []
        for site_id, released in groups.items():
            released_ids = {row['id'] for row in released}
            retained = [row for row in _rows(conn, site_id) if row['id'] not in released_ids]
            after = _selected_release_boundary(_json(sites[site_id]['polygon_geojson']), retained, released)
            plans.append((sites[site_id], after, released))
        for site, after, released in plans:
            for point in released:
                conn.execute('''INSERT INTO replanting_requests (planting_event_id, approved_by)
                    VALUES (?, ?) ON CONFLICT (planting_event_id) DO NOTHING''', (point['planting_event_id'], user_id))
                conn.execute('''UPDATE planter_assignment_points SET released_at = CURRENT_TIMESTAMP, released_by = ?
                    WHERE planting_point_id = ? AND released_at IS NULL''', (user_id, point['id']))
                conn.execute("""UPDATE planting_points SET status = 'planned', planted_at = NULL, planted_date = NULL,
                    death_at = NULL, death_reason = NULL, death_reason_category = NULL, death_notes = NULL WHERE id = ?""", (point['id'],))
            _save_boundary_release(conn, site, after, released, user_id)
            conn.execute("""UPDATE planter_assignments pa SET status = 'archived'
                WHERE pa.project_site_id = ? AND pa.status = 'active' AND NOT EXISTS (
                    SELECT 1 FROM planter_assignment_points pap WHERE pap.assignment_id = pa.id AND pap.released_at IS NULL)""", (site['id'],))
            append_activity(conn, action='project_site.locations_released', actor_type='staff', actor_user_id=user_id,
                organization_id=site['organization_id'], project_site_id=site['id'],
                summary=f"Released {len(released)} dead locations from {site['name']}.",
                details={'planting_event_ids': [row['planting_event_id'] for row in released], 'history_preserved': True})
        conn.commit()
        return {'released_count': len(ids), 'planting_event_ids': ids, 'status': 'planned'}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def apply_site_action(site_id, action, expected_version, user_id):
    if action not in {'release', 'delete'}:
        raise ValueError('Unknown site action.')
    db = _db()
    conn = db._get_connection()
    try:
        conn.execute("SET LOCAL lock_timeout = '5s'")
        site = _site(conn, site_id, lock=True)
        # Lock the physical locations before their event/assignment records,
        # matching the existing replacement approval workflow.
        ids = [row['id'] for row in _rows(conn, int(site_id))]
        if ids:
            marks = ','.join('?' for _ in ids)
            conn.execute(f'SELECT id FROM planting_points WHERE id IN ({marks}) ORDER BY id FOR UPDATE', ids).fetchall()
            conn.execute(f'SELECT id FROM planter_assignment_points WHERE planting_point_id IN ({marks}) ORDER BY id FOR UPDATE', ids).fetchall()
            conn.execute(f'SELECT id FROM planting_events WHERE planting_point_id IN ({marks}) ORDER BY id FOR UPDATE', ids).fetchall()
        plan = _plan(conn, site)
        if expected_version != plan['version']:
            raise ValueError('This site changed after the preview. Review the updated counts and boundary before confirming.')
        if not plan[f'can_{action}']:
            raise ValueError(plan[f'{action}_reason'])
        released = plan['_released']
        released_ids = [row['id'] for row in released]
        marks = ','.join('?' for _ in released_ids)
        if released_ids:
            # Freeze original attribution before clearing physical death state.
            # Death records and planting events are never deleted or reassigned.
            for row in released:
                if row['death_at']:
                    if user_id is None:
                        raise ValueError('An LGU staff member must approve releasing dead locations.')
                    conn.execute('INSERT INTO replanting_requests (planting_event_id, approved_by) VALUES (?, ?) ON CONFLICT (planting_event_id) DO NOTHING',
                                 (row['planting_event_id'], user_id))
                    conn.execute("""UPDATE planting_events SET project_site_id = COALESCE(project_site_id, ?),
                        closed_at = COALESCE(closed_at, ?), closure_reason = COALESCE(closure_reason, 'recorded_death')
                        WHERE id = ?""", (site_id, row['death_at'], row['planting_event_id']))
            conn.execute(f"""UPDATE planter_assignment_points SET released_at = CURRENT_TIMESTAMP, released_by = ?
                WHERE planting_point_id IN ({marks}) AND released_at IS NULL""", (user_id, *released_ids))
            conn.execute(f"""UPDATE replanting_requests r SET assignment_id = NULL, version = version + 1
                FROM planting_events pe WHERE pe.id = r.planting_event_id
                AND pe.planting_point_id IN ({marks}) AND r.assignment_id IS NOT NULL
                AND r.replacement_event_id IS NULL""", released_ids)
            conn.execute(f"""UPDATE planting_points SET status = 'planned', planted_at = NULL, planted_date = NULL,
                death_at = NULL, death_reason = NULL, death_reason_category = NULL, death_notes = NULL
                WHERE id IN ({marks})""", released_ids)
        conn.execute("""UPDATE planter_assignments pa SET status = 'archived'
            WHERE pa.project_site_id = ? AND pa.status = 'active' AND NOT EXISTS (
                SELECT 1 FROM planter_assignment_points pap WHERE pap.assignment_id = pa.id AND pap.released_at IS NULL
            )""", (site_id,))
        after = plan['after_geometry']
        _save_boundary_release(conn, dict(site), after, released, user_id, action)
        append_activity(conn, action=f'project_site.{"area_released" if action == "release" else "deleted"}', actor_type='staff',
            actor_user_id=user_id, organization_id=site['organization_id'], project_site_id=int(site_id),
            summary=f'{"Released area from" if action == "release" else "Deleted active site"} {site["name"]}.',
            details={'counts': plan['counts'], 'history_preserved': True})
        conn.commit()
        return {'status': 'released' if action == 'release' else 'deleted', 'id': int(site_id), 'counts': plan['counts']}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def available_released_areas():
    """Currently unclaimed parts of recorded releases, available to any owner."""
    conn = _db()._get_connection()
    try:
        row = conn.execute("""
            WITH released AS (SELECT extensions.ST_UnaryUnion(extensions.ST_Collect(
                extensions.ST_SetSRID(extensions.ST_GeomFromGeoJSON(released_polygon::text), 4326))) AS geometry
                FROM project_site_area_releases), allocated AS (
                SELECT extensions.ST_UnaryUnion(extensions.ST_Collect(geometry)) AS geometry
                FROM project_sites WHERE deleted_at IS NULL AND area_released_at IS NULL
            ) SELECT extensions.ST_AsGeoJSON(CASE WHEN allocated.geometry IS NULL THEN released.geometry
                ELSE extensions.ST_Difference(released.geometry, allocated.geometry) END, 15) AS geometry
            FROM released CROSS JOIN allocated
        """).fetchone()
        geometry = _json(row['geometry']) if row and row['geometry'] else None
        if not geometry or shape(geometry).is_empty or abs(Geod(ellps='WGS84').geometry_area_perimeter(shape(geometry))[0]) < .01:
            return {'type': 'FeatureCollection', 'features': []}
        return {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {'name': 'Released area'}, 'geometry': geometry}]}
    finally:
        conn.close()
