"""Create owned project sites from an explicit selection of available locations."""
from __future__ import annotations

import json
import math

from pyproj import Transformer
from shapely import get_coordinates
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box, mapping, shape
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

from .activity import append_activity
from .zones import normalize_polygon


def _polygonal(geometry):
    polygons = []
    def collect(candidate):
        if candidate.geom_type == 'Polygon' and not candidate.is_empty:
            polygons.append(candidate)
        elif hasattr(candidate, 'geoms'):
            for part in candidate.geoms:
                collect(part)
    collect(geometry)
    if not polygons:
        raise ValueError('Draw over at least one mapped point inside the drone map.')
    return unary_union(polygons)


def _simplify_ring(ring, tolerance, map_edge):
    vertices = list(ring.coords)[:-1]
    fixed = [index for index,vertex in enumerate(vertices) if map_edge.distance(Point(vertex)) < 1e-12]
    if not fixed:
        return list(Polygon(ring).simplify(tolerance,preserve_topology=True).exterior.coords)
    # Keep every vertex on the real imagery edge or a coverage hole. Simplify
    # only the user-drawn sections between those anchors.
    output = []
    for index,start in enumerate(fixed):
        end = fixed[(index+1)%len(fixed)]
        end += len(vertices) if end <= start else 0
        segment = [vertices[position%len(vertices)] for position in range(start,end+1)]
        output.extend(list(LineString(segment).simplify(tolerance,preserve_topology=True).coords)[:-1])
    return output+[output[0]] if len(output)>=3 else list(ring.coords)


def _clean_drawn_shape(requested, coverage, points, allowed=None):
    """Reduce tracing noise without changing membership or the real map edge."""
    original = _polygonal(requested.intersection(coverage))
    current = original
    locations = [Point(row['longitude'],row['latitude']) for row in points]
    membership = [original.covers(point) for point in locations]
    limit = coverage if allowed is None else allowed
    changed = False
    # Use a small geographic tolerance (at most about 35 cm in this map).
    # Vertices keep their exact original coordinates, avoiding projection drift.
    # Reach a fixed point here so preview and saving return the same outline.
    for _ in range(8):
        next_shape = None
        for metres in (.35,.2,.1,.05):
            tolerance = metres/111320
            parts = [current] if current.geom_type == 'Polygon' else list(current.geoms)
            try:
                candidates = [Polygon(_simplify_ring(part.exterior,tolerance,coverage.boundary),
                    [_simplify_ring(ring,tolerance,coverage.boundary) for ring in part.interiors]) for part in parts]
                candidate = MultiPolygon(candidates)
                if not candidate.is_valid or candidate.is_empty:
                    continue
                candidate = _polygonal(candidate.intersection(coverage))
            except ValueError:
                continue
            if len(get_coordinates(candidate)) >= len(get_coordinates(current)):
                continue
            if limit.covers(candidate) and [candidate.covers(point) for point in locations] == membership:
                next_shape = candidate
                break
        if next_shape is None:
            return current, changed
        current = next_shape
        changed = True
    # Extremely complex traces keep their original shape rather than letting
    # repeated saves progressively alter a boundary that has not settled.
    return original, False


def prepare_drawn_site_boundary(geometry, coverage, allowed=None):
    """Clean a freehand site, trim it to the map, and retain its mapped points."""
    import planting_database as db

    if coverage is None or coverage.is_empty or not coverage.is_valid:
        raise RuntimeError('The mapped area is unavailable.')
    try:
        if not isinstance(geometry, dict) or geometry.get('type') not in {'Polygon', 'MultiPolygon'}:
            raise ValueError('Not a polygon')
        raw = shape(geometry)
        if raw.is_empty or any(not math.isfinite(x) or not math.isfinite(y) or abs(x)>180 or abs(y)>90
                               for x,y in get_coordinates(raw)):
            raise ValueError('Invalid coordinates')
        repaired = not raw.is_valid
        requested = _polygonal(make_valid(raw) if repaired else raw)
    except Exception as error:
        raise ValueError('Draw a closed project-site outline using valid map coordinates.') from error
    boundary = _polygonal(requested.intersection(coverage))
    prepared = normalize_polygon(mapping(boundary))
    conn = db._get_connection()
    try:
        points = [dict(row) for row in conn.execute('''SELECT id,longitude,latitude FROM planting_points
            WHERE deleted_at IS NULL AND location && extensions.ST_Expand(
                extensions.ST_SetSRID(extensions.ST_GeomFromGeoJSON(?),4326),.00001)''',
            (json.dumps(prepared),)).fetchall()]
    finally:
        conn.close()
    count = sum(boundary.covers(Point(row['longitude'],row['latitude'])) for row in points)
    if not count:
        raise ValueError('Draw over at least one mapped point inside the drone map.')
    cleaned, changed = _clean_drawn_shape(requested, coverage, points, allowed)
    return {'geometry': normalize_polygon(mapping(cleaned)), 'point_count': count,
            'clipped': not requested.equals(boundary), 'cleaned': repaired or changed}


def lock_site_boundaries(conn):
    """Serialize ownership checks with the manual project-site boundary writers."""
    conn.execute("SET LOCAL lock_timeout = '5s'")
    conn.execute("SELECT pg_advisory_xact_lock(hashtext('mangrovision.project_site_boundaries'))")


def check_other_owners(conn, geometry, organization_id, site_id=None):
    conflict = conn.execute('''SELECT 1 FROM project_sites
        WHERE deleted_at IS NULL AND area_released_at IS NULL
          AND organization_id IS DISTINCT FROM ? AND (CAST(? AS bigint) IS NULL OR id <> ?)
          AND extensions.ST_Relate(geometry,
            extensions.ST_SetSRID(extensions.ST_GeomFromGeoJSON(?), 4326), 'T********') LIMIT 1''',
        (organization_id, site_id, site_id, json.dumps(geometry))).fetchone()
    if conflict:
        raise ValueError("This area now belongs to another organization. Refresh the map and select available points.")


def selected_point_boundary(selected, other_points, coverage, excluded):
    """Use planting cells, leaving every unselected mapped location outside."""
    forward = Transformer.from_crs('EPSG:4326', 'EPSG:32651', always_xy=True).transform
    backward = Transformer.from_crs('EPSG:32651', 'EPSG:4326', always_xy=True).transform
    cells, centers = [], []
    for row in selected:
        center = Point(*forward(row['longitude'], row['latitude']))
        radius = 1.1 if str(row.get('species', '')).casefold() == 'rhizophora' else .6
        centers.append(center)
        cells.append(box(center.x-radius, center.y-radius, center.x+radius, center.y+radius))
    boundary = unary_union(cells).intersection(transform(forward, coverage))
    if excluded:
        boundary = boundary.difference(transform(forward, unary_union(excluded)))
    if not boundary.is_empty:
        west, south, east, north = boundary.bounds
        neighbors = []
        for row in other_points:
            x, y = forward(row['longitude'], row['latitude'])
            if west-.12 <= x <= east+.12 and south-.12 <= y <= north+.12:
                neighbors.append(Point(x, y).buffer(.12))
        if neighbors:
            boundary = boundary.difference(unary_union(neighbors))
    if boundary.is_empty or not boundary.is_valid or any(not boundary.covers(center) for center in centers):
        raise ValueError('Some selected points overlap a claimed area, an unavailable area, or another mapped point. Refresh the map and review the selection.')
    # Projection back to GPS can put microscopic slivers across a shared edge.
    # Recut in the exact coordinate system used by the stored ownership checks,
    # leaving about a tenth of a millimetre at existing boundaries. Never relax
    # the ownership check or permit a real overlap with another organization.
    result = transform(backward, boundary).intersection(coverage)
    if excluded:
        result = result.difference(unary_union(excluded).buffer(1e-9))
    result = _polygonal(result)
    if not result.is_valid or any(not result.covers(Point(row['longitude'],row['latitude'])) for row in selected):
        raise ValueError('Some selected points touch an unavailable boundary. Refresh the map and review the selection.')
    return normalize_polygon(mapping(result))


def create_site_from_points(*, organization_id, planting_point_ids, coverage, name='', notes='', user_id, connection=None):
    import planting_database as db

    ids = sorted(set(planting_point_ids))
    if not ids or len(ids) > 10000 or any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError('Select between 1 and 10000 available planting points.')
    if coverage is None or coverage.is_empty or not coverage.is_valid:
        raise ValueError('The mapped area is unavailable. Refresh the map and try again.')
    conn = connection or db._get_connection()
    try:
        lock_site_boundaries(conn)
        organization = conn.execute('SELECT * FROM organizations WHERE id = ? FOR SHARE', (organization_id,)).fetchone()
        if not organization:
            raise ValueError('Choose an existing organization owner.')
        marks = ','.join('?' for _ in ids)
        conn.execute(f'SELECT id FROM planting_points WHERE id IN ({marks}) ORDER BY id FOR UPDATE', ids).fetchall()
        rows = [dict(row) for row in conn.execute(f'''SELECT pp.id, pp.latitude, pp.longitude,
            pp.status, pp.death_at, pp.deleted_at, pp.planted_at, a.species,
            EXISTS(SELECT 1 FROM project_sites ps WHERE ps.deleted_at IS NULL
                AND ps.area_released_at IS NULL AND extensions.ST_Covers(ps.geometry, pp.location)) AS claimed,
            EXISTS(SELECT 1 FROM planter_assignment_points pap JOIN planter_assignments pa ON pa.id=pap.assignment_id
                WHERE pap.planting_point_id=pp.id AND pap.released_at IS NULL
                  AND pa.status IN ('active','completed','archived')) AS allocated,
            EXISTS(SELECT 1 FROM planting_events pe WHERE pe.planting_point_id=pp.id AND pe.closed_at IS NULL
                AND COALESCE(pe.closure_reason,'') <> 'completion_reversed') AS has_living_event,
            EXISTS(SELECT 1 FROM map_zones mz WHERE mz.deleted_at IS NULL AND mz.zone_type IN ('forbidden','eroded')
                AND extensions.ST_Covers(mz.geometry,pp.location)) AS restricted
            FROM planting_points pp JOIN analyses a ON a.id=pp.analysis_id
            WHERE pp.id IN ({marks}) ORDER BY pp.id''', ids).fetchall()]
        if len(rows) != len(ids) or any(row['status'] != 'planned' or any(row.get(key) for key in
            ('deleted_at','death_at','planted_at','claimed','allocated','has_living_event','restricted')) for row in rows):
            raise ValueError('Some selected points are no longer available. Use unclaimed planned points without assignments, planting, or erosion restrictions.')
        if any(not coverage.covers(Point(row['longitude'], row['latitude'])) for row in rows):
            raise ValueError('Select points inside the drone map.')
        excluded = [shape(db._monitoring_json(row['polygon_geojson'])) for row in conn.execute('''
            SELECT polygon_geojson FROM project_sites WHERE deleted_at IS NULL AND area_released_at IS NULL
            UNION ALL SELECT polygon_geojson FROM map_zones
                WHERE deleted_at IS NULL AND zone_type IN ('forbidden','eroded')''').fetchall()]
        neighbors = [dict(row) for row in conn.execute(f'''SELECT longitude,latitude FROM planting_points
            WHERE deleted_at IS NULL AND id NOT IN ({marks})''', ids).fetchall()]
        geometry = selected_point_boundary(rows, neighbors, coverage, excluded)
        check_other_owners(conn, geometry, organization_id)
        count = conn.execute('SELECT count(*) AS total FROM project_sites WHERE organization_id = ? AND deleted_at IS NULL',
                             (organization_id,)).fetchone()['total']
        site_name = str(name or '').strip() or f"{organization['name']} Site {count+1}"
        cursor = conn.execute('''INSERT INTO project_sites(name,notes,polygon_geojson,organization_id,inspection_interval_days)
            VALUES(?,?,CAST(? AS jsonb),?,?)''', (site_name[:120], str(notes or '').strip() or None,
            json.dumps(geometry), organization_id, organization['inspection_interval_days']))
        site_id = cursor.lastrowid
        # Point ownership follows the boundary. Do not move analysis links,
        # assignments, or historical planting events when claiming released land.
        append_activity(conn, action='project_site.created_from_points', actor_type='staff', actor_user_id=user_id,
            organization_id=organization_id, project_site_id=site_id,
            summary=f'Created {site_name[:120]} from {len(ids)} selected planting locations.',
            details={'selected_point_count': len(ids), 'planting_point_ids': ids})
        if connection is None:
            conn.commit()
        return {'id': site_id, 'name': site_name[:120], 'organization_id': organization_id,
                'selected_point_count': len(ids), 'geometry': geometry}
    except Exception:
        if connection is None:
            conn.rollback()
        raise
    finally:
        if connection is None:
            conn.close()


def assign_points_as_site(*, organization_id, planting_point_ids, coverage,
                          planting_schedule_id=None, user_id, **assignment_values):
    """Claim the selected cells and allocate planting work in one transaction."""
    import planting_database as db
    from .lgu_sites import is_lgu

    conn = db._get_connection()
    try:
        lock_site_boundaries(conn)
        organization = conn.execute('SELECT id FROM organizations WHERE id=? FOR UPDATE',
                                    (organization_id,)).fetchone()
        if not organization:
            raise ValueError('Choose an existing organization.')
        if planting_schedule_id is None and not is_lgu(conn, organization_id):
            raise ValueError('Choose a confirmed tree-planting activity before assigning points.')
        activity = None
        site_id = None
        if planting_schedule_id is not None:
            activity = conn.execute('SELECT * FROM planting_schedules WHERE id=? FOR UPDATE',
                                    (planting_schedule_id,)).fetchone()
            if (not activity or activity['organization_id'] != organization_id
                    or activity['appointment_type'] != 'tree_planting'
                    or activity['status'] not in {'confirmed', 'in_progress'}):
                raise ValueError('Choose a confirmed tree-planting activity belonging to this organization.')
            # Previously assigned activities retain their site and history. This
            # is automatic compatibility, never a new/existing-site choice.
            site_id = activity['project_site_id']
        if site_id is None:
            site = create_site_from_points(organization_id=organization_id,
                planting_point_ids=planting_point_ids, coverage=coverage,
                user_id=user_id, connection=conn)
            site_id = site['id']
        assignment_ids = db.create_organization_assignments(organization_id,
            planting_point_ids, site_zone_id=site_id,
            planting_schedule_id=planting_schedule_id,
            assigned_by_user_id=user_id, _connection=conn, **assignment_values)
        conn.commit()
        return {'assignment_id': assignment_ids[0], 'assignment_ids': assignment_ids,
                'project_site_id': site_id, 'selected_point_count': len(set(planting_point_ids))}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
