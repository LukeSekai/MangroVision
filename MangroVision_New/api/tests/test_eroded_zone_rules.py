"""Synthetic zone/seedling regressions; never connects to shared PostgreSQL."""
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from shapely.geometry import Point, box, mapping, shape

import planting_database as db
from mangrovision_db import zones, seedlings
from mangrovision_db.erosion import PLANTING_BLOCKED, SeedlingOverlap, point_is_eroded


@pytest.fixture
def spatial_db(monkeypatch):
    raw = sqlite3.connect(':memory:')
    raw.row_factory = sqlite3.Row
    raw.create_function('covers', 2, lambda polygon, location: int(shape(json.loads(polygon)).covers(shape(json.loads(location)))))
    raw.create_function('point', 2, lambda x, y: json.dumps(mapping(Point(x, y))))
    raw.create_function('srid', 2, lambda geometry, _: geometry)
    raw.create_function('geojson', 1, lambda geometry: geometry)
    raw.executescript('''
        CREATE TABLE map_zones(id INTEGER PRIMARY KEY, zone_type TEXT, name TEXT,
            warning_type TEXT, severity TEXT, notes TEXT, properties TEXT, polygon_geojson TEXT,
            geometry TEXT GENERATED ALWAYS AS (polygon_geojson), created_by_user_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP, deleted_at TEXT);
        CREATE TABLE planting_points(id INTEGER PRIMARY KEY, point_num INTEGER, latitude REAL, longitude REAL,
            location TEXT, status TEXT, death_at TEXT, death_reason TEXT, deleted_at TEXT);
        CREATE TABLE planting_events(id INTEGER PRIMARY KEY, planting_point_id INTEGER,
            initial_condition TEXT, initial_height_cm REAL, initial_notes TEXT);
        CREATE TABLE planter_assignment_points(id INTEGER PRIMARY KEY, planting_point_id INTEGER, status TEXT, released_at TEXT);
    ''')
    for ident, x, status, death in [(1, 1, 'planned', None), (2, 2, 'planted', None),
                                   (3, 3, 'planted', '2026-10-01'), (4, 4, 'planned', None)]:
        raw.execute('INSERT INTO planting_points VALUES(?,?,?,?,?,?,?,?,?)',
                    (ident, ident, 1, x, json.dumps(mapping(Point(x, 1))), status, death, 'waves' if death else None, None))
    raw.executescript("INSERT INTO planting_events VALUES(2,2,'healthy',10,'living'), (3,3,'stressed',8,'dead');")
    raw.commit()

    class Connection:
        def execute(self, sql, params=()):
            if 'advisory_xact_lock' in sql or sql.startswith('SET LOCAL'):
                return MagicMock()
            for before, after in [('extensions.ST_Covers', 'covers'), ('extensions.ST_SetSRID', 'srid'),
                                  ('extensions.ST_MakePoint', 'point'), ('extensions.ST_GeomFromGeoJSON', 'geojson')]:
                sql = sql.replace(before, after)
            return raw.execute(sql, params)

        def commit(self): raw.commit()
        def rollback(self): raw.rollback()
        def close(self): pass

    conn = Connection()
    monkeypatch.setattr(zones, 'get_connection', lambda: conn)
    yield conn
    raw.close()


def feature(left=0, right=3.5):
    return {'type': 'Feature', 'properties': {'name': 'Erosion'}, 'geometry': mapping(box(left, 0, right, 2))}


def seedling_snapshot(conn):
    return [[tuple(row) for row in conn.execute(f'SELECT * FROM {table} ORDER BY id').fetchall()]
            for table in ('planting_points', 'planting_events')]


def test_overlap_requires_confirmation_and_never_changes_recorded_seedlings(spatial_db):
    conn = spatial_db
    before = seedling_snapshot(conn)
    with pytest.raises(SeedlingOverlap):
        zones.create_zone('eroded', feature())
    assert conn.execute('SELECT COUNT(*) FROM map_zones').fetchone()[0] == 0
    assert seedling_snapshot(conn) == before
    saved = zones.create_zone('eroded', feature(), confirm_overlap=True)
    assert point_is_eroded(conn, 1, 2)
    assert seedling_snapshot(conn) == before
    # A cancelled edit (unconfirmed overlap) leaves the saved boundary intact.
    with pytest.raises(SeedlingOverlap):
        zones.update_zone('eroded', saved['id'], feature(1.5, 4.5))
    assert zones.get_zone('eroded', saved['id']) == saved
    zones.update_zone('eroded', saved['id'], feature(1.5, 4.5), confirm_overlap=True)
    assert not point_is_eroded(conn, 1, 1)
    assert point_is_eroded(conn, 1, 4)
    zones.soft_delete_zone('eroded', saved['id'])
    assert not point_is_eroded(conn, 1, 4)
    assert seedling_snapshot(conn) == before


def test_overlapping_zones_boundaries_and_capacity_follow_current_geometry(spatial_db):
    conn = spatial_db
    first = zones.create_zone('eroded', feature(0, 1))  # Boundary is covered too.
    second = zones.create_zone('eroded', feature(1, 1.5))
    assert db._current_planting_capacity(conn) == 1
    zones.soft_delete_zone('eroded', first['id'])
    assert point_is_eroded(conn, 1, 1)
    assert db._current_planting_capacity(conn) == 1
    zones.update_zone('eroded', second['id'], feature(1.1, 1.5))
    assert db._current_planting_capacity(conn) == 2
    assert conn.execute("SELECT COUNT(*) FROM planting_points WHERE status='planted'").fetchone()[0] == 2
    assert conn.execute('SELECT COUNT(*) FROM planting_points WHERE death_at IS NOT NULL').fetchone()[0] == 1


def test_replacing_collection_is_atomic_when_later_zone_needs_confirmation(spatial_db):
    conn = spatial_db
    existing = zones.create_zone('eroded', feature(0, .5))
    with pytest.raises(SeedlingOverlap):
        zones.replace_zone_collection('eroded', [feature(4, 4.5), feature(2, 3)])
    assert zones.feature_collection('eroded')['features'] == [existing]


def test_warning_overlap_keeps_nonblocking_behavior(spatial_db):
    conn = spatial_db
    before = seedling_snapshot(conn)
    saved = zones.create_zone('warning', feature())
    zones.update_zone('warning', saved['id'], feature(1, 4))
    assert not point_is_eroded(conn, 1, 2)
    assert db._current_planting_capacity(conn) == 2
    assert seedling_snapshot(conn) == before


def test_mixed_bulk_planting_rejects_every_point_before_updates():
    conn = MagicMock()
    rows = [dict(assignment_point_id=n, planting_point_id=n, latitude=1, longitude=n,
                 deleted_at=None) for n in (1, 2)]
    conn.execute.return_value.fetchall.return_value = rows
    with patch.object(db, '_get_connection', return_value=conn), \
         patch.object(db, '_is_point_inside_eroded_zone', side_effect=[False, True]) as check:
        with pytest.raises(ValueError, match=PLANTING_BLOCKED):
            db.mark_planter_points_completed(1, [1, 2])
    assert check.call_args.kwargs['conn'] is conn
    assert not any(call.args[0].lstrip().startswith(('UPDATE', 'INSERT')) for call in conn.execute.call_args_list)
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_condition_update_still_allowed_for_seedling_inside_erosion(spatial_db):
    zones.create_zone('eroded', feature(), confirm_overlap=True)
    assert point_is_eroded(spatial_db, 1, 2)
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = dict(id=2, planting_point_id=2,
        point_num=2, project_site_id=1, organization_id=1, status='planted')
    with patch.object(seedlings, 'get_connection', return_value=conn), patch.object(seedlings, 'append_activity'):
        result = seedlings.update_seedling_details(2, staff_user_id=1,
            initial_height_cm=12, initial_condition='fair', initial_notes='Observed in eroded zone')
    assert result['initial_condition'] == 'fair'
    conn.commit.assert_called_once()
    assert not any('map_zones' in call.args[0] for call in conn.execute.call_args_list)


@pytest.mark.parametrize('record_death', [False, True])
def test_lgu_monitoring_keeps_working_inside_erosion(spatial_db, record_death):
    from mangrovision_db import lgu_sites
    zones.create_zone('eroded', feature(), confirm_overlap=True)
    assert point_is_eroded(spatial_db, 1, 2)
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [{'planting_point_id': 2}]
    context = dict(version='current', monitoring_available=True, points=[dict(id=2,
        planting_point_id=2, planted_at='2026-01-01T12:00:00+08:00', inspection_interval_days=14)],
        next_monitoring_date='2026-01-15', total_planted=1, alive_count=1, dead_count=0)
    with patch.object(db, '_get_connection', return_value=conn), \
         patch.object(lgu_sites, '_site', return_value=dict(id=1, name='Demo site', organization_id=1)), \
         patch.object(lgu_sites, '_context', return_value=context), patch.object(lgu_sites, 'append_activity'):
        result = lgu_sites.record_visit(1, monitored_at='2026-02-01T12:00:00+08:00',
            version='current', dead_planting_event_ids=[2] if record_death else [],
            health_status='good', actions_taken='Observed seedlings inside the existing erosion boundary',
            inspector_user_id=1, death_reason_category='waves' if record_death else None)
    assert result['new_dead_count'] == int(record_death)
    conn.commit.assert_called_once()
    assert not any('map_zones' in call.args[0] for call in conn.execute.call_args_list)
