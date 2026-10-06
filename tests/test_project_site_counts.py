"""Count linked site points once across analyses and unreleased assignments."""
import sqlite3

import planting_database as db


def test_grouped_site_counts_preserve_membership_without_join_multiplication():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript('''
            CREATE TABLE site_zones (id INTEGER, geometry TEXT);
            CREATE VIEW project_sites AS SELECT * FROM site_zones;
            CREATE TABLE analyses (id INTEGER, site_zone_id INTEGER);
            CREATE TABLE planter_assignments (id INTEGER, site_zone_id INTEGER);
            CREATE TABLE planting_points (id INTEGER, analysis_id INTEGER, location INTEGER);
            CREATE TABLE planter_assignment_points
                (assignment_id INTEGER, planting_point_id INTEGER, released_at TEXT);
            CREATE TABLE planting_schedules (project_site_id INTEGER);
            INSERT INTO site_zones VALUES (1,'1'),(2,'2,3'),(3,'');
            INSERT INTO analyses VALUES (1,1),(2,1),(3,NULL);
            INSERT INTO planter_assignments VALUES (1,1),(2,1),(3,2);
            INSERT INTO planting_points VALUES (1,1,1),(2,2,2),(3,3,3),(4,3,4);
            INSERT INTO planter_assignment_points VALUES
                (1,1,NULL),(2,1,NULL),(1,3,NULL),(2,4,'2026-09-01'),
                (3,3,NULL),(3,2,'2026-09-01');
            INSERT INTO planting_schedules VALUES (1),(1),(2);
        ''')
        # SQLite supplies the spatial predicate for this small membership fixture.
        conn.create_function('covers', 2, lambda geometry, location: str(location) in geometry.split(','))
        class SpatialConnection:
            def execute(self, sql, params=()):
                return conn.execute(sql.replace('extensions.ST_Covers', 'covers'), params)
        result = db._project_site_counts(SpatialConnection())
        assert result == {
            1: {'assignment_count': 2, 'analysis_count': 2, 'point_count': 2, 'schedule_count': 2},
            2: {'assignment_count': 1, 'analysis_count': 0, 'point_count': 2, 'schedule_count': 1},
            3: {'assignment_count': 0, 'analysis_count': 0, 'point_count': 0, 'schedule_count': 0},
        }
    finally:
        conn.close()
