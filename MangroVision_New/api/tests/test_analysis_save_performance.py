"""Fast spacing cleanup must make the same choices as an exhaustive scan."""
import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import planting_database as database


class ExhaustiveNeighbors:
    def __init__(self, radius):
        self.rows = []

    def add(self, row):
        self.rows.append(row)

    def neighbors(self, row):
        return iter(self.rows)


class CleanupConnection:
    def __init__(self, rows):
        self.rows, self.deleted = rows, set()

    def execute(self, sql, params=()):
        if sql.strip().startswith('SELECT'):
            return SimpleNamespace(fetchall=lambda: self.rows)
        if 'UPDATE planting_points' in sql:
            self.deleted.update(params[4:])
        return SimpleNamespace()


def point(number, x, y, lat=10.8, lon=122.6, species='rhizophora', status='planned', assigned=False, spacing=2):
    return dict(id=number, analysis_id=number % 4 + 1,
                latitude=lat + y / 111320,
                longitude=lon + x / (111320 * max(.2, math.cos(math.radians(lat)))),
                species=species, planting_distance_m=spacing, status=status,
                has_active_assignment=assigned, analyzed_at=f'2026-10-{number%10+1:02}')


class AnalysisSavePerformanceTests(unittest.TestCase):
    def test_cleanup_matches_exhaustive_scan_including_protected_records_and_cell_boundaries(self):
        rng = random.Random(42)
        rows = []
        for lat in (10.8, -30, 85):
            for i in range(180):
                number = len(rows) + 1
                rows.append(point(number, rng.uniform(-8, 8), rng.uniform(-8, 8), lat=lat,
                    species=('bungalon', 'rhizophora')[i % 2], spacing=(1, 2, 3)[i % 3],
                    status='planted' if i % 17 == 0 else 'planned', assigned=i % 19 == 0))
        for cleanup in (database._cleanup_same_species_spacing_conflicts, database._cleanup_cross_species_spacing_conflicts):
            fast, exhaustive = CleanupConnection(rows), CleanupConnection(rows)
            count = cleanup(fast)
            with patch.object(database, '_CleanupNeighborGrid', ExhaustiveNeighbors):
                expected_count = cleanup(exhaustive)
            self.assertEqual(fast.deleted, exhaustive.deleted)
            self.assertEqual(count, expected_count)
            self.assertTrue(count > 0)
            protected = {r['id'] for r in rows if r['status'] != 'planned' or r['has_active_assignment']}
            self.assertFalse(protected & fast.deleted)

    def test_cross_species_check_skips_distant_pairs(self):
        rows = [point(i+1, i*4, 0, species='bungalon') for i in range(1000)]
        rows += [point(i+1001, i*4, 4, species='rhizophora') for i in range(1000)]
        conn = CleanupConnection(rows)
        with patch.object(database, '_local_distance_m', wraps=database._local_distance_m) as distance:
            self.assertEqual(database._cleanup_cross_species_spacing_conflicts(conn), 0)
        self.assertLess(distance.call_count, 1000)  # Exhaustive scan evaluates 1,000,000 pairs.
