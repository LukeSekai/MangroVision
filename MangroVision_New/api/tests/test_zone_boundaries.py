"""Zone writes must stay inside the actual visible drone-map footprint."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.routes import zones


class ZoneBoundaryTests(unittest.TestCase):
    def setUp(self):
        # An indentation which lies inside the image's rectangular bounds.
        self.coverage = Polygon([(0, 0), (8, 0), (8, 8), (5, 8),
                                 (5, 3), (3, 3), (3, 8), (0, 8), (0, 0)])
        app = FastAPI()
        app.include_router(zones.router, prefix="/api/zones")
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.real_loader = zones._load_zone_coverage
        self.mocks = {}
        for name, value in {
            "_require_lgu_user": {"id": 1, "role": "admin"},
            "is_processing_active": False,
            "_load_zone_coverage": self.coverage,
            "create_zone": {"id": 7},
            "update_zone": {"id": 7},
            "create_warning_zone": 7,
            "get_warning_zone": {"id": 7},
            "update_warning_zone": {"id": 7},
            "replace_zone_collection": {"type": "FeatureCollection", "features": []},
            "invalidate_eroded_zone_cache": None,
        }.items():
            patcher = patch.object(zones, name, return_value=value)
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def create(self, kind, geometry):
        body = {"name": "Test zone", "geometry": geometry} if kind == "warnings" else {
            "type": "Feature", "properties": {"name": "Test zone"}, "geometry": geometry,
        }
        return self.client.post(f"/api/zones/{kind}", json=body)

    def assert_no_write(self):
        for name in ["create_zone", "create_warning_zone", "update_warning_zone", "replace_zone_collection"]:
            self.mocks[name].assert_not_called()

    def test_outside_partial_and_bridge_over_irregular_edge_are_rejected(self):
        bridge = box(2, 4, 6, 6)
        self.assertTrue(all(self.coverage.covers(Point(x, y)) for x, y in bridge.exterior.coords))
        candidates = [box(9, 9, 10, 10), box(2, 4, 4, 6), bridge]
        for polygon in candidates:
            for kind in ["warnings", "eroded"]:
                with self.subTest(kind=kind, geometry=polygon.wkt):
                    response = self.create(kind, mapping(polygon))
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("outside the mapped area", response.json()["detail"])
        self.assert_no_write()

    def test_inside_and_edge_touching_zones_save_normally(self):
        for polygon in [box(.5, .5, 2, 2), box(0, 0, 2, 2),
                        MultiPolygon([box(.5, .5, 1, 1), box(6, 6, 7, 7)])]:
            for kind in ["warnings", "eroded"]:
                with self.subTest(kind=kind, geometry=polygon.wkt):
                    self.assertEqual(self.create(kind, mapping(polygon)).status_code, 200)
        self.assertEqual(self.mocks["create_zone"].call_count, 3)
        self.assertEqual(self.mocks["create_warning_zone"].call_count, 3)

    def test_holes_and_disconnected_components_are_not_ignored(self):
        with_hole = Polygon(box(0, 0, 8, 8).exterior.coords, [box(3, 3, 5, 5).exterior.coords])
        self.mocks["_load_zone_coverage"].return_value = with_hole
        for candidate in [box(2, 2, 6, 6), MultiPolygon([box(1, 1, 2, 2), box(9, 9, 10, 10)])]:
            for kind in ["warnings", "eroded"]:
                self.assertEqual(self.create(kind, mapping(candidate)).status_code, 400)
        self.assert_no_write()

    def test_invalid_or_empty_geometry_never_reaches_storage(self):
        candidates = [None, {}, mapping(Point(1, 1)), {"type": "Polygon", "coordinates": []},
                      mapping(Polygon([(1, 1), (2, 2), (1, 2), (2, 1), (1, 1)])),
                      mapping(box(200, 1, 201, 2))]
        for geometry in candidates:
            with self.subTest(geometry=geometry):
                response = self.client.post("/api/zones/eroded", json={"geometry": geometry})
                self.assertEqual(response.status_code, 400)
                self.assertIn("boundary is invalid", response.json()["detail"])
        self.assert_no_write()
        self.mocks["_load_zone_coverage"].assert_not_called()

    def test_bulk_replacement_checks_every_zone_before_writing(self):
        features = [{"geometry": mapping(box(1, 1, 2, 2))}, {"geometry": mapping(box(9, 9, 10, 10))}]
        response = self.client.put("/api/zones/eroded", json={"features": features})
        self.assertEqual(response.status_code, 400)
        self.assert_no_write()
        self.mocks["_load_zone_coverage"].assert_called_once()

    def test_warning_geometry_updates_and_preflight_use_the_same_check(self):
        geometry = mapping(box(2, 4, 4, 6))
        response = self.client.patch("/api/zones/warnings/7", json={"geometry": geometry})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/zones/validate-boundary", json={"geometry": geometry})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/zones/validate-boundary", json={"geometry": mapping(box(1, 1, 2, 2))})
        self.assertEqual(response.json(), {"valid": True})
        self.assert_no_write()

    def test_unavailable_coverage_fails_closed_but_old_zones_can_be_cleared(self):
        self.mocks["_load_zone_coverage"].side_effect = RuntimeError("Unavailable")
        geometry = mapping(box(1, 1, 2, 2))
        for kind in ["warnings", "eroded"]:
            with self.assertLogs(zones.logger, level="WARNING"):
                response = self.create(kind, geometry)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"], "Could not check the map boundary. Please try again.")
        self.assert_no_write()
        response = self.client.put("/api/zones/eroded", json={"features": []})
        self.assertEqual(response.status_code, 200)
        self.mocks["replace_zone_collection"].assert_called_once_with("eroded", [], 1)
        response = self.client.patch("/api/zones/warnings/7", json={"notes": "Updated note"})
        self.assertEqual(response.status_code, 200)

    def test_loader_follows_visible_pixels_rather_than_raster_bounds(self):
        # Exercise the real footprint reader using a small georeferenced image.
        self.mocks["_load_zone_coverage"].side_effect = self.real_loader
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.tif"
            image = np.full((4, 64, 64), 110, dtype=np.uint8)
            image[3] = 0
            image[3, :48, :48] = 255
            with rasterio.open(path, "w", driver="GTiff", width=64, height=64,
                               count=4, dtype="uint8", crs="EPSG:4326",
                               transform=from_origin(122.6, 10.8, .00001, .00001)) as dataset:
                dataset.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
                dataset.write(image)
            inside = mapping(box(122.6001, 10.7997, 122.6002, 10.7998))
            outside = mapping(box(122.6005, 10.7997, 122.6006, 10.7998))
            partial = mapping(box(122.6004, 10.7997, 122.6005, 10.7998))
            with patch("canopy_detection.ortho_matcher._ensure_active_ortho", return_value={"path": path}), \
                 patch.object(zones, "feature_collection", return_value={"features": [
                     {"geometry": mapping(box(122.6, 10.79936, 122.60064, 10.8))}
                 ]}):
                for kind in ["warnings", "eroded"]:
                    self.assertEqual(self.create(kind, inside).status_code, 200)
                    self.assertEqual(self.create(kind, outside).status_code, 400)
                    self.assertEqual(self.create(kind, partial).status_code, 400)

    def test_eroded_overlap_is_confirmed_once_for_create_and_edit(self):
        from mangrovision_db.erosion import SeedlingOverlap, ZONE_OVERLAP_MESSAGE
        feature = {'geometry': mapping(box(1, 1, 2, 2)), 'properties': {'name': 'Erosion'}}
        for method, path, name in [('POST', '/api/zones/eroded', 'create_zone'),
                                   ('PATCH', '/api/zones/eroded/7', 'update_zone')]:
            self.mocks[name].side_effect = SeedlingOverlap(ZONE_OVERLAP_MESSAGE)
            response = self.client.request(method, path, json=feature)
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()['detail']['code'], 'seedling_overlap')
            self.mocks[name].side_effect = None
            response = self.client.request(method, path, json=feature,
                headers={'X-Confirm-Seedling-Overlap': 'true'})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(self.mocks[name].call_args.kwargs['confirm_overlap'])

    def test_eroded_edit_checks_map_processing_lock_and_missing_zone(self):
        response = self.client.patch('/api/zones/eroded/7', json={'geometry': mapping(box(9, 9, 10, 10))})
        self.assertEqual(response.status_code, 400)
        self.mocks['update_zone'].assert_not_called()
        self.mocks['is_processing_active'].return_value = True
        response = self.client.patch('/api/zones/eroded/7', json={'geometry': mapping(box(1, 1, 2, 2))})
        self.assertEqual(response.status_code, 409)
        self.mocks['update_zone'].assert_not_called()
        self.mocks['is_processing_active'].return_value = False
        self.mocks['update_zone'].side_effect = KeyError(7)
        response = self.client.patch('/api/zones/eroded/7', json={'geometry': mapping(box(1, 1, 2, 2))})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
