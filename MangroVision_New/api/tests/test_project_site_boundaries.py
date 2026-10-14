"""Project sites must fit the selected owner's recorded planting area."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from shapely.geometry import MultiPolygon, Polygon, box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.routes import project_sites, zones


def feature(owner, geometry, site_id=1):
    return {"type": "Feature", "id": site_id,
            "properties": {"organization_id": owner, "name": f"Organization {owner}"},
            "geometry": mapping(geometry)}


class ProjectSiteBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.owned = feature(11, box(0, 0, 3, 3))
        self.other = feature(22, box(5, 5, 8, 8), 2)
        app = FastAPI()
        app.include_router(project_sites.router, prefix="/api/project-sites")
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.mocks = {}
        for name, value in {
            "_require_lgu_user": {"id": 1, "role": "admin"},
            "is_processing_active": False,
            "list_project_sites": [self.owned, self.other],
            "list_site_zones": [],
            "create_project_site": {"id": 3},
            "update_project_site": {"id": 1},
            "get_project_site": self.owned,
            "available_released_areas": {"type": "FeatureCollection", "features": []},
        }.items():
            patcher = patch.object(project_sites, name, return_value=value)
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(zones, "_load_zone_coverage", return_value=box(0, 0, 8, 8))
        self.map_boundary = patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, geometry, owner=11, preflight=False):
        body = {"geometry": mapping(geometry), "organization_id": owner}
        if not preflight:
            body["name"] = "New site"
        return self.client.post("/api/project-sites/validate-boundary" if preflight else "/api/project-sites", json=body)

    def assert_no_write(self):
        self.mocks["create_project_site"].assert_not_called()
        self.mocks["update_project_site"].assert_not_called()

    def test_other_owner_unassigned_land_and_partial_outside_are_rejected(self):
        for candidate in [box(6, 6, 7, 7), box(3.5, 3.5, 4.5, 4.5), box(2, 2, 4, 4),
                          MultiPolygon([box(1, 1, 2, 2), box(6, 6, 7, 7)])]:
            for preflight in [False, True]:
                response = self.post(candidate, preflight=preflight)
                self.assertEqual(response.status_code, 400)
                self.assertIn("outside the selected organization's area", response.json()["detail"])
        self.assert_no_write()

    def test_inside_owner_area_and_shared_edges_are_allowed(self):
        self.mocks["list_site_zones"].return_value = [feature(22, box(3, 0, 4, 3), 8)]
        candidate = box(1, 1, 3, 2)
        self.assertEqual(self.post(candidate, preflight=True).json(), {"valid": True})
        self.assertEqual(self.post(candidate).status_code, 201)
        self.mocks["create_project_site"].assert_called_once()
        self.assertEqual(self.mocks["create_project_site"].call_args.kwargs["organization_id"], 11)

    def test_overlap_with_another_owner_is_rejected_even_inside_own_outline(self):
        self.mocks["list_project_sites"].return_value = [feature(11, box(0, 0, 8, 8)), self.other]
        response = self.post(box(6, 6, 7, 7))
        self.assertEqual(response.status_code, 400)
        self.assertIn("overlaps another organization's area", response.json()["detail"])
        self.assert_no_write()

    def test_existing_sites_take_precedence_and_assignment_area_bootstraps_first_site(self):
        self.mocks["list_site_zones"].return_value = [feature(11, box(0, 0, 4, 4))]
        self.assertEqual(self.post(box(3.2, 3.2, 3.8, 3.8)).status_code, 400)
        self.assert_no_write()
        self.mocks["list_project_sites"].return_value = [self.other]
        self.assertEqual(self.post(box(3.2, 3.2, 3.8, 3.8)).status_code, 201)

    def test_no_recorded_area_or_failed_loading_never_allows_creation(self):
        response = self.post(box(1, 1, 2, 2), owner=99)
        self.assertEqual(response.status_code, 400)
        self.assertIn("No mapped area is assigned", response.json()["detail"])
        self.mocks["list_project_sites"].side_effect = RuntimeError("Unavailable")
        response = self.post(box(1, 1, 2, 2))
        self.assertEqual(response.status_code, 503)
        self.assertIn("Could not check the organization's area", response.json()["detail"])
        self.assert_no_write()

    def test_whole_surface_must_fit_including_gaps_in_the_organization_area(self):
        boundary = Polygon([(0, 0), (8, 0), (8, 8), (5, 8), (5, 3), (3, 3), (3, 8), (0, 8)])
        self.mocks["list_project_sites"].return_value = [feature(11, boundary)]
        response = self.post(box(2, 4, 6, 6))
        self.assertEqual(response.status_code, 400)
        self.assert_no_write()

    def test_visible_drone_map_restriction_still_applies_to_owned_areas(self):
        self.map_boundary.return_value = box(0, 0, 1.5, 3)
        response = self.post(box(1, 1, 2, 2))
        self.assertEqual(response.status_code, 400)
        self.assertIn("outside the mapped area", response.json()["detail"])
        self.assert_no_write()

    def test_geometry_edits_are_checked_against_the_saved_owner(self):
        response = self.client.patch("/api/project-sites/1", json={"geometry": mapping(box(6, 6, 7, 7))})
        self.assertEqual(response.status_code, 400)
        self.assert_no_write()
        response = self.client.patch("/api/project-sites/1", json={"geometry": mapping(box(1, 1, 2, 2))})
        self.assertEqual(response.status_code, 200)
        self.mocks["update_project_site"].assert_called_once()

    def test_released_land_is_available_to_another_owner_and_legacy_assignment_hulls_do_not_reclaim_it(self):
        self.mocks['available_released_areas'].return_value = {'features': [feature(None, box(3.5, 3.5, 4.5, 4.5))]}
        self.mocks['list_site_zones'].return_value = [feature(11, box(0, 0, 4.5, 4.5))]
        response = self.post(box(3.7, 3.7, 4.2, 4.2), owner=22)
        self.assertEqual(response.status_code, 201)


if __name__ == "__main__":
    unittest.main()
