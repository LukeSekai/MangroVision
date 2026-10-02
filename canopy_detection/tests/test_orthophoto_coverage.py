"""The upload boundary and overlay must agree with GeoTIFF transparency."""

import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import Point, box, mapping

from canopy_detection.orthophoto_coverage import (
    active_visible_coverage, overlay_visibility_mask, point_visibility_flags,
)


class OrthophotoCoverageTests(unittest.TestCase):
    def test_irregular_visible_edge_and_overlay_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.tif"
            image = np.zeros((4, 64, 64), dtype=np.uint8)
            image[:3] = 110
            image[:3, 10, 10] = 0  # A tiny transparent hole inside the outer contour.
            for row in range(64):
                image[3, row, :max(0, 48-row//2)] = 255
            with rasterio.open(path, "w", driver="GTiff", width=64, height=64,
                               count=4, dtype="uint8", crs="EPSG:4326",
                               transform=from_origin(122.6, 10.8, .00001, .00001)) as dataset:
                dataset.colorinterp = (ColorInterp.red, ColorInterp.green,
                                       ColorInterp.blue, ColorInterp.alpha)
                dataset.write(image)
            entry = {"path": path}
            coverage = active_visible_coverage(entry)
            self.assertTrue(coverage.covers(Point(122.6001, 10.7999)))
            self.assertFalse(coverage.covers(Point(122.6006, 10.7999)))
            self.assertTrue(coverage.intersects(box(122.6001, 10.7998, 122.6006, 10.7999)))
            mask = overlay_visibility_mask(entry, 0, 0, 64, 64, 64, 64)
            self.assertEqual(mask[11, 11], 255)
            self.assertEqual(mask[10, 60], 0)
            self.assertEqual(mask[60, 30], 0)
            self.assertEqual(mask.shape, (64, 64))
            flags = point_visibility_flags(entry, coverage, [
                (10.7999, 122.6001),
                (10.7999, 122.6006),
            ])
            self.assertEqual(flags, [False, False])
            self.assertEqual(point_visibility_flags(entry, coverage, [
                (10.79988, 122.60012),
            ]), [True])

            # The footprint, rather than its center alone, controls the upload
            # decision. This includes images centered just beyond the edge.
            sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MangroVision_New"))
            from api.routes.processing import _classify_footprint_against_gis, _visible_map_point_features
            inside = mapping(box(122.60008, 10.79986, 122.60018, 10.79994))
            partial = mapping(box(122.60015, 10.79986, 122.60055, 10.79994))
            outside = mapping(box(122.60055, 10.79986, 122.60063, 10.79994))
            self.assertEqual(_classify_footprint_against_gis(inside, 10.7999, 122.60013, coverage)["status"], "inside")
            response = _classify_footprint_against_gis(partial, 10.7999, 122.60035, coverage)
            self.assertEqual(response["status"], "partial")
            self.assertGreater(response["estimated_inside_pct"], 0)
            self.assertLess(response["estimated_inside_pct"], 100)
            self.assertEqual(_classify_footprint_against_gis(outside, 10.7999, 122.60059, coverage)["status"], "outside")
            features = {"type": "FeatureCollection", "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [122.6001, 10.7999]}},
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [122.6006, 10.7999]}},
            ]}
            self.assertEqual(len(_visible_map_point_features(features, coverage)["features"]), 1)

            # An invalid saved candidate must fail before opening a database
            # transaction or uploading any image assets.
            from planting_database import OutsideVisibleMapError, save_analysis
            coverage_feature = {"geometry": mapping(box(122.6, 10.79936, 122.60064, 10.8))}
            with patch("canopy_detection.ortho_matcher._ensure_active_ortho", return_value=entry), \
                 patch("mangrovision_db.zones.feature_collection", return_value={"features": [coverage_feature]}), \
                 patch("planting_database._get_connection", side_effect=AssertionError("database touched")), \
                 patch("planting_database.upload_analysis_data_urls", side_effect=AssertionError("storage touched")):
                with self.assertRaises(OutsideVisibleMapError):
                    save_analysis("test.jpg", 10.7999, 122.6001, {},
                                  [{"_gps_lat": 10.7999, "_gps_lon": 122.6006}])


if __name__ == "__main__":
    unittest.main()
