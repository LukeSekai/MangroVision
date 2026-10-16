"""Preview spacing shapes share the point projection without changing safety."""
import copy
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.routes import processing as p
from api.tests import test_eroded_analysis as erosion_fixture


class HexagonPreviewProjectionTests(unittest.TestCase):
    lat, lon = 10.8, 122.6
    longitude_scale = 111320 * math.cos(math.radians(lat))
    radius = 1 / math.sqrt(3)

    def point(self, east=0, north=0, center=(100., 100.), gsd=.01):
        detector = object.__new__(p.HexagonDetector)
        return dict(center=center, _gps_lat=self.lat+north/111320,
                    _gps_lon=self.lon+east/self.longitude_scale,
                    buffer_radius_m=self.radius, _eroded_unavailable=True,
                    buffer=detector.create_hexagon(*center, self.radius/gsd),
                    core=detector.create_hexagon(*center, self.radius/gsd*.2))

    def projection(self, matrix):
        def project(lat, lon):
            metric=np.array([(lon-self.lon)*self.longitude_scale, (lat-self.lat)*111320, 1.])
            projected=matrix @ metric
            return tuple(projected[:2]/projected[2])
        return project

    def test_scale_mismatch_is_corrected_without_moving_centers_or_planting_cores(self):
        # As in Analysis 35: the camera GSD is .726 cm/px, but the measured
        # alignment scale is .850 cm/px. The prior visual buffers overlap.
        camera_gsd, alignment_gsd = .00726113585746102, .0085018
        project = self.projection(np.array([[1/alignment_gsd,0,100], [0,1/alignment_gsd,100], [0,0,1.]]))
        source = [self.point(gsd=camera_gsd)]
        visual = p._project_visual_spacing_buffers(source, project, self.lat, self.radius)
        actual_radius = np.linalg.norm(np.asarray(visual[0]['buffer'].exterior.coords)[0]-source[0]['center'])
        self.assertAlmostEqual(actual_radius, self.radius/alignment_gsd, places=5)
        self.assertEqual(visual[0]['center'], source[0]['center'])
        self.assertEqual(visual[0]['core'].wkb, source[0]['core'].wkb)
        self.assertEqual(visual[0]['_gps_lat'], source[0]['_gps_lat'])
        self.assertTrue(visual[0]['_eroded_unavailable'])
        self.assertAlmostEqual(source[0]['buffer'].bounds[2]-100, self.radius/camera_gsd)

    def test_rotation_and_perspective_use_projected_vertices_but_keep_authoritative_center(self):
        matrix=np.array([[70.,-30.,20], [20.,110.,40], [.04,-.02,1.]])
        project=self.projection(matrix)
        source=[self.point(east=2, north=3, center=(421.,287.))]
        visual=p._project_visual_spacing_buffers(source, project, self.lat, self.radius)[0]
        lat,lon=source[0]['_gps_lat'],source[0]['_gps_lon']
        origin=np.asarray(project(lat,lon))
        expected=[]
        for i in range(6):
            angle=i*math.pi/3
            vertex=project(lat+self.radius*math.sin(angle)/111320,
                           lon+self.radius*math.cos(angle)/self.longitude_scale)
            expected.append(np.asarray(source[0]['center'])+vertex-origin)
        np.testing.assert_allclose(np.asarray(visual['buffer'].exterior.coords)[:6], expected)
        self.assertEqual(visual['center'], (421.,287.))
        self.assertEqual(visual['core'].wkb, source[0]['core'].wkb)

    def test_missing_or_failed_projection_keeps_every_original_point(self):
        source=[self.point()]
        for project in (None, lambda *_: (float('nan'),0), lambda *_: (0,0)):
            visual=p._project_visual_spacing_buffers(source,project,self.lat,self.radius)
            self.assertEqual(len(visual),1)
            self.assertEqual(visual[0]['buffer'].wkb,source[0]['buffer'].wkb)
            self.assertEqual(visual[0]['core'].wkb,source[0]['core'].wkb)
        self.assertEqual(p._project_visual_spacing_buffers([],None,None,self.radius),[])

    def test_oversized_tiles_merge_before_correction_and_separate_afterward(self):
        camera_gsd, alignment_gsd = .00726113585746102, .0085018
        project=self.projection(np.array([[1/alignment_gsd,0,150], [0,-1/alignment_gsd,700], [0,0,1.]]))
        points=[]
        for column in range(4):
            for row in range(4):
                east=column*1.5*self.radius
                north=row+(column % 2)*.5
                center=project(self.lat+north/111320,self.lon+east/self.longitude_scale)
                points.append(self.point(east,north,center=center,gsd=camera_gsd))
        def visible_components(hexagons):
            mask=np.zeros((900,900),np.uint8)
            for h in hexagons:
                center=np.asarray(h['center'])
                polygon=(np.asarray(h['buffer'].exterior.coords)-center)*.88+center
                cv2.fillPoly(mask,[polygon.astype(np.int32)],255)
            return cv2.connectedComponents(mask)[0]-1
        self.assertEqual(visible_components(points),1)
        visual=p._project_visual_spacing_buffers(points,project,self.lat,self.radius)
        self.assertEqual(visible_components(visual),len(points))

    def test_complete_pipeline_changes_only_display_buffers_not_points_exports_or_safety(self):
        fixture=erosion_fixture.ErodedAnalysisTests()
        self.addCleanup(fixture.doCleanups)
        erosion=fixture.area(0,80,160,160)
        forbidden=fixture.area(0,0,30,160)
        with patch.object(p,'_project_visual_spacing_buffers',side_effect=lambda points,*_: copy.deepcopy(points)):
            before,cached_before=fixture.run_analysis(erosion,forbidden,vegetation=True,projection_scale=1.17)
        after,cached_after=fixture.run_analysis(erosion,forbidden,vegetation=True,projection_scale=1.17)
        self.assertGreater(len(after['map']['coordinates']),0)
        self.assertEqual(after['map']['coordinates'],before['map']['coordinates'])
        self.assertEqual(after['map']['safe_points_geojson']['features'],before['map']['safe_points_geojson']['features'])
        self.assertEqual(after['exports']['waypoints'],before['exports']['waypoints'])
        for name in ('hexagon_count','safe_hexagon_count','eroded_unavailable_count','forbidden_filtered_count'):
            self.assertEqual(after['metrics'][name],before['metrics'][name])
        np.testing.assert_array_equal(cached_after['results']['danger_mask'],cached_before['results']['danger_mask'])
        self.assertTrue(any(row['_eroded_unavailable'] for row in cached_after['safe_hexagons']))
        for safe,visual in zip(cached_after['safe_hexagons'],cached_after['rendered_hexagons']):
            self.assertEqual(safe['center'],visual['center'])
            self.assertEqual(safe['core'].wkb,visual['core'].wkb)
            self.assertLess(visual['buffer'].area,safe['buffer'].area)
