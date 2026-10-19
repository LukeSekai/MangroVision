"""Selecting owned land uses the authenticated staff boundary."""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shapely.geometry import box, shape, mapping, MultiPolygon, Polygon, Point
from shapely import get_coordinates
from math import sin
import json
import planting_database as db

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
from api.routes import project_sites as routes

BODY = {'organization_id': 2, 'planting_point_ids': [10,12], 'name': 'Chosen points'}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(routes, 'get_user_by_session_token', lambda _: {'id': 17,'role': 'planner'})
    monkeypatch.setattr(routes, 'is_processing_active', lambda: False)
    monkeypatch.setattr(routes, '_load_zone_coverage', lambda: box(0,0,1,1))
    app = FastAPI()
    app.include_router(routes.router, prefix='/api/project-sites')
    with TestClient(app) as client:
        class API:
            def post(self, path, **kwargs):
                return client.post('/api/project-sites' + path, **kwargs)
            def patch(self, path, **kwargs):
                return client.patch('/api/project-sites' + path, **kwargs)
        yield API()


@pytest.mark.parametrize('user,status', [(None,401),({'id': 1,'role': 'planter'},403)])
def test_staff_authentication_is_required(api, monkeypatch, user, status):
    monkeypatch.setattr(routes, 'get_user_by_session_token', lambda _: user)
    assert api.post('/from-points',json=BODY).status_code == status


def test_manual_creation_is_retired_and_does_not_claim_land(api, monkeypatch):
    def no_coverage():
        pytest.fail('Retired creation must not read coverage or write a site.')
    monkeypatch.setattr(routes, '_load_zone_coverage', no_coverage)
    response = api.post('/from-points',json=BODY)
    assert response.status_code == 410 and 'Assign Points' in response.json()['detail']
    response = api.post('',json={'organization_id':2,'name':'Old form','geometry':mapping(box(0,0,1,1))})
    assert response.status_code == 410
    for invalid in ({'user_id':4}, {'geometry':{}}, {'planting_point_ids':[]}, {'organization_id':0}):
        assert api.post('/from-points',json={**BODY,**invalid}).status_code == 422


@pytest.fixture
def drawing(api,monkeypatch):
    monkeypatch.setattr(routes,'list_project_sites',lambda: [])
    monkeypatch.setattr(routes,'list_site_zones',lambda: [])
    class Connection:
        def execute(self,_sql,params):
            self.count = int(shape(json.loads(params[0])).covers(shape({'type':'Point','coordinates':[.2,.2]})))
            return self
        def fetchall(self):
            return [{'id':1,'longitude':.2,'latitude':.2}] if self.count else []
        def close(self):
            pass
    monkeypatch.setattr(db,'_get_connection',Connection)
    return api


@pytest.mark.parametrize('coverage', [box(0,0,1,1),MultiPolygon([box(0,0,.4,1),box(.6,0,1,1)]),
    box(0,0,1,1).difference(box(.4,.4,.6,.6))])
def test_preview_trims_to_map_edges_holes_and_separate_parts(drawing,monkeypatch,coverage):
    monkeypatch.setattr(routes,'_load_zone_coverage',lambda: coverage)
    original = box(-1,-1,2,2)
    result = drawing.post('/validate-boundary',json={'organization_id':2,'geometry':mapping(original)})
    assert result.status_code == 200
    preview = result.json()
    assert preview['valid'] and preview['clipped'] and preview['point_count'] == 1
    assert shape(preview['geometry']).equals(original.intersection(coverage))
    unchanged = drawing.post('/validate-boundary',json={'organization_id':2,'geometry':preview['geometry']}).json()
    assert unchanged['clipped'] is False and shape(unchanged['geometry']).equals(coverage)


def test_direct_save_and_boundary_update_receive_only_clipped_geometry(drawing,monkeypatch):
    received = []
    body = {'organization_id':2,'name':'Coastal site','geometry':mapping(box(-1,-1,.6,.6))}
    assert drawing.post('',json=body).status_code == 410
    assert received == []
    monkeypatch.setattr(routes,'get_project_site',lambda _: {'properties':{'organization_id':2}})
    monkeypatch.setattr(routes,'update_project_site',lambda **values: received.append(values) or {'id':30})
    assert drawing.patch('/30',json={'geometry':body['geometry']}).status_code == 200
    assert shape(received[0]['geometry']).equals(box(0,0,.6,.6))


@pytest.mark.parametrize('boundary',[box(2,2,3,3),box(.7,.7,.9,.9),box(1,0,2,1)])
def test_empty_or_outside_drawings_cannot_be_saved(drawing,monkeypatch,boundary):
    body = {'organization_id':2,'name':'Empty','geometry':mapping(boundary)}
    for path in ('/validate-boundary',):
        values = body if not path else {key:value for key,value in body.items() if key!='name'}
        response = drawing.post(path,json=values)
        assert response.status_code == 400 and 'at least one mapped point' in response.json()['detail']


def test_clipping_does_not_allow_claiming_another_organizations_area(drawing,monkeypatch):
    monkeypatch.setattr(routes,'list_project_sites',lambda: [
        {'geometry':mapping(box(.1,.1,.4,.4)),'properties':{'organization_id':9}}])
    response = drawing.post('/validate-boundary',json={'organization_id':2,'geometry':mapping(box(-1,-1,.6,.6))})
    assert response.status_code == 400 and 'another organization' in response.json()['detail']


def noisy_site():
    unit=1/111320
    def coordinate(x,y):
        return (122.627+x*unit,10.780+y*unit)
    vertices=[]
    for n in range(80): vertices.append(coordinate(n/4,.12*sin(n*1.9)))
    for n in range(80): vertices.append(coordinate(20+.15*sin(n*1.9),n/4))
    for n in range(80): vertices.append(coordinate(20-n/4,20+.15*sin(n*1.9)))
    for n in range(80): vertices.append(coordinate(.15*sin(n*1.9),20-n/4))
    return Polygon(vertices),coordinate


def prepare_with_points(monkeypatch,geometry,coverage,locations,allowed=None):
    from mangrovision_db.site_point_selection import prepare_drawn_site_boundary
    class Connection:
        def execute(self,*_):
            return self
        def fetchall(self):
            return [{'id':index,'longitude':x,'latitude':y} for index,(x,y) in enumerate(locations,1)]
        def close(self):
            pass
    monkeypatch.setattr(db,'_get_connection',Connection)
    return prepare_drawn_site_boundary(mapping(geometry),coverage,allowed)


def test_cleanup_reduces_tracing_noise_and_saving_keeps_the_preview(monkeypatch):
    original,xy=noisy_site()
    coverage=box(*xy(-2,-2),*xy(22,22))
    locations=[xy(1,1),xy(10,10),xy(19,19),xy(-1,10)]
    preview=prepare_with_points(monkeypatch,original,coverage,locations)
    cleaned=shape(preview['geometry'])
    assert preview['cleaned'] and preview['point_count']==3
    assert len(get_coordinates(cleaned))<len(get_coordinates(original))/2
    assert cleaned.is_valid and coverage.covers(cleaned)
    assert [original.covers(Point(p)) for p in locations]==[cleaned.covers(Point(p)) for p in locations]
    repeated=prepare_with_points(monkeypatch,cleaned,coverage,locations)
    assert shape(repeated['geometry']).equals(cleaned)
    assert repeated['point_count']==3 and repeated['cleaned'] is False


def test_cleanup_cannot_add_a_nearby_unselected_point_or_enter_a_blocked_area(monkeypatch):
    original,xy=noisy_site()
    coverage=box(*xy(-2,-2),*xy(22,22))
    outside=xy(19.95,6/4)
    assert not original.covers(Point(outside))
    locations=[xy(1,1),xy(10,10),outside]
    preview=prepare_with_points(monkeypatch,original,coverage,locations)
    cleaned=shape(preview['geometry'])
    assert preview['point_count']==2 and not cleaned.covers(Point(outside))
    constrained=prepare_with_points(monkeypatch,original,coverage,locations,allowed=original)
    assert original.covers(shape(constrained['geometry']))


def test_cleanup_keeps_irregular_map_edges_and_coverage_holes(monkeypatch):
    original,xy=noisy_site()
    coverage=box(*xy(-2,-2),*xy(10,22)).difference(box(*xy(4,4),*xy(6,6)))
    preview=prepare_with_points(monkeypatch,original,coverage,[xy(1,1),xy(8,8)])
    cleaned=shape(preview['geometry'])
    assert preview['clipped'] and coverage.covers(cleaned)
    assert original.intersection(coverage).intersection(coverage.boundary).equals(cleaned.intersection(coverage.boundary))


def test_small_crossing_in_a_freehand_outline_is_repaired(monkeypatch):
    _,xy=noisy_site()
    original=Polygon([xy(0,0),xy(10,10),xy(0,10),xy(10,0),xy(0,0)])
    assert not original.is_valid
    result=prepare_with_points(monkeypatch,original,box(*xy(-2,-2),*xy(12,12)),[xy(5,2),xy(5,8)])
    assert result['cleaned'] and result['point_count']==2 and shape(result['geometry']).is_valid
