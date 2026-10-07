"""Navigation follows mapped roads and continues with local point guidance."""
import importlib
import io
import json
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def routing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    module = importlib.import_module('api.routes.routing')
    monkeypatch.setattr(module, '_load_google_routes_api_key', lambda: 'test-key')
    monkeypatch.setattr(module, 'reserve_google_request', lambda key: None)
    return module


def encode(points):
    last = [0, 0]
    result = ''
    for point in points:
        for axis in range(2):
            value = round(point[axis]*100000)
            delta = value-last[axis]
            last[axis] = value
            delta = ~(delta << 1) if delta < 0 else delta << 1
            while delta >= 0x20:
                result += chr((0x20 | (delta & 0x1f))+63)
                delta >>= 5
            result += chr(delta+63)
    return result


def request(routing, origin=(10.783, 122.619), destination=(10.78012, 122.62707)):
    return routing.RouteRequest(origin_lat=origin[0], origin_lon=origin[1], dest_lat=destination[0], dest_lon=destination[1])


def assert_navigation_endpoints(result, origin, destination):
    assert result['origin'] == list(origin)
    assert result['target'] == list(destination)
    assert all(segment['kind'] in {'road', 'guidance'} for segment in result['segments'])
    assert result['navigation_path'] == [p for segment in result['segments'] for p in segment['polyline']]
    guides = [segment for segment in result['segments'] if segment['kind'] == 'guidance']
    assert len(guides) <= 1
    if guides:
        start = list(origin) if result['route_source'] == 'within_site' else result['entrance'] or result['polyline'][-1]
        assert guides[0]['polyline'] == [start, list(destination)]
        assert result['segments'][-1] == guides[0]


def test_shortest_alternative_selected_and_white_road_appended(routing, monkeypatch):
    from api.site_access import access_routes, path_distance
    site = access_routes()[0]
    origin = [10.783, 122.619]
    junction = site['access_path'][0]
    captured = []

    def provider(http_request, **kwargs):
        body = json.loads(http_request.data)
        captured.append(body)
        routes = [dict(distanceMeters=meters, duration=f'{seconds}s',
                       polyline={'encodedPolyline': encode([origin, junction])})
                  for meters, seconds in [(1200, 700), (180, 150), (240, 100)]]
        return io.BytesIO(json.dumps({'routes': routes}).encode())

    monkeypatch.setattr(routing, 'urlopen', provider)
    result = routing.compute_route(request(routing, origin))
    assert captured[0]['computeAlternativeRoutes'] is True
    assert captured[0]['travelMode'] == 'WALK'
    assert captured[0]['destination']['location']['latLng'] == {'latitude': junction[0], 'longitude': junction[1]}
    assert result['polyline'][-len(site['access_path']):] == site['access_path']
    assert 180+path_distance(site['access_path']) <= result['distance_m'] < 182+path_distance(site['access_path'])
    assert result['route_source'] == 'site_route'
    assert result['target'] != result['polyline'][-1]
    assert_navigation_endpoints(result, origin, (10.78012,122.62707))
    assert result['segments'][-2] == {'kind': 'road', 'polyline': site['access_path']}
    assert result['segments'][-1] == {'kind': 'guidance', 'polyline': [site['access_path'][-1], result['target']]}
    assert result['road_route_available'] is True


def test_already_on_white_road_joins_nearest_segment_without_backtracking(routing, monkeypatch):
    from api.site_access import access_routes, path_distance
    path = access_routes()[0]['access_path']
    monkeypatch.setattr(routing, '_google_route', lambda *args: pytest.fail('Already on access road'))
    midpoint = [(a+b)/2 for a,b in zip(path[9], path[10])]
    result = routing.compute_route(request(routing, midpoint))
    assert result['polyline'][0] == midpoint
    assert result['polyline'][-1] == path[-1]
    assert path[0] not in result['polyline']
    assert result['distance_m'] < path_distance(path[9:])
    assert_navigation_endpoints(result, midpoint, (10.78012,122.62707))


def test_pond_position_does_not_snap_to_access_road(routing):
    from api.site_access import access_routes, remaining_access_path
    assert remaining_access_path([10.7822,122.6240], access_routes()[0]['access_path']) is None


def test_inside_site_guides_to_exact_point_without_routing_back_out(routing, monkeypatch):
    monkeypatch.setattr(routing, '_google_route', lambda *args: pytest.fail('Already inside site'))
    result = routing.compute_route(request(routing, (10.78012,122.62708)))
    assert result['polyline'] == []
    assert result['route_source'] == 'within_site'
    assert result['distance_m'] is None
    assert 'straight-line' in result['distance_label']
    assert_navigation_endpoints(result, (10.78012,122.62708), (10.78012,122.62707))
    assert result['segments'] == [{'kind': 'guidance', 'polyline': [[10.78012,122.62708], [10.78012,122.62707]]}]


def test_nasugban_point_198_at_traced_shoreline_uses_field_guidance(routing, monkeypatch):
    # Actual saved point #198 is about 0.6 m outside the hand-traced outline.
    destination = (10.7798913043478, 122.627158010418)
    monkeypatch.setattr(routing, '_google_route', lambda *args: pytest.fail('Off-road planting point sent to Google'))
    result = routing.compute_route(request(routing, destination, destination))
    assert result['route_source'] == 'within_site'
    assert result['target'] == list(destination)
    assert result['polyline'] == []
    assert 'near the site boundary' in result['navigation_note']


def test_phone_gps_uncertainty_near_boundary_does_not_send_planter_back_to_road(routing, monkeypatch):
    destination = (10.7798913043478, 122.627158010418)
    origin = (destination[0] - 0.000085, destination[1])
    monkeypatch.setattr(routing, '_google_route', lambda *args: pytest.fail('GPS uncertainty should use field guidance'))
    body = request(routing, origin, destination)
    body.origin_accuracy_m = 12
    result = routing.compute_route(body)
    assert result['route_source'] == 'within_site'
    assert result['polyline'] == []
    assert 'GPS may drift' in result['navigation_note']
    assert result['segments'] == [{'kind': 'guidance', 'polyline': [list(origin), list(destination)]}]


def test_large_gps_error_cannot_bypass_entrance_route(routing, monkeypatch):
    calls = []
    def provider(origin, destination, mode):
        calls.append(destination)
        return 100, 90, [origin, destination]
    monkeypatch.setattr(routing, '_google_route', provider)
    body = request(routing)
    body.origin_accuracy_m = 5000
    result = routing.compute_route(body)
    assert calls
    assert result['route_source'] == 'site_route'


def test_boundary_tolerance_does_not_include_unrelated_pond(routing):
    from api.site_access import access_for_destination, inside_or_near_area
    assert access_for_destination([10.7822, 122.6240]) is None
    square = [[0,0], [0,0.001], [0.001,0.001], [0.001,0]]
    assert inside_or_near_area([0.0005, 0], square)
    assert inside_or_near_area([-0.00001, 0.0005], square)
    assert not inside_or_near_area([-0.0001, 0.0005], square)


def test_other_sites_are_not_forced_through_leganes_entrance(routing, monkeypatch):
    calls = []
    def provider(origin, destination, mode):
        calls.append(destination)
        return 100, 90, [origin, destination]
    monkeypatch.setattr(routing, '_google_route', provider)
    result = routing.compute_route(request(routing, (11,122), (11.001,122)))
    assert calls == [[11.001,122]]
    assert result['entrance'] is None


def test_eastern_road_snap_rejected(routing, monkeypatch):
    def provider(*args, **kwargs):
        return io.BytesIO(json.dumps({'routes': [{'distanceMeters': 1500, 'duration': '1500s',
            'polyline': {'encodedPolyline': encode([[10.783,122.619], [10.782,122.628]])}}]}).encode())
    monkeypatch.setattr(routing, 'urlopen', provider)
    from api.site_access import access_routes
    with pytest.raises(HTTPException, match='No route'):
        routing._google_route([10.783,122.619], access_routes()[0]['access_path'][0], 'walking')
    result = routing.compute_route(request(routing))
    assert result['route_source'] == 'partial_route'
    assert result['polyline'] == access_routes()[0]['access_path']
    assert [10.782,122.628] not in result['polyline']


def test_provider_failure_keeps_roads_and_labels_direct_guidance(routing, monkeypatch):
    from urllib.error import URLError
    def provider(*args, **kwargs):
        raise URLError('unavailable')
    monkeypatch.setattr(routing, 'urlopen', provider)
    from api.site_access import access_routes
    result = routing.compute_route(request(routing))
    assert result['route_source'] == 'partial_route'
    assert result['polyline'] == access_routes()[0]['access_path']
    assert result['distance_m'] is None
    assert result['duration_s'] is None
    assert_navigation_endpoints(result, (10.783,122.619), (10.78012,122.62707))
    path = access_routes()[0]['access_path']
    assert result['segments'] == [{'kind':'road','polyline':path},
        {'kind':'guidance','polyline':[path[-1],result['target']]}]
    assert result['road_route_available'] is False


def test_remote_provider_failure_keeps_local_point_guide_without_remote_shortcut(routing, monkeypatch):
    from api.site_access import access_routes
    monkeypatch.setattr(routing, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'{"routes": []}'))
    app = FastAPI()
    app.include_router(routing.router)
    point = [10.7798913043478,122.627158010418]
    with TestClient(app) as client:
        response = client.post('/compute', json=dict(origin_lat=10.70, origin_lon=122.55,
            dest_lat=point[0], dest_lon=point[1], origin_accuracy_m=12))
    assert response.status_code == 200
    result = response.json()
    path = access_routes()[0]['access_path']
    assert result['route_source'] == 'partial_route'
    assert result['target'] == point
    assert result['entrance'] == path[-1]
    assert result['access_start'] == path[0]
    assert result['polyline'] == path
    assert [10.70,122.55] not in result['polyline']
    assert result['distance_m'] is None and result['duration_s'] is None
    assert result['distance_label'] == 'Road directions unavailable'
    assert 'Road directions from your location are unavailable' in result['navigation_note']
    assert_navigation_endpoints(result, (10.70,122.55), point)
    assert result['segments'][-1] == {'kind':'guidance','polyline':[path[-1],point]}
    assert all([10.70,122.55] not in segment['polyline'] for segment in result['segments'])


def test_unmapped_site_still_shows_point_when_provider_has_no_route(routing, monkeypatch):
    monkeypatch.setattr(routing, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'{"routes": []}'))
    result = routing.compute_route(request(routing, (11,122), (11.001,122)))
    assert result['route_source'] == 'point_guidance'
    assert result['target'] == [11.001,122]
    assert result['polyline'] == []
    assert result['entrance'] is None and result['access_start'] is None
    assert_navigation_endpoints(result, (11,122), (11.001,122))
    assert result['segments'] == []


@pytest.mark.parametrize('status', [502,503])
def test_provider_outage_or_missing_key_keeps_mapped_access_guidance(routing, monkeypatch, status):
    def unavailable(*args):
        raise HTTPException(status_code=status, detail='Provider unavailable')
    monkeypatch.setattr(routing, '_google_route', unavailable)
    assert routing.compute_route(request(routing))['route_source'] == 'partial_route'


def test_off_road_phone_origin_is_preserved_when_google_snaps_to_nearby_street(routing, monkeypatch):
    from api.site_access import access_routes
    origin = [10.78,122.61]
    street_start = [10.781,122.61]  # over 100 m from a house's GPS fix
    junction = access_routes()[0]['access_path'][0]
    def provider(http_request, **kwargs):
        sent = json.loads(http_request.data)['origin']['location']['latLng']
        assert sent == {'latitude': origin[0], 'longitude': origin[1]}
        return io.BytesIO(json.dumps({'routes': [dict(distanceMeters=1500,duration='1200s',
            polyline={'encodedPolyline':encode([street_start,junction])})]}).encode())
    monkeypatch.setattr(routing, 'urlopen', provider)
    result = routing.compute_route(request(routing, origin))
    assert result['route_source'] == 'site_route'
    assert_navigation_endpoints(result, origin, (10.78012,122.62707))
    assert result['segments'][0]['kind'] == 'road'
    assert result['segments'][0]['polyline'][0] == street_start


def test_gap_between_google_road_and_access_road_is_not_invented(routing, monkeypatch):
    from api.site_access import access_routes
    site = access_routes()[0]
    endpoint = [site['access_path'][0][0]+0.00005,site['access_path'][0][1]]
    monkeypatch.setattr(routing, '_google_route', lambda origin, *args: (100,90,[origin,endpoint]))
    result = routing.compute_route(request(routing))
    assert_navigation_endpoints(result, (10.783,122.619), (10.78012,122.62707))
    assert result['segments'][0]['polyline'][-1] == endpoint
    assert result['segments'][1]['polyline'][0] == site['access_path'][0]
    assert len(result['segments']) == 3


def test_cict_point_206_uses_access_road_without_cross_water_shortcut(routing, monkeypatch):
    from api.site_access import access_routes
    destination = [10.7799452030183,122.626825821686]
    origin = [10.70,122.55]
    junction = access_routes()[0]['access_path'][0]
    street_path = [origin,[10.76,122.55],[10.783,122.58],junction]
    monkeypatch.setattr(routing,'_google_route',lambda start,end,mode:(12000,9000,street_path))
    result = routing.compute_route(request(routing,origin,destination))
    assert result['route_source'] == 'site_route'
    assert result['segments'][0]['polyline'] == street_path
    assert result['segments'][1]['polyline'] == access_routes()[0]['access_path']
    assert result['target'] == destination
    assert_navigation_endpoints(result,origin,destination)
    assert result['segments'][-1] == {'kind':'guidance',
        'polyline':[access_routes()[0]['access_path'][-1],destination]}
    assert all(origin not in segment['polyline'] for segment in result['segments'] if segment['kind']=='guidance')


def test_unmapped_site_route_continues_from_nearby_road_endpoint_to_exact_point(routing, monkeypatch):
    destination = [11.001,122]
    endpoint = [11.00091,122]
    monkeypatch.setattr(routing, '_google_route', lambda origin, *args: (100,90,[origin,endpoint]))
    result = routing.compute_route(request(routing,(11,122),destination))
    assert result['segments'][-1] == {'kind':'guidance','polyline':[endpoint,destination]}
    assert_navigation_endpoints(result,(11,122),destination)


def test_other_errors_are_not_hidden_as_navigation_guidance(routing, monkeypatch):
    def failure(*args):
        raise HTTPException(status_code=422, detail='Invalid navigation request')
    monkeypatch.setattr(routing, '_google_route', failure)
    with pytest.raises(HTTPException) as error:
        routing.compute_route(request(routing))
    assert error.value.status_code == 422


@pytest.mark.parametrize('lat,lon', [(91,122), (10,181), (-91,122)])
def test_invalid_coordinates_rejected(routing, lat, lon):
    app = FastAPI()
    app.include_router(routing.router)
    with TestClient(app) as client:
        assert client.post('/compute', json=dict(origin_lat=lat, origin_lon=lon, dest_lat=10, dest_lon=122)).status_code == 422


@pytest.mark.parametrize('accuracy', [-1, 'NaN', 'Infinity'])
def test_invalid_gps_accuracy_rejected(routing, accuracy):
    app = FastAPI()
    app.include_router(routing.router)
    with TestClient(app) as client:
        assert client.post('/compute', json=dict(origin_lat=10, origin_lon=122, dest_lat=10,
            dest_lon=122, origin_accuracy_m=accuracy)).status_code == 422


def test_google_request_caps_persist_and_reset_by_date(routing, monkeypatch, tmp_path):
    from api.routing_usage import reserve_google_request
    from datetime import datetime, timezone
    import sqlite3
    monkeypatch.setenv('GOOGLE_ROUTES_DAILY_LIMIT', '2')
    monkeypatch.setenv('GOOGLE_ROUTES_MONTHLY_LIMIT', '3')
    path = tmp_path / 'usage.sqlite3'
    def reserve(day):
        reserve_google_request('test-key', usage_path=path,
            now=datetime.fromisoformat(day).replace(tzinfo=timezone.utc))
    reserve('2026-10-08')
    reserve('2026-10-08')
    with pytest.raises(HTTPException) as daily:
        reserve('2026-10-08')
    assert daily.value.status_code == 429
    assert 'daily' in daily.value.detail
    reserve('2026-10-09')
    with pytest.raises(HTTPException) as monthly:
        reserve('2026-10-09')
    assert 'monthly' in monthly.value.detail
    with sqlite3.connect(path) as connection:
        assert dict(connection.execute('SELECT period, count FROM usage'))['2026-10'] == 3
    assert b'test-key' not in path.read_bytes()
    reserve('2026-11-01')


def test_simultaneous_requests_cannot_exceed_the_limit(routing, monkeypatch, tmp_path):
    from api.routing_usage import reserve_google_request
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setenv('GOOGLE_ROUTES_DAILY_LIMIT', '3')
    monkeypatch.setenv('GOOGLE_ROUTES_MONTHLY_LIMIT', '3')
    path = tmp_path / 'usage.sqlite3'
    def attempt(_):
        try:
            reserve_google_request('test-key', usage_path=path)
            return 200
        except HTTPException as error:
            return error.status_code
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(attempt, range(8)))
    assert results.count(200) == 3
    assert results.count(429) == 5


def test_usage_storage_failure_blocks_the_provider_request(routing, tmp_path):
    from api.routing_usage import reserve_google_request
    bad_directory = tmp_path / 'not-a-directory'
    bad_directory.write_text('occupied', encoding='utf-8')
    with pytest.raises(HTTPException) as unavailable:
        reserve_google_request('test-key', usage_path=bad_directory / 'usage.sqlite3')
    assert unavailable.value.status_code == 503


def test_zero_limit_stops_google_before_sending_a_request(routing, monkeypatch, tmp_path):
    from api.routing_usage import reserve_google_request
    monkeypatch.setenv('GOOGLE_ROUTES_DAILY_LIMIT', '0')
    monkeypatch.setattr(routing, 'reserve_google_request',
        lambda key: reserve_google_request(key, usage_path=tmp_path / 'usage.sqlite3'))
    monkeypatch.setattr(routing, 'urlopen', lambda *args, **kwargs: pytest.fail('Google was contacted after the cap'))
    with pytest.raises(HTTPException) as capped:
        routing._google_route([10.7,122.55], [10.78,122.62], 'walking')
    assert capped.value.status_code == 429


def test_laptop_key_is_read_from_private_env_and_hosted_key_takes_priority(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    routing = importlib.import_module('api.routes.routing')
    monkeypatch.delenv('GOOGLE_ROUTES_API_KEY', raising=False)
    environment = tmp_path / '.env'
    environment.write_text('GOOGLE_ROUTES_API_KEY=local-test-key\n', encoding='utf-8')
    monkeypatch.setattr(routing, '_ENV_PATH', environment)
    assert routing._load_google_routes_api_key() == 'local-test-key'
    environment.write_text('GOOGLE_ROUTES_API_KEY=updated-test-key\n', encoding='utf-8')
    assert routing._load_google_routes_api_key() == 'updated-test-key'
    monkeypatch.setenv('GOOGLE_ROUTES_API_KEY', 'hosted-test-key')
    assert routing._load_google_routes_api_key() == 'hosted-test-key'
