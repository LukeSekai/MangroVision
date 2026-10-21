"""Saved-point attribution and availability survive every download format.

Uses an isolated FastAPI app and synthetic points; never connects to the DB.
"""
import csv
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from PIL import Image

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.routes import dashboard, export
from api.activity_audit import describe_activity
from waypoint_export import generate_geojson


STYLE_CASES = [
    ('planned', 'bungalon', False, '#16a34a', 'bungalon', 'Flag, Green'),
    ('assigned', 'rhizophora', False, '#db2777', 'rhizophora', 'Flag, Pink'),
    ('planted', 'bungalon', False, '#eab308', 'planted', 'Flag, Yellow'),
    ('planted', 'rhizophora', False, '#eab308', 'planted', 'Flag, Yellow'),
    ('planted', 'bungalon', True, '#f97316', 'eroded', 'Flag, Orange'),
    ('dead', 'rhizophora', False, '#7f1d1d', 'dead', 'Flag, Red'),
    ('dead', 'bungalon', True, '#7f1d1d', 'dead', 'Flag, Red'),
    ('unavailable', 'rhizophora', False, '#f97316', 'eroded', 'Flag, Orange'),
    ('eroded_unavailable', None, False, '#f97316', 'eroded', 'Flag, Orange'),
    ('completed', 'bungalon', False, '#eab308', 'planted', 'Flag, Yellow'),
    ('planned', 'Legacy species', False, '#64748b', 'unknown', 'Flag'),
    ('planned', None, False, '#64748b', 'unknown', 'Flag'),
]


def styled_payload():
    return {'waypoints': [
        {'lat': 10.78 + index / 100000, 'lon': 122.62, 'point_id': index + 1,
         'species': species, 'status': status, 'eroded_unavailable': eroded}
        for index, (status, species, eroded, *_style) in enumerate(STYLE_CASES)
    ]}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(export.router, prefix='/api/export')
    app.dependency_overrides[export._require_lgu_user] = lambda: {'id': 1, 'role': 'lgu'}
    return TestClient(app)


def payload():
    return {'image_name': 'North site | Analysis 2 | Bungalon', 'waypoints': [
        {'lat': 10.7801234, 'lon': 122.6201234, 'point_num': 7, 'name': 'Analysis 2-007',
         'point_id': 101, 'analysis_id': 52, 'image_name': 'Analysis 2',
         'project_site_id': 4, 'project_site_name': 'North & <site>', 'species': 'bungalon',
         'status': 'unavailable', 'eroded_unavailable': True, 'availability_reason': 'Inside an eroded zone'},
        {'lat': 10.781, 'lon': 122.621, 'point_num': 8, 'point_id': 102,
         'status': 'dead', 'species': 'rhizophora'},
    ]}


def test_csv_preserves_current_attribution_and_neutralizes_formula_names(client):
    body = payload()
    body['waypoints'][1]['project_site_name'] = '=HYPERLINK("unsafe")'
    response = client.post('/api/export/csv', json=body)
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 2
    assert rows[0]['Point #'] == '7'
    assert rows[0]['Latitude'] == '10.7801234'
    assert rows[0]['Analysis ID'] == '52'
    assert rows[0]['Project Site'] == 'North & <site>'
    assert rows[0]['Species'] == 'bungalon'
    assert rows[0]['Eroded / Unavailable'] == 'True'
    assert rows[0]['Availability Reason'] == 'Inside an eroded zone'
    assert rows[1]['Status'] == 'dead'
    assert rows[1]['Project Site'].startswith("'=")


@pytest.mark.parametrize('format', ['gpx', 'kml'])
def test_xml_exports_preserve_point_count_coordinates_and_escaped_context(client, format):
    response = client.post(f'/api/export/{format}', json=payload())
    assert response.status_code == 200
    root = ET.fromstring(response.text)
    if format == 'gpx':
        ns = {'g': 'http://www.topografix.com/GPX/1/1', 'mv': 'http://mangrovision.local/gpx/ext/1'}
        rows = root.findall('g:wpt', ns)
        assert len(rows) == 2
        assert float(rows[0].attrib['lat']) == 10.7801234
        assert rows[0].find('.//mv:project_site_name', ns).text == 'North & <site>'
        assert rows[0].find('.//mv:eroded_unavailable', ns).text == 'true'
        assert rows[1].find('.//mv:species', ns).text == 'rhizophora'
    else:
        ns = {'k': 'http://www.opengis.net/kml/2.2'}
        rows = root.findall('.//k:Placemark', ns)
        assert len(rows) == 2
        assert rows[0].find('.//k:coordinates', ns).text == '122.62012340,10.78012340,0'
        assert rows[0].find('.//k:Data[@name="project_site_name"]/k:value', ns).text == 'North & <site>'
        assert rows[1].find('.//k:Data[@name="species"]/k:value', ns).text == 'rhizophora'
        assert root.find('.//k:LabelStyle/k:scale', ns).text == '0'


def test_geojson_preserves_current_species_and_unavailable_points(client):
    response = client.post('/api/export/geojson', json=payload())
    assert response.status_code == 200
    data = response.json()
    assert data['metadata']['total_points'] == 2
    first, second = data['features']
    assert first['geometry']['coordinates'] == [122.6201234, 10.7801234]
    assert first['properties']['analysis_id'] == 52
    assert first['properties']['project_site_id'] == 4
    assert first['properties']['eroded_unavailable'] is True
    assert second['properties']['species'] == 'rhizophora'
    assert second['properties']['status'] == 'dead'
    assert first['properties']['marker-color'] == '#f97316'
    assert first['properties']['mangrovision_style']['color_category'] == 'eroded'
    assert second['properties']['marker-color'] == '#7f1d1d'
    assert second['properties']['marker-symbol'] == ''


@pytest.mark.parametrize('format', ['csv', 'gpx', 'kml', 'kmz', 'geojson'])
def test_older_analysis_export_payloads_remain_supported(client, format):
    response = client.post(f'/api/export/{format}', json={'waypoints': [{'lat': 10.78, 'lon': 122.62}]})
    assert response.status_code == 200
    assert 'attachment;' in response.headers['content-disposition']


def test_kmz_packages_icons_and_retains_one_point_per_seedling_without_external_assets(client):
    body = styled_payload()
    response = client.post('/api/export/kmz', json=body)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/vnd.google-earth.kmz'
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['content-disposition'].endswith('planting_points.kmz"')
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    with ZipFile(io.BytesIO(response.content)) as archive:
        assert all(name == 'doc.kml' or name.startswith('icons/point-') and name.endswith('.png') for name in archive.namelist())
        root = ET.fromstring(archive.read('doc.kml'))
        styles = {style.attrib['id']: style for style in root.findall('.//k:Style', ns)}
        assert len(styles) == 6, 'one reusable solid dot per color'
        rows = root.findall('.//k:Placemark', ns)
        assert len(rows) == len(body['waypoints'])
        assert len(root.findall('.//k:Point', ns)) == len(body['waypoints'])
        assert not root.findall('.//k:LineString', ns), 'styling must not introduce extra location geometry'
        for row, expected, (_status, _species, _eroded, color, category, _symbol) in zip(rows, body['waypoints'], STYLE_CASES):
            assert row.find('k:Point/k:coordinates', ns).text == f"{expected['lon']:.8f},{expected['lat']:.8f},0"
            style = styles[row.find('k:styleUrl', ns).text[1:]]
            assert style.find('k:LabelStyle/k:scale', ns).text == '0'
            assert style.find('k:IconStyle/k:color', ns).text == 'ffffffff'
            assert style.find('k:IconStyle/k:scale', ns).text == '0.35'
            assert row.find('k:ExtendedData/k:Data[@name="color_category"]/k:value', ns).text == category
            assert row.find('k:ExtendedData/k:Data[@name="eroded_unavailable"]/k:value', ns).text == str(expected['eroded_unavailable']).lower()
            if expected['species']:
                assert row.find('k:ExtendedData/k:Data[@name="species"]/k:value', ns).text == expected['species']
            path = style.find('k:IconStyle/k:Icon/k:href', ns).text
            assert path in archive.namelist()
            assert '://' not in path
            with Image.open(io.BytesIO(archive.read(path))) as icon:
                assert icon.mode == 'RGBA'
                assert icon.getpixel((0, 0))[3] == 0
                fill = tuple(int(color[index:index+2], 16) for index in (1, 3, 5))
                for position in ((32, 32), (44, 32), (20, 20)):
                    assert all(abs(actual - target) < 5 for actual, target in zip(icon.getpixel(position)[:3], fill)), 'the full dot must use one color without a cross'
                assert icon.getpixel((32, 6))[3] < 5, 'solid dots must not add an erosion ring'


def test_kmz_requires_staff_permission(monkeypatch):
    app = FastAPI()
    app.include_router(export.router, prefix='/api/export')
    monkeypatch.setattr(dashboard, 'get_user_by_session_token', lambda token: None)
    with TestClient(app) as anonymous:
        assert anonymous.post('/api/export/kmz', json=payload()).status_code == 401
    monkeypatch.setattr(dashboard, 'get_user_by_session_token', lambda token: {'id': 1, 'role': 'planter'})
    with TestClient(app) as planter:
        assert planter.post('/api/export/kmz', json=payload()).status_code == 403


def test_kmz_download_is_classified_for_the_existing_activity_log():
    assert describe_activity('POST', '/api/export/kmz').action == 'export.kmz'


def test_all_formats_share_status_color_priority_without_losing_species_and_erosion(client):
    body = styled_payload()
    csv_rows = list(csv.DictReader(io.StringIO(client.post('/api/export/csv', json=body).text)))
    gpx = ET.fromstring(client.post('/api/export/gpx', json=body).text)
    ns = {'g': 'http://www.topografix.com/GPX/1/1', 'mv': 'http://mangrovision.local/gpx/ext/1'}
    gpx_rows = gpx.findall('g:wpt', ns)
    features = client.post('/api/export/geojson', json=body).json()['features']
    assert len(csv_rows) == len(gpx_rows) == len(features) == len(STYLE_CASES)
    for row, gpx_row, feature, (status, species, eroded, color, category, symbol) in zip(csv_rows, gpx_rows, features, STYLE_CASES):
        assert row['Marker Color'] == color
        assert row['Color Category'] == category
        assert row['Status'] == status
        assert row['Species'] == (species or '')
        assert row['Eroded / Unavailable'] == str(eroded)
        assert gpx_row.find('g:sym', ns).text == symbol
        assert gpx_row.find('.//mv:marker_color', ns).text == color
        assert gpx_row.find('.//mv:color_category', ns).text == category
        assert gpx_row.find('.//mv:eroded_unavailable', ns).text == str(eroded).lower()
        if species:
            assert gpx_row.find('.//mv:species', ns).text == species
        if eroded:
            assert 'Inside an eroded zone' in gpx_row.find('g:desc', ns).text
        assert feature['properties']['mangrovision_style'] == {'marker_color': color, 'color_category': category}
        assert feature['properties']['marker-color'] == color
        assert feature['properties']['color_category'] == category
        assert feature['properties']['marker-symbol'] == ''
        assert feature['properties']['species'] == species
        assert feature['properties']['status'] == status
        assert feature['properties']['eroded_unavailable'] == eroded


def test_processing_geojson_keeps_its_existing_contract():
    data = json.loads(generate_geojson(payload()['waypoints']))
    assert all('mangrovision_style' not in point['properties'] and 'marker-color' not in point['properties'] for point in data['features'])


def test_raw_kml_escapes_field_text_and_uses_status_colors(client):
    body = payload()
    body['waypoints'][0]['project_site_name'] = '<script>unsafe</script>'
    response = client.post('/api/export/kml', json=body)
    root = ET.fromstring(response.text)
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    description = root.find('.//k:Placemark/k:description', ns).text
    assert '<script>' not in description
    assert '&lt;script&gt;' in description
    colors = {style.find('k:IconStyle/k:color', ns).text for style in root.findall('.//k:Style', ns)}
    assert colors == {'ff1673f9', 'ff1d1d7f'}


@pytest.mark.parametrize('format', ['csv', 'gpx', 'kml', 'kmz', 'geojson'])
def test_download_names_are_sanitized_for_headers(client, format):
    body = payload()
    body.update(filename_stem='../../North site"\r\nInjected: header', export_name='North site')
    response = client.post(f'/api/export/{format}', json=body)
    assert response.status_code == 200
    header = response.headers['content-disposition']
    assert header.startswith('attachment; filename="North-site-Injected-header.')
    assert '\r' not in header and '\n' not in header and '..' not in header and '/' not in header


def test_gpx_keeps_short_navigation_ids_and_full_human_context(client):
    body = payload()
    body['export_name'] = 'MangroVision — North site — Planting points — 2026-10-22'
    body['waypoints'][0].update(name='Analysis 2 — Point 007 (ID 101)', navigation_name='MV-000101')
    root = ET.fromstring(client.post('/api/export/gpx', json=body).text)
    ns = {'g': 'http://www.topografix.com/GPX/1/1'}
    assert root.find('g:metadata/g:name', ns).text == body['export_name']
    point = root.find('g:wpt', ns)
    assert point.find('g:name', ns).text == 'MV-000101'
    assert point.find('g:cmt', ns).text == 'Analysis 2 — Point 007 (ID 101)'
    assert 'Project site: North & <site>' in point.find('g:desc', ns).text
    assert 'Species: bungalon' in point.find('g:desc', ns).text
