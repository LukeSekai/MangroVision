"""Point-owned site checks use temporary PostgreSQL tables and outer rollback."""
import os

import pytest
from shapely.geometry import Point, box, shape
from sqlalchemy import text

import planting_database as db
from mangrovision_db.site_point_selection import create_site_from_points
from mangrovision_db.monitoring_locations import release_dead_locations_batch
from test_site_release_postgres import workspace as release_workspace

pytestmark = pytest.mark.skipif(os.getenv('MANGROVISION_RUN_TEMP_ACCOUNT_TESTS') != '1', reason='Rollback-only PostgreSQL checks are opt-in.')


@pytest.fixture
def workspace(release_workspace):
    conn = release_workspace
    conn.exec_driver_sql('''ALTER TABLE pg_temp.project_sites ALTER COLUMN id RESTART WITH 1000;
        INSERT INTO organizations(id,name,normalized_name,inspection_interval_days) VALUES (2,'New organization','new organization',14);
        INSERT INTO planting_points(id,analysis_id,point_num,latitude,longitude,status)
            SELECT n,1,n,10.780,122.627+(n-10)*.000009,'planned' FROM generate_series(10,14) n;
    ''')
    return conn


def create(ids, **kwargs):
    return create_site_from_points(organization_id=2, planting_point_ids=ids,
        coverage=box(122.624,10.778,122.630,10.782), user_id=1, **kwargs)


def test_exact_points_become_owned_without_claiming_an_unselected_middle_point(workspace):
    before_events = workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all()
    before_analysis = workspace.execute(text('SELECT * FROM analyses ORDER BY id')).all()
    site = create([10,12], name='Chosen locations')
    boundary = shape(site['geometry'])
    assert site['selected_point_count'] == 2
    assert all(boundary.covers(Point(122.627+(n-10)*.000009,10.780)) == (n in (10,12)) for n in range(10,15))
    points = {row['id']: row for row in db.list_planter_assignment_map_points()}
    assert points[10]['source_organization_id'] == points[12]['source_organization_id'] == 2
    assert points[11]['source_organization_id'] is None
    assert points[10]['planting_status'] == 'planned' and points[10]['assigned_planter_id'] is None
    assert workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all() == before_events
    assert workspace.execute(text('SELECT * FROM analyses ORDER BY id')).all() == before_analysis
    assert workspace.scalar(text("SELECT count(*) FROM activity_logs WHERE action='project_site.created_from_points'")) == 1


def test_existing_organization_can_claim_another_separate_site(workspace):
    first = create([10])
    second = create([14])
    assert first['id'] != second['id'] and first['organization_id'] == second['organization_id'] == 2
    assert workspace.scalar(text('SELECT count(*) FROM project_sites WHERE organization_id=2')) == 2


@pytest.mark.parametrize('change', ['claimed','planted','assigned','eroded','deleted','outside'])
def test_changed_or_unavailable_selection_is_rejected_atomically(workspace, change):
    coverage = box(122.624,10.778,122.630,10.782)
    if change == 'claimed':
        create([10])
    elif change == 'planted':
        workspace.exec_driver_sql("UPDATE planting_points SET status='planted',planted_at=CURRENT_TIMESTAMP WHERE id=10")
    elif change == 'assigned':
        workspace.exec_driver_sql("INSERT INTO planter_assignment_points(assignment_id,planting_point_id,sequence_num,status) VALUES(1,10,10,'pending')")
    elif change == 'eroded':
        workspace.exec_driver_sql("""INSERT INTO map_zones(zone_type,name,polygon_geojson) VALUES('eroded','Test erosion',
            '{"type":"Polygon","coordinates":[[[122.6269,10.7799],[122.6271,10.7799],[122.6271,10.7801],[122.6269,10.7801],[122.6269,10.7799]]]}'::jsonb)""")
    elif change == 'deleted':
        workspace.exec_driver_sql('UPDATE planting_points SET deleted_at=CURRENT_TIMESTAMP WHERE id=10')
    elif change == 'outside':
        coverage = box(122.624,10.778,122.626,10.782)
    before = workspace.scalar(text('SELECT count(*) FROM project_sites'))
    with pytest.raises(ValueError):
        create_site_from_points(organization_id=2,planting_point_ids=[10,14],coverage=coverage,user_id=1)
    assert workspace.scalar(text('SELECT count(*) FROM project_sites')) == before
    assert workspace.scalar(text('SELECT count(*) FROM project_sites ps JOIN planting_points pp ON extensions.ST_Covers(ps.geometry,pp.location) WHERE pp.id=14')) == 0


def test_released_dead_location_gains_a_new_owner_without_moving_history(workspace):
    release_dead_locations_batch([1], {'1': 0}, 1)
    original_events = workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all()
    original_deaths = workspace.execute(text('SELECT * FROM point_death_records ORDER BY id')).all()
    site = create([1])
    assert shape(site['geometry']).covers(Point(122.6251,10.780))
    assert workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all() == original_events
    assert workspace.execute(text('SELECT * FROM point_death_records ORDER BY id')).all() == original_deaths
    points = {row['id']: row for row in db.list_planter_assignment_map_points()}
    assert points[1]['source_organization_id'] == 2 and points[1]['planting_status'] == 'planned'
    assert points[2]['source_organization_id'] == 1


def test_manual_site_creation_cannot_take_land_claimed_by_point_selection(workspace):
    site = create([10])
    with pytest.raises(ValueError, match='another organization'):
        db.create_project_site('Stale manual boundary', site['geometry'], organization_id=1)


def test_drawn_site_is_saved_inside_the_map_and_counts_only_kept_points(workspace,monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MangroVision_New'))
    from api.routes import project_sites as routes
    from shapely.geometry import mapping
    coverage = box(122.6265,10.7799,122.62701,10.7801)
    drawing = box(122.6266,10.7798,122.6271,10.7802)
    monkeypatch.setattr(routes,'get_user_by_session_token',lambda _: {'id':1,'role':'lgu'})
    monkeypatch.setattr(routes,'is_processing_active',lambda: False)
    monkeypatch.setattr(routes,'_load_zone_coverage',lambda: coverage)
    original_events = workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all()
    preview = routes.validate_project_site_boundary(routes.ProjectSiteBoundary(organization_id=2,geometry=mapping(drawing)))
    assert preview['clipped'] and preview['point_count'] == 2
    site = db.create_project_site(organization_id=2,name='Map-edge site',geometry=preview['geometry'])
    stored = shape(workspace.scalar(text('SELECT polygon_geojson FROM project_sites WHERE id=:id'),{'id':site['id']}))
    assert coverage.covers(stored) and stored.equals(shape(preview['geometry']))
    assert site['properties']['point_count'] == 2
    assert workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all() == original_events


def test_cleaned_site_saves_the_preview_without_claiming_a_nearby_unselected_point(workspace,monkeypatch):
    from test_project_site_selection_api import noisy_site
    from api.routes import project_sites as routes
    from shapely.geometry import mapping
    original,xy=noisy_site()
    for point_id,offset in zip(range(10,15),range(2,12,2)):
        longitude,latitude=xy(offset,offset)
        workspace.execute(text('UPDATE planting_points SET longitude=:lon,latitude=:lat WHERE id=:id'),
                          {'id':point_id,'lon':longitude,'lat':latitude})
    longitude,latitude=xy(19.95,6/4)
    workspace.execute(text("INSERT INTO planting_points(id,analysis_id,point_num,longitude,latitude,status) VALUES(15,1,15,:lon,:lat,'planned')"),
                      {'lon':longitude,'lat':latitude})
    coverage=box(*xy(-2,-2),*xy(22,22))
    monkeypatch.setattr(routes,'get_user_by_session_token',lambda _: {'id':1,'role':'lgu'})
    monkeypatch.setattr(routes,'is_processing_active',lambda: False)
    monkeypatch.setattr(routes,'_load_zone_coverage',lambda: coverage)
    before=workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all()
    preview=routes.validate_project_site_boundary(routes.ProjectSiteBoundary(organization_id=2,geometry=mapping(original)))
    assert preview['cleaned'] and preview['point_count']==5
    assert not shape(preview['geometry']).covers(Point(longitude,latitude))
    site=db.create_project_site(organization_id=2,name='Cleaned site',geometry=preview['geometry'])
    stored=shape(workspace.scalar(text('SELECT polygon_geojson FROM project_sites WHERE id=:id'),{'id':site['id']}))
    assert stored.equals(shape(preview['geometry'])) and site['properties']['point_count']==5
    assert not stored.covers(Point(longitude,latitude))
    assert workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all()==before


def test_lgu_can_claim_a_dense_released_patch_without_projection_overlap(workspace):
    import json
    from pyproj import Transformer
    from shapely.affinity import rotate
    from shapely.ops import transform
    from shapely.geometry import mapping
    forward=Transformer.from_crs(4326,32651,always_xy=True).transform
    backward=Transformer.from_crs(32651,4326,always_xy=True).transform
    x,y=forward(122.6275,10.780)
    hole=transform(backward,rotate(box(x-.65,y-.65,x+3.65,y+3.65),7.3,origin=(x+1.5,y+1.5)))
    retained=box(122.625,10.779,122.628,10.781).difference(hole)
    workspace.execute(text('UPDATE project_sites SET polygon_geojson=CAST(:geometry AS jsonb) WHERE id=1'),
                      {'geometry':json.dumps(mapping(retained))})
    workspace.exec_driver_sql("UPDATE organizations SET name='LGU',normalized_name='lgu',system_key='lgu' WHERE id=2")
    ids=[]
    for index in range(16):
        point=rotate(Point(x+index%4,y+index//4),7.3,origin=(x+1.5,y+1.5))
        longitude,latitude=backward(point.x,point.y)
        point_id=100+index; ids.append(point_id)
        workspace.execute(text("INSERT INTO planting_points(id,analysis_id,point_num,longitude,latitude,status) VALUES(:id,1,:id,:lon,:lat,'planned')"),
                          {'id':point_id,'lon':longitude,'lat':latitude})
    history=workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all()
    deaths=workspace.execute(text('SELECT * FROM point_death_records ORDER BY id')).all()
    site=create(ids,name='LGU released patch')
    overlap=workspace.scalar(text('''SELECT extensions.ST_Relate(a.geometry,b.geometry,'T********')
        FROM project_sites a CROSS JOIN project_sites b WHERE a.id=1 AND b.id=:id'''),{'id':site['id']})
    assert overlap is False and site['selected_point_count']==16
    points={row['id']:row for row in db.list_planter_assignment_map_points()}
    assert all(points[point_id]['source_organization_id']==2 for point_id in ids)
    assert workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all()==history
    assert workspace.execute(text('SELECT * FROM point_death_records ORDER BY id')).all()==deaths
