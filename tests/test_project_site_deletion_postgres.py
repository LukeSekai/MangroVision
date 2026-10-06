"""Deletion regressions run in temporary PostgreSQL tables, never live rows."""
import importlib
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

import planting_database as db
from mangrovision_db.compat import CompatConnection, get_engine

pytestmark = pytest.mark.skipif(
    os.getenv("MANGROVISION_RUN_TEMP_ACCOUNT_TESTS") != "1",
    reason="Temporary PostgreSQL checks are opt-in.",
)


@pytest.fixture()
def workspace(monkeypatch):
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(text("SET LOCAL statement_timeout = '20s'"))
            for table in (
                "project_sites", "analyses", "planting_points",
                "planter_assignments", "planter_assignment_points", "planting_events", "planting_schedules",
            ):
                connection.execute(text(
                    f"CREATE TEMP TABLE {table} (LIKE mangrovision.{table} INCLUDING ALL)"
                ))
            connection.execute(text(
                "SET LOCAL search_path = pg_temp, mangrovision, extensions, public"
            ))
            connection.execute(text(
                "CREATE TEMP VIEW site_zones AS SELECT * FROM pg_temp.project_sites"
            ))
            # LIKE does not copy foreign keys. Recreate the production actions
            # against temporary parents so the preservation check exercises SQL.
            for table, action in (
                ("analyses", "SET NULL"), ("planter_assignments", "SET NULL"),
                ("planting_events", "SET NULL"), ("planting_schedules", "RESTRICT"),
            ):
                connection.execute(text(
                    f"ALTER TABLE pg_temp.{table} ADD FOREIGN KEY (project_site_id) "
                    f"REFERENCES pg_temp.project_sites(id) ON DELETE {action}"
                ))
            connection.execute(text(
                "ALTER TABLE pg_temp.planting_points ADD FOREIGN KEY (analysis_id) "
                "REFERENCES pg_temp.analyses(id) ON DELETE CASCADE"
            ))

            class TempConnection(CompatConnection):
                def __init__(self):
                    super().__init__(connection)
                    self.savepoint = connection.begin_nested()

                def commit(self):
                    self.savepoint.commit()

                def rollback(self):
                    if self.savepoint.is_active:
                        self.savepoint.rollback()

                def close(self):
                    self.rollback()

            monkeypatch.setattr(db, "_get_connection", TempConnection)
            connection.execute(text("""
                INSERT INTO project_sites(id, name, polygon_geojson) VALUES
                  (1, 'OTON test', '{"type":"Polygon","coordinates":[[[122,10],[123,10],[123,11],[122,11],[122,10]]]}'::jsonb);
                INSERT INTO analyses(id, analysis_number, image_name, analyzed_at, project_site_id)
                  VALUES (1, 1, 'saved-oton-test.jpg', CURRENT_TIMESTAMP, 1);
                INSERT INTO planting_points(id, analysis_id, point_num, latitude, longitude)
                  SELECT value, 1, value, 10.5, 122.5 + value * .00001
                  FROM generate_series(1, 27) value;
            """))
            yield connection
        finally:
            transaction.rollback()


@pytest.fixture()
def client(workspace, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "MangroVision_New"))
    routes = importlib.import_module("api.routes.project_sites")
    monkeypatch.setattr(routes, "get_user_by_session_token", lambda _: {"id": 1, "role": "lgu"})
    monkeypatch.setattr(routes, "is_processing_active", lambda: False)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/project-sites")
    with TestClient(app) as test_client:
        yield test_client, routes


def test_delete_analysis_only_site_keeps_all_mapped_points(workspace, client):
    response = client[0].delete("/api/project-sites/1")
    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "id": 1}
    assert workspace.scalar(text("SELECT COUNT(*) FROM project_sites")) == 0
    analysis = workspace.execute(text("SELECT * FROM analyses")).mappings().one()
    assert analysis["project_site_id"] is None
    assert analysis["image_name"] == "saved-oton-test.jpg"
    assert workspace.scalar(text("SELECT COUNT(*) FROM planting_points WHERE analysis_id = 1")) == 27
    assert workspace.scalar(text("SELECT COUNT(*) FROM planting_points WHERE status = 'planned'")) == 27


@pytest.mark.parametrize("sql", [
    """INSERT INTO planting_events(id, source_key, planted_at, project_site_id, closed_at, species)
         VALUES (1, 'existing-event', CURRENT_TIMESTAMP, 1, CURRENT_TIMESTAMP, 'Rhizophora')""",
    "UPDATE planting_points SET status = 'planted' WHERE id = 1",
    "UPDATE planting_points SET planted_at = CURRENT_TIMESTAMP WHERE id = 1",
    """INSERT INTO planting_events(id, source_key, planted_at, planting_point_id, species)
         VALUES (1, 'unlinked-event', CURRENT_TIMESTAMP, 1, 'Rhizophora')""",
    """INSERT INTO planter_assignments(id, planter_id, title, assignment_date, project_site_id, species)
         VALUES (1, 1, 'Existing assignment', CURRENT_DATE, 1, 'Rhizophora');
       INSERT INTO planter_assignment_points(id, assignment_id, planting_point_id, sequence_num, status)
         VALUES (1, 1, 1, 1, 'completed')""",
])
def test_recorded_planting_preserves_site_and_analysis_links(workspace, client, sql):
    workspace.execute(text(sql))
    response = client[0].delete("/api/project-sites/1")
    assert response.status_code == 409
    assert "recorded planting history" in response.json()["detail"]
    assert workspace.scalar(text("SELECT COUNT(*) FROM project_sites WHERE id = 1")) == 1
    assert workspace.scalar(text("SELECT project_site_id FROM analyses WHERE id = 1")) == 1
    assert workspace.scalar(text("SELECT COUNT(*) FROM planting_points")) == 27


def test_assignments_and_schedules_allow_deletion_before_planting(workspace, client):
    workspace.execute(text("""
        INSERT INTO planter_assignments(id, planter_id, title, assignment_date, project_site_id, species)
          VALUES (1, 1, 'Unplanted assignment', CURRENT_DATE, 1, 'Rhizophora');
        INSERT INTO planter_assignment_points(id, assignment_id, planting_point_id, sequence_num)
          VALUES (1, 1, 1, 1);
        INSERT INTO planting_schedules(id, organization, title, start_at, end_at, project_site_id)
          VALUES (1, 'Test organization', 'Planned activity', CURRENT_TIMESTAMP,
                  CURRENT_TIMESTAMP + INTERVAL '1 hour', 1);
    """))
    assert client[0].delete("/api/project-sites/1").status_code == 200
    assignment = workspace.execute(text("SELECT * FROM planter_assignments")).mappings().one()
    assert assignment["project_site_id"] is None
    assert assignment["title"] == "Unplanted assignment"
    assert workspace.scalar(text("SELECT status FROM planter_assignment_points")) == "pending"
    schedule = workspace.execute(text("SELECT * FROM planting_schedules")).mappings().one()
    assert schedule["project_site_id"] is None
    assert schedule["title"] == "Planned activity"
    assert workspace.scalar(text("SELECT COUNT(*) FROM planting_points")) == 27


def test_planting_outside_boundary_does_not_block_linked_image(workspace, client):
    workspace.execute(text("UPDATE planting_points SET latitude = 12, status = 'planted' WHERE id = 1"))
    assert client[0].delete("/api/project-sites/1").status_code == 200
    assert workspace.scalar(text("SELECT status FROM planting_points WHERE id = 1")) == "planted"


def test_unknown_site_returns_404(workspace, client):
    assert client[0].delete("/api/project-sites/999").status_code == 404
    assert workspace.scalar(text("SELECT COUNT(*) FROM project_sites")) == 1


def test_other_site_history_does_not_block_unused_site(workspace):
    workspace.execute(text("""
        INSERT INTO project_sites(id, name, polygon_geojson)
          SELECT 2, 'Other test site', polygon_geojson FROM project_sites WHERE id = 1;
        INSERT INTO planting_events(id, source_key, planted_at, project_site_id, species)
          VALUES (1, 'other-site-event', CURRENT_TIMESTAMP, 2, 'Rhizophora');
    """))
    assert db.delete_project_site(1)
    assert workspace.scalar(text("SELECT project_site_id FROM planting_events WHERE id = 1")) == 2


@pytest.mark.parametrize(("role", "processing", "expected_status"), [
    (None, False, 401), ("planter", False, 403), ("lgu", True, 409),
])
def test_delete_keeps_auth_and_processing_guards(workspace, client, monkeypatch, role, processing, expected_status):
    test_client, routes = client
    monkeypatch.setattr(routes, "get_user_by_session_token", lambda _: {"id": 1, "role": role} if role else None)
    monkeypatch.setattr(routes, "is_processing_active", lambda: processing)
    assert test_client.delete("/api/project-sites/1").status_code == expected_status
    assert workspace.scalar(text("SELECT COUNT(*) FROM project_sites")) == 1
