"""Exercise shared-account device ownership and resumed progress in TEMP tables."""
import importlib
import os
from collections import Counter
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
                "organizations", "planters", "organization_participants",
                "planter_assignments", "planter_assignment_points", "planting_events",
                "point_death_records", "auth_sessions", "project_sites", "analyses",
                "planting_points", "users", "map_zones", "replanting_requests",
                "activity_logs", "planting_schedules", "monitoring_observations",
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
            # Production's completion trigger has a fixed search_path. Clone it
            # into pg_temp, directing every reference to isolated fixture tables.
            trigger_sql = connection.scalar(text(
                "SELECT pg_get_functiondef('mangrovision.maintain_planting_event()'::regprocedure)"
            ))
            connection.execute(text(trigger_sql.replace("mangrovision", "pg_temp")))
            connection.execute(text("""
                CREATE TRIGGER test_planting_event AFTER UPDATE OF status
                ON pg_temp.planter_assignment_points FOR EACH ROW
                EXECUTE FUNCTION pg_temp.maintain_planting_event()
            """))

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
            monkeypatch.setattr(db, "_load_warning_zone_geometries", lambda: [])
            monkeypatch.setattr(db, "_is_point_inside_eroded_zone", lambda *_: False)
            connection.execute(text("""
                INSERT INTO organizations(id, name, normalized_name, inspection_interval_days)
                  VALUES (1, 'Shared device test', 'shared device test', 14);
                INSERT INTO project_sites(id, name, organization_id, polygon_geojson)
                  VALUES (1, 'Device test site', 1,
                    '{"type":"Polygon","coordinates":[[[122,10],[123,10],[123,11],[122,11],[122,10]]]}'::jsonb);
                INSERT INTO analyses(id, analysis_number, image_name, analyzed_at, project_site_id, species)
                  VALUES (1, 1, 'device-test.jpg', CURRENT_TIMESTAMP, 1, 'Rhizophora');
                INSERT INTO planting_points(id, analysis_id, point_num, latitude, longitude)
                  SELECT value, 1, value, 10.5, 122.5 + value * .00001
                  FROM generate_series(1, 100) value;
            """))
            # Reserve before registration, as in Quick Assign's pending account.
            db.create_organization_assignment(1, list(range(1, 101)), site_zone_id=1, species="Rhizophora")
            monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "MangroVision_New"))
            auth = importlib.import_module("api.routes.planter_auth")
            security = importlib.import_module("api.security")
            app = FastAPI()
            app.add_middleware(security.SessionSecurityMiddleware)
            app.include_router(auth.router, prefix="/api/planter-auth")
            yield connection, app
        finally:
            transaction.rollback()


def post(client, path, payload=None):
    headers = {"X-CSRF-Token": client.cookies.get("mv_csrf", "")}
    return client.post(f"/api/planter-auth{path}", json=payload, headers=headers)


def login(client, device):
    return post(client, "/login", {
        "username": "device_progress_test", "password": "test-password",
        "device_key": f"persistent-test-device-{device:03d}",
    })


def test_registration_ten_devices_logout_and_login_restore_points_and_progress(workspace):
    connection, app = workspace
    origin = "https://device-progress.example.test"
    clients = [TestClient(app, base_url=origin, headers={"Origin": origin}) for _ in range(11)]
    try:
        registered = post(clients[0], "/register", {
            "username": "device_progress_test", "password": "test-password",
            "organization_id": 1, "participant_count": 10,
            "device_key": "persistent-test-device-001",
        })
        assert registered.status_code == 200, registered.text
        assert registered.json()["planter"]["participant_slot"] == 1
        assert connection.scalar(text("SELECT COUNT(*) FROM planters")) == 1
        for number, client in enumerate(clients[1:10], start=2):
            response = login(client, number)
            assert response.status_code == 200, response.text
            assert response.json()["planter"]["participant_slot"] == number

        original_ids = []
        for number, client in enumerate(clients[:10], start=1):
            response = client.get("/api/planter-auth/me/field-points")
            assert response.status_code == 200, response.text
            points = response.json()["points"]
            assert len(points) == 10
            assert {point["participant_slot"] for point in points} == {number}
            original_ids.append({point["assignment_point_id"] for point in points})
        assert len(set.union(*original_ids)) == 100
        assert Counter(connection.execute(text(
            "SELECT participant_slot FROM planter_assignment_points"
        )).scalars()) == {number: 10 for number in range(1, 11)}
        assert login(clients[10], 11).status_code == 409

        first = clients[0]
        owned_ids = sorted(original_ids[0])
        response = first.patch(
            f"/api/planter-auth/me/points/{owned_ids[0]}/status",
            json={"status": "completed"},
            headers={"X-CSRF-Token": first.cookies.get("mv_csrf")},
        )
        assert response.status_code == 200, response.text
        response = post(first, "/me/points/mark-all-completed", {"assignment_point_ids": owned_ids[1:3]})
        assert response.status_code == 200, response.text
        assert connection.scalar(text("SELECT COUNT(*) FROM planting_events")) == 3
        assert post(clients[1], "/me/points/mark-all-completed", {
            "assignment_point_ids": [owned_ids[0]],
        }).status_code == 403
        bindings = connection.execute(text(
            "SELECT slot, device_key_hash FROM organization_participants ORDER BY slot"
        )).all()
        assert len(bindings) == 10
        assert all(device_hash for _, device_hash in bindings)

        assert post(first, "/logout").status_code == 200
        assert first.get("/api/planter-auth/me/field-points").status_code == 401
        response = login(first, 1)
        assert response.status_code == 200, response.text
        assert response.json()["planter"]["participant_slot"] == 1
        restored = first.get("/api/planter-auth/me/field-points").json()["points"]
        assert {point["assignment_point_id"] for point in restored} == original_ids[0]
        assert Counter(point["assignment_status"] for point in restored) == {"completed": 3, "pending": 7}
        assert connection.execute(text(
            "SELECT slot, device_key_hash FROM organization_participants ORDER BY slot"
        )).all() == bindings
        for index, client in enumerate(clients[1:10], start=1):
            points = client.get("/api/planter-auth/me/field-points").json()["points"]
            assert {point["assignment_point_id"] for point in points} == original_ids[index]
            assert {point["assignment_status"] for point in points} == {"pending"}
    finally:
        for client in clients:
            client.close()
