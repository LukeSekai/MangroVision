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
from mangrovision_db.organization_accounts import list_participant_devices, reset_participant_device

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
        unknown = post(clients[10], "/login", {
            "username": "device_progress_test", "password": "test-password",
            "device_key": "unknown-recovery-device-key", "resume_device": True,
        })
        assert unknown.status_code == 409
        assert "recovery code" in unknown.json()["detail"]
        assert connection.scalar(text("SELECT COUNT(*) FROM organization_participants WHERE device_key_hash IS NOT NULL")) == 1
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

        # Repeated sessions on the same device never allocate another slot,
        # even when the organization's device limit is already full.
        for _ in range(5):
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
        # A return visit after the server session expires uses ordinary login,
        # with the browser's saved identity and no recovery code or slot number.
        connection.execute(text("""
            UPDATE auth_sessions SET expires_at = CURRENT_TIMESTAMP - INTERVAL '1 day'
            WHERE subject_type = 'planter' AND subject_id = :id
              AND participant_slot = 1 AND revoked_at IS NULL
        """), {"id": registered.json()["planter"]["id"]})
        assert first.get("/api/planter-auth/session").status_code == 401
        returning = login(first, 1)
        assert returning.status_code == 200, returning.text
        assert returning.json()["planter"]["participant_slot"] == 1
        continued = first.get("/api/planter-auth/me/field-points").json()["points"]
        assert {point["assignment_point_id"] for point in continued} == original_ids[0]
        assert Counter(point["assignment_status"] for point in continued) == {"completed": 3, "pending": 7}
        assert connection.execute(text(
            "SELECT slot, device_key_hash FROM organization_participants ORDER BY slot"
        )).all() == bindings
        summary = list_participant_devices(registered.json()["planter"]["id"])
        assert summary["registered_devices"] == 10
        assert summary["available_devices"] == 0
        assert summary["devices"][0]["assigned_points"] == 10
        assert summary["devices"][0]["completed_points"] == 3
        assert summary["devices"][0]["active_sessions"] == 1
        assert summary["devices"][0]["last_seen_at"]
        assert all("device_key_hash" not in device and "token_hash" not in device for device in summary["devices"])
        # A new shared-link origin has no original cookies. Password + the
        # original device key must resume the same slot even when all are full.
        with TestClient(app, base_url="https://changed-field.example.test",
                        headers={"Origin": "https://changed-field.example.test"}) as changed_link:
            credentials = {"username": "device_progress_test", "password": "test-password",
                           "device_key": "persistent-test-device-001", "resume_device": True}
            response = post(changed_link, "/login", {**credentials, "password": "incorrect"})
            assert response.status_code == 401
            response = post(changed_link, "/login", credentials)
            assert response.status_code == 200, response.text
            assert response.json()["planter"]["participant_slot"] == 1
            resumed = changed_link.get("/api/planter-auth/me/field-points").json()["points"]
            assert {point["assignment_point_id"] for point in resumed} == original_ids[0]
            assert Counter(point["assignment_status"] for point in resumed) == {"completed": 3, "pending": 7}
            assert list_participant_devices(registered.json()["planter"]["id"])["registered_devices"] == 10
            assert connection.execute(text(
                "SELECT slot, device_key_hash FROM organization_participants ORDER BY slot"
            )).all() == bindings
        for index, client in enumerate(clients[1:10], start=1):
            points = client.get("/api/planter-auth/me/field-points").json()["points"]
            assert {point["assignment_point_id"] for point in points} == original_ids[index]
            assert {point["assignment_status"] for point in points} == {"pending"}
        reset_participant_device(registered.json()["planter"]["id"], 1)
        assert list_participant_devices(registered.json()["planter"]["id"])["registered_devices"] == 9
        assert post(first, "/login", {
            "username": "device_progress_test", "password": "test-password",
            "device_key": "persistent-test-device-001", "resume_device": True,
        }).status_code == 409
        wrong_participant = post(clients[1], "/login", {
            "username": "device_progress_test", "password": "test-password",
            "device_key": "persistent-test-device-002", "participant_slot": 1, "recover_slot": True,
        })
        assert wrong_participant.status_code == 409
        assert "already Participant 2" in wrong_participant.json()["detail"]
        replacement = post(first, "/login", {
            "username": "device_progress_test", "password": "test-password",
            "device_key": "replacement-test-device-001", "participant_slot": 1, "recover_slot": True,
        })
        assert replacement.status_code == 200, replacement.text
        assert replacement.json()["planter"]["participant_slot"] == 1
        recovered = first.get("/api/planter-auth/me/field-points").json()["points"]
        assert {point["assignment_point_id"] for point in recovered} == original_ids[0]
        assert Counter(point["assignment_status"] for point in recovered) == {"completed": 3, "pending": 7}
        assert connection.scalar(text("SELECT COUNT(*) FROM planting_events")) == 3
    finally:
        for client in clients:
            client.close()
