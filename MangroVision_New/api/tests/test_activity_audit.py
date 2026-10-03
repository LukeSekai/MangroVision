"""Keep the activity catalog in step with workspace actions."""

from __future__ import annotations

import sys
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.activity_audit import (
    ActivityAuditMiddleware, _ACTIVITIES, _EXCLUDED, _ID, _write_for_actor,
    describe_activity,
)
from mangrovision_db.activity import append_activity


class _FakeConnection:
    def execute(self, _query, _params):
        return None


class ActivityAuditTests(unittest.TestCase):
    def test_every_current_write_route_is_classified(self):
        from api.main import app

        missing = []
        for included in app.router.routes:
            if not hasattr(included, "original_router"):
                continue
            prefix = included.include_context.prefix
            for route in included.original_router.routes:
                if not hasattr(route, "path"):
                    continue
                path = (prefix + route.path).rstrip("/")
                template = re.sub(r"\{[^{}]+\}", "{id}", _ID.sub("/{id}", path))
                for method in (getattr(route, "methods", set()) or set()):
                    if method not in {"POST", "PUT", "PATCH", "DELETE"}:
                        continue
                    if (method, template) not in _ACTIVITIES and (method, template) not in _EXCLUDED:
                        missing.append(f"{method} {path}")
        self.assertEqual(missing, [])

    def test_navigation_and_background_checks_are_not_logged(self):
        self.assertIsNone(describe_activity("GET", "/api/project-sites"))
        self.assertIsNone(describe_activity("POST", "/api/notifications/refresh"))
        self.assertIsNone(describe_activity("POST", "/api/analyses/preflight"))
        self.assertEqual(
            describe_activity("POST", "/api/planting-schedules/").action,
            "schedule.created",
        )

    def test_existing_transaction_activity_suppresses_generic_duplicate(self):
        app = FastAPI()
        app.add_middleware(ActivityAuditMiddleware)

        @app.post("/api/planting-schedules")
        def save_schedule():
            append_activity(
                _FakeConnection(), action="schedule.created", actor_type="staff",
                summary="Created a planting schedule.",
            )
            return {"status": "saved"}

        with patch("api.activity_audit._write_for_actor") as writer:
            response = TestClient(app).post("/api/planting-schedules")
        self.assertEqual(response.status_code, 200)
        writer.assert_not_called()

    def test_failed_action_is_not_logged(self):
        app = FastAPI()
        app.add_middleware(ActivityAuditMiddleware)

        @app.delete("/api/planting-schedules/7")
        def failed_delete():
            return Response(status_code=409)

        with patch("api.activity_audit._write_for_actor") as writer:
            response = TestClient(app).delete("/api/planting-schedules/7")
        self.assertEqual(response.status_code, 409)
        writer.assert_not_called()

    def test_success_is_logged_but_redirect_is_not(self):
        app = FastAPI()
        app.add_middleware(ActivityAuditMiddleware)

        @app.post("/api/planting-schedules")
        def saved_schedule():
            return {"status": "saved"}

        @app.post("/api/project-sites")
        def redirected_site():
            return Response(status_code=307)

        with patch("api.activity_audit._write_for_actor") as writer:
            client = TestClient(app, follow_redirects=False)
            self.assertEqual(client.post("/api/planting-schedules").status_code, 200)
            self.assertEqual(client.post("/api/project-sites").status_code, 307)
        writer.assert_called_once()

    def test_staff_action_records_actor_and_target_without_request_body(self):
        event = describe_activity("DELETE", "/api/planting-schedules/7")
        with patch("api.activity_audit.get_user_by_session_token", return_value={"id": 12}), \
                patch("api.activity_audit.record_activity") as writer:
            _write_for_actor(event, "DELETE", "/api/planting-schedules/7", "session-token", "")
        writer.assert_called_once_with(
            action="schedule.deleted",
            summary="Deleted a planting schedule (record ID 7).",
            details={"method": "DELETE", "resource_ids": [7]},
            actor_type="staff", actor_user_id=12,
        )


if __name__ == "__main__":
    unittest.main()
