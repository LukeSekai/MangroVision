"""Record successful workspace actions without logging page views or request data."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from mangrovision_db.activity import record_activity, request_activity_count
from planting_database import get_planter_by_session_token, get_user_by_session_token


_logger = logging.getLogger(__name__)
_ID = re.compile(r"/\d+(?=/|$)")


@dataclass(frozen=True)
class ActivityDescription:
    action: str
    summary: str


# The first key is the HTTP method; IDs in the URL become {id}. Keep the
# catalog explicit so new mutating endpoints are reviewed for useful wording.
_ACTIVITIES = {
    ("DELETE", "/api/analyses/{id}"): ("analysis.deleted", "Deleted an image analysis."),
    ("POST", "/api/analyses/save"): ("analysis.saved", "Saved an image analysis and its planting points."),
    ("PATCH", "/api/planters/map-points/{id}/death"): ("point.marked_dead", "Marked a planting point as dead."),
    ("POST", "/api/planters/map-points/{id}/restore"): ("point.restored", "Restored a planting point to planted."),
    ("POST", "/api/planters/map-points/{id}/reset-to-planned"): ("point.reset", "Reset a planting point to planned."),
    ("POST", "/api/planters"): ("organization_account.created", "Created an organization account."),
    ("PATCH", "/api/planters/{id}"): ("organization_account.updated", "Updated an organization account."),
    ("DELETE", "/api/planters/{id}"): ("organization_account.removed", "Deactivated or deleted an organization account."),
    ("POST", "/api/planters/{id}/assignments"): ("assignment.created", "Created a planting assignment."),
    ("POST", "/api/planters/{id}/participants/{id}/reset-device"): ("participant.device_reset", "Reset a participant device."),
    ("POST", "/api/seedlings/lgu-plantings"): ("seedling.planted", "Recorded LGU seedling planting."),
    ("PATCH", "/api/seedlings/records/{id}"): ("seedling.details_updated", "Updated a seedling record."),
    ("PATCH", "/api/assignments/points/{id}/status"): ("point.status_updated", "Updated an assigned planting point."),
    ("POST", "/api/assignments/{id}/archive"): ("assignment.archived", "Archived a planting assignment."),
    ("DELETE", "/api/assignments/{id}"): ("assignment.deleted", "Deleted a planting assignment."),
    ("POST", "/api/planter-auth/me/points/mark-all-completed"): ("point.completed", "Recorded completed planting points."),
    ("PATCH", "/api/planter-auth/me/points/{id}/status"): ("point.status_updated", "Updated an assigned planting point."),
    ("PUT", "/api/dashboard/settings"): ("dashboard.settings_updated", "Updated dashboard targets or inspection settings."),
    ("POST", "/api/monitoring/organization-records"): ("monitoring.visit_recorded", "Recorded an organization monitoring visit."),
    ("PUT", "/api/monitoring/organization-records/{id}/death-locations"): ("monitoring.deaths_corrected", "Corrected death locations in a monitoring visit."),
    ("POST", "/api/monitoring/replanting/{id}/approve"): ("replanting.approved", "Approved a replanting request."),
    ("POST", "/api/monitoring/replanting/approve"): ("replanting.approved", "Approved replanting requests."),
    ("POST", "/api/monitoring/replanting/assign"): ("replanting.assigned", "Assigned approved replanting work."),
    ("POST", "/api/monitoring/observations"): ("monitoring.seedling_inspected", "Recorded a seedling inspection."),
    ("POST", "/api/notifications/{id}/read"): ("notification.read", "Marked a notification as read."),
    ("POST", "/api/planting-schedules"): ("schedule.created", "Created a planting schedule."),
    ("POST", "/api/like-appointments/{id}/review"): ("appointment.reviewed", "Reviewed a LIKE website appointment request."),
    ("PUT", "/api/planting-schedules/{id}"): ("schedule.updated", "Updated a planting schedule."),
    ("PATCH", "/api/planting-schedules/{id}"): ("schedule.updated", "Updated a planting schedule."),
    ("DELETE", "/api/planting-schedules/{id}"): ("schedule.deleted", "Deleted a planting schedule."),
    ("POST", "/api/project-sites"): ("project_site.created", "Created a project site."),
    ("PUT", "/api/project-sites/{id}"): ("project_site.updated", "Updated a project site."),
    ("PATCH", "/api/project-sites/{id}"): ("project_site.updated", "Updated a project site."),
    ("DELETE", "/api/project-sites/{id}"): ("project_site.deleted", "Deleted a project site."),
    ("PUT", "/api/project-sites/{id}/tide-calibration"): ("project_site.tide_calibrated", "Updated a project site's tide calibration."),
    ("PUT", "/api/zones/eroded"): ("zone.eroded_replaced", "Saved the eroded-zone map."),
    ("POST", "/api/zones/eroded"): ("zone.eroded_created", "Added an eroded zone."),
    ("DELETE", "/api/zones/eroded/{id}"): ("zone.eroded_deleted", "Deleted an eroded zone."),
    ("POST", "/api/zones/warnings"): ("zone.warning_created", "Added a warning zone."),
    ("PATCH", "/api/zones/warnings/{id}"): ("zone.warning_updated", "Updated a warning zone."),
    ("DELETE", "/api/zones/warnings/{id}"): ("zone.warning_deleted", "Deleted a warning zone."),
    ("POST", "/api/export/csv"): ("export.csv", "Exported planting points as CSV."),
    ("POST", "/api/export/gpx"): ("export.gpx", "Exported planting points as GPX."),
    ("POST", "/api/export/kml"): ("export.kml", "Exported planting points as KML."),
    ("POST", "/api/export/geojson"): ("export.geojson", "Exported planting points as GeoJSON."),
    ("POST", "/api/share/field-link/cloudflare"): ("field_link.started", "Generated a field sharing link."),
    ("POST", "/api/share/field-link/stop"): ("field_link.stopped", "Stopped the field sharing link."),
}

# These POSTs are either read-only, automatic background checks, handled by a
# workflow-specific activity entry, or intentionally return an error.
_EXCLUDED = {
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/logout"),
    ("POST", "/api/planter-auth/register"),
    ("POST", "/api/planter-auth/login"),
    ("POST", "/api/planter-auth/logout"),
    ("POST", "/api/analyses/preflight"),
    ("POST", "/api/analyses/process"),
    ("POST", "/api/analyses/process-stream"),
    ("POST", "/api/analyses/jobs"),
    ("POST", "/api/notifications/refresh"),
    ("POST", "/api/routing/compute"),
    ("POST", "/api/planters/assign-point"),
}


def describe_activity(method: str, path: str) -> ActivityDescription | None:
    route = _ID.sub("/{id}", path.rstrip("/"))
    key = (method.upper(), route)
    if key in _EXCLUDED:
        return None
    values = _ACTIVITIES.get(key)
    return ActivityDescription(*values) if values else None


def _write_for_actor(event: ActivityDescription, method: str, path: str,
                     staff_token: str, planter_token: str) -> None:
    fields = {}
    if staff_token:
        user = get_user_by_session_token(staff_token)
        if user:
            fields = {"actor_type": "staff", "actor_user_id": int(user["id"])}
    if not fields and planter_token:
        planter = get_planter_by_session_token(planter_token)
        if planter:
            fields = {
                "actor_type": "planter",
                "actor_planter_id": int(planter["id"]),
                "participant_slot": planter.get("participant_slot"),
                "organization_id": planter.get("organization_id"),
            }
    if not fields:
        return
    ids = [int(value) for value in re.findall(r"/(?P<id>\d+)(?=/|$)", path)]
    summary = event.summary
    if ids:
        summary = f"{summary.rstrip('.')} (record ID {ids[0]})."
    record_activity(
        action=event.action,
        summary=summary,
        details={"method": method, "resource_ids": ids},
        **fields,
    )


class ActivityAuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        event = describe_activity(request.method, request.url.path)
        if event is None:
            return await call_next(request)
        counter = [0]
        reset = request_activity_count.set(counter)
        try:
            response = await call_next(request)
            if 200 <= response.status_code < 300 and counter[0] == 0:
                try:
                    await run_in_threadpool(
                        _write_for_actor, event, request.method, request.url.path,
                        request.cookies.get("mv_staff_session", ""),
                        request.cookies.get("mv_planter_session", ""),
                    )
                except Exception:
                    # The user action has already committed. Never turn it into
                    # a 500 response that could prompt a duplicate submission.
                    _logger.exception("Could not append activity for %s %s", request.method, request.url.path)
            return response
        finally:
            request_activity_count.reset(reset)
