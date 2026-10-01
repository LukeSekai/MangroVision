"""Persistent staff notifications."""

from fastapi import APIRouter, Depends, HTTPException

from api.routes.monitoring import _require_lgu_user
from mangrovision_db.notifications import (
    list_staff_notifications,
    mark_notification_read,
    sync_monitoring_reminders,
)

router = APIRouter(dependencies=[Depends(_require_lgu_user)])


@router.get("")
@router.get("/", include_in_schema=False)
def notifications(user: dict = Depends(_require_lgu_user)):
    return list_staff_notifications(int(user["id"]))


@router.post("/refresh")
def refresh_notifications(user: dict = Depends(_require_lgu_user)):
    sync_monitoring_reminders()
    return list_staff_notifications(int(user["id"]))


@router.post("/{notification_id}/read")
def read_notification(notification_id: int, user: dict = Depends(_require_lgu_user)):
    if not mark_notification_read(int(user["id"]), notification_id):
        raise HTTPException(status_code=404, detail="Notification not found.")
    return {"status": "read"}
