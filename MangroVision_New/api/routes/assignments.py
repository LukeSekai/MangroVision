"""Assignment endpoints."""

from fastapi import APIRouter
from planting_database import (
    list_planter_assignments,
    get_assignment_points,
    update_assignment_point_status,
    archive_planter_assignment,
    delete_planter_assignment,
)
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class UpdateStatusBody(BaseModel):
    status: str


@router.get("/")
def get_assignments(planter_id: Optional[int] = None, active_only: bool = False):
    return list_planter_assignments(planter_id=planter_id, active_only=active_only)


@router.get("/{assignment_id}/points")
def get_points(assignment_id: int):
    return get_assignment_points(assignment_id)


@router.patch("/points/{point_id}/status")
def update_status(point_id: int, body: UpdateStatusBody):
    update_assignment_point_status(point_id, body.status)
    return {"status": "updated"}


@router.post("/{assignment_id}/archive")
def archive(assignment_id: int):
    archive_planter_assignment(assignment_id)
    return {"status": "archived"}


@router.delete("/{assignment_id}")
def delete(assignment_id: int):
    delete_planter_assignment(assignment_id)
    return {"status": "deleted"}
