"""LGU planting and first seedling observations at mapped points."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api.routes.monitoring import _require_lgu_user
from mangrovision_db.seedlings import list_seedling_records, record_lgu_planting, update_seedling_details

router = APIRouter(dependencies=[Depends(_require_lgu_user)])


class LGUPlantingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planting_point_ids: list[int] = Field(min_length=1, max_length=500)
    project_site_id: int = Field(gt=0)
    planted_date: str
    species: str
    initial_height_cm: float | None = Field(default=None, ge=0, le=1000)
    initial_condition: str | None = None
    initial_notes: str | None = Field(default=None, max_length=1000)


class SeedlingDetailsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_height_cm: float | None = Field(default=None, ge=0, le=1000)
    initial_condition: str | None = None
    initial_notes: str | None = Field(default=None, max_length=1000)


@router.post("/lgu-plantings", status_code=201)
def create_lgu_plantings(body: LGUPlantingCreate, user: dict = Depends(_require_lgu_user)):
    try:
        return record_lgu_planting(
            point_ids=body.planting_point_ids,
            project_site_id=body.project_site_id,
            planted_date=body.planted_date,
            species=body.species,
            staff_user_id=int(user["id"]),
            initial_height_cm=body.initial_height_cm,
            initial_condition=body.initial_condition,
            initial_notes=body.initial_notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/records")
def seedling_records(limit: int = Query(default=100, ge=1, le=500)):
    return {"records": list_seedling_records(limit=limit)}


@router.patch("/records/{event_id}")
def edit_seedling_record(event_id: int, body: SeedlingDetailsUpdate,
                         user: dict = Depends(_require_lgu_user)):
    try:
        return update_seedling_details(
            event_id, staff_user_id=int(user["id"]),
            initial_height_cm=body.initial_height_cm,
            initial_condition=body.initial_condition,
            initial_notes=body.initial_notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
