"""Staff-only LGU workspace, credentials, and site monitoring."""
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from api.routes.monitoring import _require_lgu_user
from mangrovision_db import lgu_sites

router = APIRouter(dependencies=[Depends(_require_lgu_user)])


class SiteVisit(BaseModel):
    model_config = ConfigDict(extra='forbid')
    monitored_at: str
    version: str = Field(min_length=64, max_length=64)
    dead_planting_event_ids: list[int] = Field(default_factory=list, max_length=10000)
    health_status: str
    actions_taken: str = Field(min_length=1, max_length=2000)
    death_reason_category: str | None = None
    death_notes: str = Field(default='', max_length=500)
    average_height_cm: FiniteFloat | None = Field(default=None, ge=0, le=1000)


@router.get('')
def workspace(response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    try:
        return lgu_sites.workspace()
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.get('/{site_id}/monitoring')
def context(site_id: int, response: Response, monitored_at: str | None = None):
    response.headers['Cache-Control'] = 'private, no-store'
    try:
        return lgu_sites.site_context(site_id, monitored_at)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post('/{site_id}/monitoring', status_code=201)
def save_visit(site_id: int, body: SiteVisit, user: dict = Depends(_require_lgu_user)):
    try:
        return lgu_sites.record_visit(site_id, **body.model_dump(), inspector_user_id=int(user['id']))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
