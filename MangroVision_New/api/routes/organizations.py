"""Staff-only organization contact, site and member directory."""

from fastapi import APIRouter, Depends, HTTPException, Response

from api.routes.monitoring import _require_lgu_user
from mangrovision_db.organization_directory import list_directory, organization_detail

router = APIRouter(dependencies=[Depends(_require_lgu_user)])


@router.get('')
def organizations(response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    return {'organizations': list_directory()}


@router.get('/{organization_id}')
def organization(organization_id: int, response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    detail = organization_detail(organization_id)
    if detail is None:
        raise HTTPException(status_code=404, detail='Organization not found.')
    return detail
