"""Narrow public LIKE intake and authenticated LGU request review."""

import re
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api.routes.planting_schedules import _require_lgu_user
from mangrovision_db.like_appointments import (
    AppointmentConflict, AppointmentRateLimit, list_appointments, review_appointment, submit_appointment,
)

public_router = APIRouter()
staff_router = APIRouter()
MANILA = timezone(timedelta(hours=8))


class AppointmentCreate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    submission_key: UUID
    organization: str = Field(min_length=1, max_length=200)
    contact_name: str = Field(min_length=1, max_length=100)
    phone: str = Field(min_length=7, max_length=32)
    email: str | None = Field(default=None, max_length=254)
    title: str = Field(default='Mangrove planting activity', min_length=1, max_length=200)
    start_at: datetime
    end_at: datetime
    participants: int = Field(ge=1, le=10000, strict=True)
    notes: str | None = Field(default=None, max_length=2000)
    consent: Literal[True]
    website: str = Field(default='', max_length=200)

    @field_validator('phone')
    @classmethod
    def phone_number(cls, value):
        if len(re.sub(r'\D', '', value)) < 7 or not re.fullmatch(r'[+()\d\s.\-]+', value):
            raise ValueError('Enter a valid phone number.')
        return value

    @field_validator('email')
    @classmethod
    def email_address(cls, value):
        if not value:
            return None
        if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
            raise ValueError('Enter a valid email address.')
        return value

    @field_validator('start_at', 'end_at')
    @classmethod
    def aware_time(cls, value):
        if value.tzinfo is None:
            raise ValueError('Appointment times must include a timezone.')
        return value.astimezone(MANILA)

    @model_validator(mode='after')
    def appointment_window(self):
        if self.end_at <= self.start_at or self.end_at.date() != self.start_at.date():
            raise ValueError('Choose a start and end time on the same Philippine date, with end after start.')
        return self


class AppointmentReview(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    action: Literal['confirmed', 'declined', 'cancelled']
    organization: str | None = Field(default=None, min_length=1, max_length=200)
    organization_id: int | None = Field(default=None, ge=1)
    start_at: datetime | None = None
    end_at: datetime | None = None
    contacted: bool = False
    decision_notes: str | None = Field(default=None, max_length=2000)

    @field_validator('start_at', 'end_at')
    @classmethod
    def aware_time(cls, value):
        return AppointmentCreate.aware_time(value) if value is not None else None

    @model_validator(mode='after')
    def review_window(self):
        if bool(self.start_at) != bool(self.end_at):
            raise ValueError('Supply both the agreed start and end times.')
        if self.start_at and (self.end_at <= self.start_at or self.end_at.date() != self.start_at.date()):
            raise ValueError('Choose a valid time window on the same Philippine date.')
        return self


@public_router.post('/appointments', status_code=201)
def create_appointment(body: AppointmentCreate, request: Request):
    if body.website:
        raise HTTPException(400, 'Could not accept this request. Please contact LGU staff.')
    try:
        return submit_appointment(body.model_dump(), request.client.host if request.client else 'unknown')
    except AppointmentRateLimit as error:
        raise HTTPException(429, str(error), headers={'Retry-After': '3600'}) from error
    except AppointmentConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@staff_router.get('')
def appointments():
    _require_lgu_user()
    return {'requests': list_appointments(), 'timezone': 'Asia/Manila'}


@staff_router.post('/{request_id}/review')
def review(request_id: int, body: AppointmentReview):
    user = _require_lgu_user()
    try:
        return review_appointment(request_id, body.model_dump(), int(user['id']))
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except AppointmentConflict as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
