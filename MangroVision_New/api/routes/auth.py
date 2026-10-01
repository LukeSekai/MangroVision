"""Authentication endpoints."""

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional

from planting_database import (
    authenticate_user,
    create_user_session,
    get_user_by_session_token,
    revoke_user_session,
)
from api.security import clear_staff_session, set_staff_session

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class SessionResponse(BaseModel):
    user_id: int
    full_name: str
    role: str


@router.post("/login", response_model=SessionResponse)
def login(body: LoginRequest, response: Response):
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_user_session(user["id"])
    set_staff_session(response, token)
    return SessionResponse(
        user_id=user["id"],
        full_name=user["full_name"],
        role=user.get("role", "planner"),
    )


@router.get("/session")
def get_session():
    user = get_user_by_session_token("")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return {
        "user_id": user["id"],
        "full_name": user["full_name"],
        "role": user.get("role", "planner"),
    }


@router.post("/logout")
def logout(response: Response):
    revoke_user_session("")
    clear_staff_session(response)
    return {"status": "ok"}
