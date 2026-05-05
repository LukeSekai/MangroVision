"""Authentication endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from planting_database import (
    authenticate_user,
    create_user_session,
    get_user_by_session_token,
    revoke_user_session,
    ensure_admin_user,
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class SessionResponse(BaseModel):
    token: str
    user_id: int
    full_name: str
    role: str


@router.post("/login", response_model=SessionResponse)
def login(body: LoginRequest):
    ensure_admin_user()
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_user_session(user["id"])
    return SessionResponse(
        token=token,
        user_id=user["id"],
        full_name=user["full_name"],
        role=user.get("role", "planner"),
    )


@router.get("/session")
def get_session(token: str):
    user = get_user_by_session_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return {
        "user_id": user["id"],
        "full_name": user["full_name"],
        "role": user.get("role", "planner"),
    }


@router.post("/logout")
def logout(token: str):
    revoke_user_session(token)
    return {"status": "ok"}
