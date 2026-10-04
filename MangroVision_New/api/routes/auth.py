"""Staff login, email verification, recovery and credential settings."""

from typing import Callable

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from mangrovision_db import staff_auth
from planting_database import get_user_by_session_token, revoke_user_session
from api.security import (
    STAFF_CHALLENGE_COOKIES, clear_staff_challenge, clear_staff_session,
    set_staff_challenge, set_staff_session,
)

router = APIRouter()
ADMIN_RECOVERY_EMAIL = 'mangrovision.lgu@gmail.com'


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=256)


class CodeRequest(BaseModel):
    code: str = Field(min_length=1, max_length=12)


class RecoveryCompleteRequest(CodeRequest):
    model_config = ConfigDict(extra='forbid')
    new_password: str = Field(min_length=12, max_length=128)


class AccountChangeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=128)


def _call(action: Callable):
    try:
        return action()
    except staff_auth.AuthError as error:
        headers = {'Retry-After':str(error.retry_after)} if error.retry_after else None
        raise HTTPException(status_code=error.status, detail=str(error), headers=headers) from None


def _address(request: Request) -> str:
    return request.client.host if request.client else 'unknown'


def _challenge(request: Request, purpose: str) -> str:
    token = request.cookies.get(STAFF_CHALLENGE_COOKIES[purpose], '')
    if not token:
        raise HTTPException(status_code=401, detail='Verification expired. Please start again.')
    return token


def _account() -> dict:
    user = get_user_by_session_token('')
    if not user:
        raise HTTPException(status_code=401, detail='Sign in to manage your account.')
    return user


def _begin(result: tuple, response: Response, purpose: str) -> dict:
    token, data = result
    response.headers['Cache-Control'] = 'no-store'
    set_staff_challenge(response, purpose, token)
    return data


def _session(user: dict) -> dict:
    return {'user_id':user['id'], 'full_name':user['full_name'], 'role':user.get('role', 'planner')}


@router.post('/login')
def login(body: LoginRequest, request: Request, response: Response):
    return _begin(_call(lambda: staff_auth.begin_login(body.username, body.password, _address(request))), response, 'login')


@router.post('/login/verify')
def verify_login(body: CodeRequest, request: Request, response: Response):
    user, token = _call(lambda: staff_auth.finish(_challenge(request, 'login'), body.code, 'login'))
    set_staff_session(response, token)
    clear_staff_challenge(response, 'login')
    response.headers['Cache-Control'] = 'no-store'
    return _session(user)


@router.post('/login/resend')
def resend_login(request: Request, response: Response):
    return _begin(_call(lambda: staff_auth.resend(_challenge(request, 'login'), 'login', _address(request))), response, 'login')


@router.post('/recovery/request')
def request_recovery(request: Request, response: Response):
    return _begin(_call(lambda: staff_auth.begin_recovery(ADMIN_RECOVERY_EMAIL, _address(request))), response, 'recovery')


@router.post('/recovery/complete')
def complete_recovery(body: RecoveryCompleteRequest, request: Request, response: Response):
    _call(lambda: staff_auth.finish(_challenge(request, 'recovery'), body.code, 'recovery',
                                   password=body.new_password))
    clear_staff_challenge(response, 'recovery')
    clear_staff_session(response)
    return {'status':'ok', 'message':'Password reset. Sign in with your new password and a new email code.'}


@router.get('/account')
def get_account(response: Response):
    user = _account()
    response.headers['Cache-Control'] = 'no-store'
    return {**_session(user), 'username':user['full_name'], 'email':user['email']}


@router.post('/account/change')
def request_account_change(body: AccountChangeRequest, request: Request, response: Response):
    user = _account()
    return _begin(_call(lambda: staff_auth.begin_account_change(user, body.current_password,
        body.new_password, _address(request))), response, 'settings')


@router.post('/account/verify')
def verify_account_change(body: CodeRequest, request: Request, response: Response):
    user = _account()
    _call(lambda: staff_auth.finish(_challenge(request, 'settings'), body.code, 'settings', user_id=user['id']))
    clear_staff_challenge(response, 'settings')
    clear_staff_session(response)
    return {'status':'ok', 'message':'Password changed. Sign in again with your new password.'}


@router.post('/account/resend')
def resend_account_change(request: Request, response: Response):
    user = _account()
    return _begin(_call(lambda: staff_auth.resend(_challenge(request, 'settings'), 'settings', _address(request), user_id=user['id'])), response, 'settings')


@router.get('/session')
def get_session(response: Response):
    response.headers['Cache-Control'] = 'no-store'
    return _session(_account())


@router.post('/logout')
def logout(response: Response):
    revoke_user_session('')
    clear_staff_session(response)
    for purpose in STAFF_CHALLENGE_COOKIES:
        clear_staff_challenge(response, purpose)
    return {'status':'ok'}
