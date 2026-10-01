"""Cookie session transport, request context, and CSRF enforcement."""

from __future__ import annotations

import hmac
import secrets
from urllib.parse import urlsplit

from fastapi import Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from mangrovision_db.config import get_settings
from mangrovision_db.request_context import planter_session_token, staff_session_token

STAFF_COOKIE = "mv_staff_session"
PLANTER_COOKIE = "mv_planter_session"
CSRF_COOKIE = "mv_csrf"
CSRF_HEADER = "x-csrf-token"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}
_CSRF_EXEMPT_PATHS = {
    "/api/auth/login",
    "/api/planter-auth/login",
    "/api/planter-auth/register",
}


def _same_origin(origin: str, request_url: str) -> bool:
    """Accept the site's own origin, including its LAN/share-link address.

    Use the actual request URL; forwarded host headers are not trusted here.
    Uvicorn separately validates which proxies may provide the URL scheme.
    """
    try:
        source, target = urlsplit(origin), urlsplit(request_url)
        if (source.scheme not in {'http', 'https'} or not source.hostname
                or source.username or source.password or source.path not in {'', '/'}
                or source.query or source.fragment):
            return False
        def authority(url):
            return (url.scheme, url.hostname, url.port or (443 if url.scheme == 'https' else 80))
        return authority(source) == authority(target)
    except ValueError:
        return False


def _cookie_options(max_age: int) -> dict:
    settings = get_settings()
    return {
        "max_age": max_age,
        "secure": settings.cookie_secure,
        "httponly": True,
        "samesite": "lax",
        "domain": settings.cookie_domain,
        "path": "/",
    }


def set_staff_session(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        STAFF_COOKIE,
        token,
        **_cookie_options(settings.staff_session_hours * 60 * 60),
    )
    set_csrf_cookie(response)


def set_planter_session(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        PLANTER_COOKIE,
        token,
        **_cookie_options(settings.planter_session_days * 24 * 60 * 60),
    )
    set_csrf_cookie(response)


def set_csrf_cookie(response: Response) -> None:
    settings = get_settings()
    response.set_cookie(
        CSRF_COOKIE,
        secrets.token_urlsafe(24),
        max_age=settings.planter_session_days * 24 * 60 * 60,
        secure=settings.cookie_secure,
        httponly=False,
        samesite="lax",
        domain=settings.cookie_domain,
        path="/",
    )


def clear_staff_session(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(STAFF_COOKIE, path="/", domain=settings.cookie_domain)


def clear_planter_session(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(PLANTER_COOKIE, path="/", domain=settings.cookie_domain)


class SessionSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        staff_reset = staff_session_token.set(request.cookies.get(STAFF_COOKIE, ""))
        planter_reset = planter_session_token.set(request.cookies.get(PLANTER_COOKIE, ""))
        try:
            settings = get_settings()
            origin = (request.headers.get("origin") or "").rstrip("/")
            if origin and origin not in settings.trusted_origins and not _same_origin(origin, str(request.url)):
                return JSONResponse(status_code=403, content={"detail": "Untrusted request origin."})

            has_session = bool(
                request.cookies.get(STAFF_COOKIE) or request.cookies.get(PLANTER_COOKIE)
            )
            if (
                request.method.upper() not in _SAFE_METHODS
                and request.url.path not in _CSRF_EXEMPT_PATHS
                and has_session
            ):
                cookie_token = request.cookies.get(CSRF_COOKIE, "")
                header_token = request.headers.get(CSRF_HEADER, "")
                if not cookie_token or not hmac.compare_digest(cookie_token, header_token):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Missing or invalid CSRF token."},
                    )
            return await call_next(request)
        finally:
            staff_session_token.reset(staff_reset)
            planter_session_token.reset(planter_reset)
