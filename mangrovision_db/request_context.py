"""Request-scoped opaque session tokens populated by FastAPI middleware."""

from contextvars import ContextVar

staff_session_token: ContextVar[str] = ContextVar("staff_session_token", default="")
planter_session_token: ContextVar[str] = ContextVar("planter_session_token", default="")
