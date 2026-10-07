"""
MangroVision v2 — FastAPI backend.

Wraps the existing Python backend modules (planting_database, canopy_detection,
waypoint_export) as a REST API consumed by the React frontend.
"""

import sys
import os
import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

# Add parent MangroVision directory to sys.path so we can import existing modules
_PARENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from alembic.script import ScriptDirectory
from sqlalchemy import text

from mangrovision_db import get_engine
from mangrovision_db.config import get_settings
from mangrovision_db.storage import storage_ready

from api import map_tiles
from api.error_responses import install_error_responses
from api.security import SessionSecurityMiddleware
from api.activity_audit import ActivityAuditMiddleware
from api.read_compression import WorkspaceReadCompression
from api.routes import (
    analyses,
    activity,
    assignments,
    auth,
    dashboard,
    env_context,
    export,
    monitoring,
    notifications,
    planting_schedules,
    like_appointments,
    planter_auth,
    planters,
    processing,
    project_sites,
    routing,
    seedlings,
    share,
    tides,
    zones,
)
from mangrovision_db.appointment_email import run_delivery_cycle

_EXPECTED_DB_REVISION = ScriptDirectory(_PARENT_DIR / "alembic").get_current_head()

@asynccontextmanager
async def lifespan(_app):
    async def deliver_emails():
        while True:
            await asyncio.sleep(60)
            await asyncio.to_thread(run_delivery_cycle)
    worker = asyncio.create_task(deliver_emails())
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


app = FastAPI(
    title="MangroVision API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)
install_error_responses(app)

_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins_env = os.getenv("MANGROVISION_CORS_ORIGINS", "").strip()
_extra_origins = [origin.strip() for origin in _extra_origins_env.split(",") if origin.strip()]
_cors_origins = _default_origins + _extra_origins

# Allow common private-network origins for phone testing on local Wi-Fi.
_cors_origin_regex = os.getenv(
    "MANGROVISION_CORS_ORIGIN_REGEX",
    r"^(https?://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+|192\.168\.\d+\.\d+)(:\d+)?|https://[a-z0-9-]+\.trycloudflare\.com|https://[a-z0-9-]+\.ngrok-free\.app|https://[a-z0-9-]+\.ngrok\.app)$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(WorkspaceReadCompression)
app.add_middleware(ActivityAuditMiddleware)
app.add_middleware(SessionSecurityMiddleware)

# Mount API routes
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(planter_auth.router, prefix="/api/planter-auth", tags=["Planter Auth"])
app.include_router(analyses.router, prefix="/api/analyses", tags=["Analyses"])
app.include_router(activity.router, prefix="/api/activity", tags=["Activity"])
app.include_router(planters.router, prefix="/api/planters", tags=["Planters"])
app.include_router(seedlings.router, prefix="/api/seedlings", tags=["Seedlings"])
app.include_router(assignments.router, prefix="/api/assignments", tags=["Assignments"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["Notifications"])
app.include_router(planting_schedules.router, prefix="/api/planting-schedules", tags=["Planting Schedules"])
app.include_router(like_appointments.public_router, prefix="/api/public/like", tags=["Public LIKE"])
app.include_router(like_appointments.staff_router, prefix="/api/like-appointments", tags=["LIKE Appointments"])
app.include_router(project_sites.router, prefix="/api/project-sites", tags=["Project Sites"])
app.include_router(zones.router, prefix="/api/zones", tags=["Zones"])
app.include_router(env_context.router, prefix="/api/zones", tags=["Env Context"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])
app.include_router(processing.router, prefix="/api/analyses", tags=["Processing"])
app.include_router(routing.router, prefix="/api/routing", tags=["Routing"])
app.include_router(tides.router, prefix="/api/tides", tags=["Tides"])
app.include_router(share.router, prefix="/api/share", tags=["Share"])
app.include_router(map_tiles.router)


# Serve the WebODM pyramid tiles directly from FastAPI so the React map can load
# the orthophoto overlay without starting a separate tile server on :8080.
_MAP_DIR = _PARENT_DIR / "MAP"
if _MAP_DIR.exists():
    app.mount("/tiles", StaticFiles(directory=str(_MAP_DIR)), name="tiles")

# Inspection photos are written lazily when an LGU officer submits an
# observation. ``check_dir=False`` keeps application startup valid before the
# first upload while still serving the directory once it exists.
_MONITORING_UPLOAD_DIR = _PARENT_DIR / "MangroVision_New" / "monitoring_uploads"
app.mount(
    "/monitoring_uploads",
    StaticFiles(directory=str(_MONITORING_UPLOAD_DIR), check_dir=False),
    name="monitoring-uploads",
)


@app.get("/api/health/live")
def liveness_check():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/health/ready")
def readiness_check():
    settings = get_settings()
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
            revision = connection.execute(
                text(f'SELECT version_num FROM "{settings.db_schema}".alembic_version')
            ).scalar_one_or_none()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Database is not ready: {error}",
        ) from error
    if revision != _EXPECTED_DB_REVISION:
        raise HTTPException(
            status_code=503,
            detail="Database migration revision is not current.",
        )
    if not storage_ready():
        raise HTTPException(status_code=503, detail="Private object storage is not ready.")
    return {
        "status": "ready",
        "version": "2.0.0",
        "database_revision": revision,
        "storage": "ready",
    }


@app.get("/api/health")
def health_check():
    """Backward-compatible readiness alias used by existing launch scripts."""
    return readiness_check()
