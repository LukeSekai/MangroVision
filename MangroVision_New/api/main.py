"""
MangroVision v2 — FastAPI backend.

Wraps the existing Python backend modules (planting_database, canopy_detection,
waypoint_export) as a REST API consumed by the React frontend.
"""

import sys
import os
from pathlib import Path

# Add parent MangroVision directory to sys.path so we can import existing modules
_PARENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import (
    analyses,
    assignments,
    auth,
    export,
    planter_auth,
    planters,
    processing,
    routing,
    zones,
)

app = FastAPI(
    title="MangroVision API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

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

# Mount API routes
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(planter_auth.router, prefix="/api/planter-auth", tags=["Planter Auth"])
app.include_router(analyses.router, prefix="/api/analyses", tags=["Analyses"])
app.include_router(planters.router, prefix="/api/planters", tags=["Planters"])
app.include_router(assignments.router, prefix="/api/assignments", tags=["Assignments"])
app.include_router(zones.router, prefix="/api/zones", tags=["Zones"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])
app.include_router(processing.router, prefix="/api/analyses", tags=["Processing"])
app.include_router(routing.router, prefix="/api/routing", tags=["Routing"])


# Serve the WebODM pyramid tiles directly from FastAPI so the React map can load
# the orthophoto overlay without starting a separate tile server on :8080.
_MAP_DIR = _PARENT_DIR / "MAP"
if _MAP_DIR.exists():
    app.mount("/tiles", StaticFiles(directory=str(_MAP_DIR)), name="tiles")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}
