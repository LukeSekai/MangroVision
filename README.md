# MangroVision

**Setting up on another laptop?** Start with the
[step-by-step groupmate setup guide](docs/groupmate-setup.md). It lists what to
clone, what must be copied separately, and how to check that the app runs.

MangroVision is a FastAPI and React mangrove-planning system. It analyzes
geotagged drone imagery, keeps eroded-area planting points visible but
dynamically unavailable, manages planting assignments, and records aggregate
organization monitoring visits for LGU staff.

## Supported production application

Only these applications are supported as production writers:

- `MangroVision_New/api`: FastAPI backend and the only database/storage client.
- `MangroVision_New/client`: React frontend using credentialed HTTP-only cookie sessions.

The older Streamlit and standalone field applications remain historical tools;
they are not permitted to write to the production database.

Production persistence uses PostgreSQL/PostGIS and private S3-compatible object
storage. SQLite and mutable zone GeoJSON files are migration inputs only. See
[Production database operations](docs/production-database.md) for local setup,
migration, cutover, backup, restore, and Supabase deployment instructions.

For a groupmate's laptop, follow the [clone and setup guide](docs/groupmate-setup.md).
The current orthophoto GeoTIFF and preferred canopy checkpoint are shared
separately; Git does not contain the image data, private credentials, or those
two large files.

## Local development

1. Copy `.env.example` to `.env` and replace every placeholder secret.
2. Start Docker Desktop.
3. Start PostgreSQL/PostGIS and MinIO:

   ```powershell
   docker compose up -d
   ```

4. Install Python dependencies and upgrade the database:

   ```powershell
   python -m pip install -r requirements.txt
   alembic upgrade head
   ```

5. Create the first administrator using explicit `BOOTSTRAP_ADMIN_*` values:

   ```powershell
   python scripts/bootstrap_admin.py
   ```

6. Install the frontend dependencies, then start the managed development server
   from the repository root:

   ```powershell
   npm --prefix MangroVision_New/client ci
   python MangroVision_New/start_dev.py
   ```

   This starts FastAPI on port 8000 and React on port 5173. Running the launcher
   again restarts this workspace's existing managed stack. Ctrl+C stops both
   servers; on Windows, closing or terminating the launcher also removes its
   child processes. To stop the stack from another terminal:

   ```powershell
   python MangroVision_New/start_dev.py --stop
   ```

   Use the launcher for both services. Independently launched servers and other
   programs occupying these ports are reported without being terminated.

Readiness is exposed at `/api/health/ready` and fails when PostgreSQL,
Alembic, or private object storage is unavailable.

## Important behavior

- Images without GPS or fully outside active `gis_coverage` zones are rejected
  before analysis. Partial overlap requires explicit user confirmation.
- Eroded zones never delete or hide planting points. Covered points display as
  `Not Available for Planting`; removing the zone dynamically returns eligible
  points to `Planned`.
- Monitoring is recorded once per organization visit: alive/dead totals,
  average height, overall health, and LGU actions. No photo is required.
- Analysis images are private objects and are returned only through short-lived
  signed URLs; image bytes are never stored in PostgreSQL.
