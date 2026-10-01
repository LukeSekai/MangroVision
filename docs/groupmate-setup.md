# Run MangroVision on another laptop

This guide is for the current FastAPI and React app. Run commands in PowerShell
from the repository root. The old `START_MANGROVISION.bat` starts the historical
Streamlit app.

If a groupmate is using Codex, ask it to follow these numbered steps in order
and report any missing files or settings. Automated tests are not needed to run
the app. The setup checker in step 6 is a one-time installation check.

## 1. Prepare what Git does and does not contain

Git should carry the API, client, database migrations, map code, model metadata,
and UI images. It does **not** carry these two large runtime files:

| File to share separately | Why it is needed | Size on the reference laptop |
| --- | --- | ---: |
| `MAP/FINAL/final_orthophoto.tif` | Visual alignment and on-demand map tiles | 113.8 MiB |
| `MangroVision_New/try_model/model_final.pth` | Preferred two-class canopy detector | 479.8 MiB |

Copy those exact files by a private shared drive or USB into the same paths
after cloning. `MangroVision_New/try_model/model_metadata.json` should arrive
through Git. The full WebODM folder and the 2,996 generated `MAP/FINAL` PNG
tiles are unnecessary: the API can render tiles from the GeoTIFF. Historical
`MAP/FINAL MAP` tiles are no longer used by the current client.
The old annotation dataset is also excluded; it is needed only to rerun model
evaluation, not to analyze images with the saved checkpoint.

The repository-root `.env` is also excluded from Git. It contains database
and object-storage credentials. The original laptop currently uses a hosted
Supabase database; pulling Git does not copy its 1,101 planting records or
private images. To see the same data, each running API must connect to the
same database and storage bucket. A separate local Docker database starts
without those records.

## 2. Before the owner pushes

Review `git status --short` and commit the current API, client, migrations,
`mangrovision_db`, required API data, public UI images, documentation, and
`.env.example`. Include `AGENTS.md` and `scripts/check_groupmate_setup.py` so
Codex can find this guide and verify the setup. A push includes only committed
files, so review the staged file list before committing:

```powershell
git add -A
git diff --cached --name-status
git diff --cached --check
```

Make sure `.env`, `venv`, `node_modules`, tests, training data, the GeoTIFF, and
the checkpoint have no `A` (added) or `M` (modified) entry. `D` (deleted)
entries for old `MAP/FINAL MAP` tiles, training annotations, and `.claude`
machine settings are expected: they were removed from Git's index while local
copies remain on disk. Commit the reviewed changes and push the branch that
the other laptops will pull.

## 3. Install tools and clone

Install Git, Python 3.12, and Node.js 22.12 or later. In PowerShell:

```powershell
git clone <your-repository-URL>
cd MangroVision
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd --prefix MangroVision_New/client ci
```

The requirements file does not install Detectron2 automatically. Full canopy
analysis needs a compatible PyTorch/torchvision/Detectron2 build. The reference
laptop uses Python 3.12.10, CPU PyTorch 2.10.0, torchvision 0.25.0, and
Detectron2 0.6. Verify that the other laptop can import it:

```powershell
.\venv\Scripts\python.exe -c "import torch, detectron2; print(torch.__version__, detectron2.__version__)"
```

Use the [official Detectron2 installation guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)
for a compatible build. Its maintainers do not provide official Windows
support, so the Python requirements alone may start the app while canopy
analysis uses a fallback detector. The setup check below flags this.

## 4. Copy the two runtime files

Create `MAP/FINAL` and `MangroVision_New/try_model` if absent, then copy the
GeoTIFF and checkpoint into their exact paths from step 1. The app uses the
GeoTIFF automatically at the default location; no WebODM path setting is
needed. Both files are ignored by Git, so `git pull` will not overwrite them.

The setup checker compares their SHA-256 hashes with this laptop's reference
copies. If the project intentionally changes either file later, update the
expected hashes in `scripts/check_groupmate_setup.py` at the same time.

## 5. Configure the database and private images

For the **same shared data**, create a private repository-root `.env` from
`.env.example`. Have the project owner provide authorized runtime settings
through a secure channel: the restricted `DATABASE_URL`, `APP_DATABASE_ROLE`,
`DB_SCHEMA`, `DB_SSLMODE=require`, and the `S3_*` values for the private bucket.
Set `TRUSTED_ORIGINS` to include `http://localhost:5173` and
`http://127.0.0.1:5173`; use `COOKIE_SECURE=false` for local HTTP. Do not put
these values in `MangroVision_New/client/.env*`, chat, or Git. Supabase S3
access keys can access all buckets, so distribute them only to groupmates who
are authorized to run a backend. An owner-hosted API avoids giving each laptop
those credentials. Remove the example `MIGRATION_DATABASE_URL` and
`BOOTSTRAP_ADMIN_*` lines from a collaborator's shared-database `.env`.

When connecting to the already-migrated shared database, **do not** run
`alembic upgrade head`, `bootstrap_admin.py`, or a SQLite data import on every
laptop. Existing accounts and planting records are already in that database.

For an **independent local database**, follow [production-database.md](production-database.md#local-postgresqlpostgis-and-minio)
to start Docker/PostGIS and MinIO, run migrations, sync GIS coverage, and create
a new admin account. That setup will not contain the shared WVSU records unless
the owner deliberately imports approved data.

Optional services: `GOOGLE_ROUTES_API_KEY` enables Google route directions;
`WORLDTIDES_API_KEY` selects WorldTides. The tide screen can use its keyless
fallback. Neither key is needed for image alignment.

## 6. Check and start

```powershell
.\venv\Scripts\python.exe scripts/check_groupmate_setup.py
.\venv\Scripts\python.exe MangroVision_New/start_dev.py
```

Open `http://localhost:5173`. In another PowerShell window, check the API,
database migration, and private storage together:

```powershell
.\venv\Scripts\python.exe scripts/check_groupmate_setup.py --online
```

The offline check must report both large files, the UI images, Detectron2,
Node.js, and the private `.env` as present. The online check must report
`API/database/storage readiness` as OK. Then sign in, confirm that the map
loads, and analyze one geotagged image within the mapped area. A missing
GeoTIFF prevents the same visual alignment and map overlay; a missing model
changes the canopy detector; a missing/shared-data connection changes what
planting and monitoring records appear.

Stop both development servers with Ctrl+C, or run:

```powershell
.\venv\Scripts\python.exe MangroVision_New/start_dev.py --stop
```
