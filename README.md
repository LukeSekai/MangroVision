# MangroVision

MangroVision is a Streamlit-based mangrove planting planner. It analyzes uploaded drone imagery, detects canopy danger zones, filters out forbidden and eroded areas, maps safe planting points onto the orthophoto, and exports field-ready coordinates.

## Current Scope

The repository has been trimmed to the files used by the active application flow:

- `app.py` for the main Streamlit UI
- `start_tile_server.py` plus `START_MANGROVISION.bat` and `STOP_MANGROVISION.bat` for local startup
- `planting_database.py` for SQLite-backed analysis storage
- `waypoint_export.py` for CSV, GPX, KML, and GeoJSON exports
- `canopy_detection/` modules required by the app:
  - `canopy_detector_hexagon.py`
  - `detectree2_proper.py`
  - `exif_extractor.py`
  - `forbidden_zone_filter.py`
  - `gsd_calculator.py`
  - `ortho_matcher.py`

## Runtime Data Kept In Repo

- `forbidden_zones.geojson` for forbidden-zone filtering
- `eroded_zones.geojson` for user-managed erosion exclusions
- `planting_zones.db` for saved analyses and planting points
- `MAP/FINAL MAP/` for the map tiles used by the UI
- `models/` for the AI weights used by detection

## Run The App

### First-time setup

Run:

```powershell
SETUP_ENV.bat
```

This creates the local `venv` and installs the Python packages from `requirements.txt`.

### One-click startup

Run:

```powershell
START_MANGROVISION.bat
```

This starts:

- the tile server on `http://localhost:8080`
- the Streamlit UI on `http://localhost:8502`

### Manual startup

In two terminals from the repo root:

```powershell
venv\Scripts\python.exe start_tile_server.py
```

```powershell
venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

## Main User Flow

1. Sign in to the Streamlit app.
2. Upload a drone image.
3. Run AI detection when detectree2 is available, otherwise use the HSV fallback.
4. Run canopy detection and planting-point generation.
5. Review planting points on the orthophoto map.
6. Save results to the database or export them for field use.

## Notes

- The current app map points to `MAP/FINAL MAP`.
- `MAP/FINAL MAP` is tracked in the repository so teammates can clone the repo and view the orthophoto map.
- `venv/` is intentionally not tracked. The current local environment is about 1.3 GB and includes Windows-specific binaries, including `torch_cpu.dll` at about 253 MB, which will not fit in a normal GitHub push. Use `SETUP_ENV.bat` after cloning instead.
- Auto-alignment in `ortho_matcher.py` still expects the configured WebODM orthophoto sources to exist on the local machine.
- If the optional login background asset is missing, the app already falls back to an embedded gradient background.
