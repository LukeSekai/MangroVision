# MangroVision_New Map Integration

MangroVision_New uses Leaflet XYZ tiles for the visible orthophoto map.
FastAPI serves the root `MAP` directory at:

```text
http://localhost:8000/tiles
```

The default visible map is:

```text
MAP/FINAL MAP/{z}/{x}/{y}.jpg
```

## Export The Map From QGIS

Use this when your QGIS project already looks like the map you want in the app.

1. In QGIS, turn on only the layers you want visible in MangroVision.
2. Set the map canvas to the exact area you want to export.
3. Open `Processing Toolbox`.
4. Search for `Generate XYZ tiles (Directory)`.
5. Set these parameters:
   - `Extent`: choose the current map canvas extent or your final map layer extent.
   - `Minimum zoom`: `12`.
   - `Maximum zoom`: `20`.
   - `Tile format`: `PNG` if you need transparency, otherwise `JPG`.
   - `Quality`: `85` to `95`.
   - `Output directory`: for example `C:\Users\Asus-Pc\Desktop\MangroVision\MAP\TRIAL MAP`.
6. Run the tool.

After export, the folder must look like this:

```text
MAP/
  TRIAL MAP/
    12/
      3443/
        1924.jpg
    13/
    14/
    ...
    20/
```

Do not export a screenshot, PDF, print layout, or one large image for the web map. Leaflet needs XYZ tile folders.

## Use The New Tile Folder In MangroVision_New

If you exported into `MAP/TRIAL MAP`, update:

```text
MangroVision_New/client/.env.development
```

Set:

```text
VITE_TILESET_PATH=TRIAL MAP
VITE_TILE_EXTENSION=png
VITE_ORTHOPHOTO_MAX_NATIVE_ZOOM=20
```

If the QGIS export uses transparent PNG tiles, set:

```text
VITE_TILE_EXTENSION=png
```

Then restart the MangroVision_New dev server:

```powershell
cd C:\Users\Asus-Pc\Desktop\MangroVision\MangroVision_New
python start_dev.py
```

If you prefer to replace the current map directly, export the QGIS tiles into `MAP/FINAL MAP` instead. Back up the old `MAP/FINAL MAP` folder first.

## Update The Forbidden Zones

The old forbidden-zone file can remain in place. MangroVision_New checks these files in order and uses the first one that exists:

```text
forbidden_zone_final.geojson
forbidden_zones_final.geojson
forbidden_zones.geojson
```

Your current QGIS export is:

```text
C:\Users\Asus-Pc\Desktop\MangroVision\forbidden_zone_final.geojson
```

That file is now preferred by the backend for both map display and planting-point filtering.

## Important: Analysis Orthophoto

The visible Leaflet map is only the display layer. The image analysis and GPS mapping still use georeferenced orthophoto `.tif` files through `canopy_detection/ortho_matcher.py`.

If the QGIS map changes the actual orthophoto coverage, also export or keep the georeferenced GeoTIFF and register it in `canopy_detection/ortho_matcher.py`; otherwise the map display can change while coordinate matching still uses the old orthophoto.
