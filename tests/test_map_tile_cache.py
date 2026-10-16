"""Reusable tile headers on both FastAPI tile-serving paths; no database access."""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
from api import map_tiles


def test_static_and_rendered_tiles_allow_browser_and_proxy_reuse(tmp_path, monkeypatch):
    directory = tmp_path / 'FINAL'
    tile = directory / '1' / '0' / '0.png'
    tile.parent.mkdir(parents=True)
    tile.write_bytes(b'fictional tile')
    geotiff = directory / 'ortho.tif'
    geotiff.write_bytes(b'fixture')
    monkeypatch.setattr(map_tiles, '_MAP_DIR', tmp_path)
    monkeypatch.setattr(map_tiles, '_tileset_geotiff_path', lambda _: str(geotiff))
    calls = []
    monkeypatch.setattr(map_tiles, '_render_geotiff_tile', lambda *args: calls.append(args) or b'rendered tile')
    app = FastAPI()
    app.include_router(map_tiles.router)
    with TestClient(app) as client:
        static = client.get('/tiles/FINAL/1/0/0.png?v=imagery-1')
        assert static.status_code == 200
        assert static.headers['cache-control'] == 'public, max-age=3600'
        assert static.headers['etag']
        assert static.content == b'fictional tile'
        assert not calls
        rendered = client.get('/tiles/FINAL/1/0/1.png')
        assert rendered.status_code == 200
        assert rendered.headers['cache-control'] == 'public, max-age=3600'
        assert rendered.content == b'rendered tile'
        invalid = client.get('/tiles/FINAL/1/2/0.png')
        assert invalid.status_code == 404
        assert 'max-age' not in invalid.headers.get('cache-control', '')
