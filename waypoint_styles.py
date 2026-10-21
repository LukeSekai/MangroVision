"""Shared export palette and small generated marker icons; no database access."""
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

EXPORT_PALETTE = json.loads((Path(__file__).resolve().parent / 'MangroVision_New' /
    'client/src/config/pointExportStyles.json').read_text(encoding='utf-8'))


def is_eroded(point):
    return bool(point.get('eroded_unavailable') or point.get('inside_eroded_zone')
                or point.get('status') in {'unavailable', 'eroded_unavailable'})


def point_export_style(point):
    # One solid color per point: death, erosion, planting, then species.
    status = point.get('status')
    if status == 'dead':
        key = 'dead'
    elif is_eroded(point):
        key = 'eroded'
    elif status in {'planted', 'completed'}:
        key = 'planted'
    else:
        species = str(point.get('species') or '').strip().lower()
        key = species if species in EXPORT_PALETTE['species'] else 'unknown'
    definition = EXPORT_PALETTE['statuses'].get(key) or EXPORT_PALETTE['species'].get(key) or EXPORT_PALETTE['unknown']
    return {'marker_color': definition['color'], 'color_category': key}


def marker_style_id(point):
    return f"point-{point_export_style(point)['marker_color'][1:]}"


def gpx_symbol(point):
    key = point_export_style(point)['color_category']
    definition = EXPORT_PALETTE['statuses'].get(key) or EXPORT_PALETTE['species'].get(key) or EXPORT_PALETTE['unknown']
    return definition['gpx_symbol']


@lru_cache(maxsize=6)
def marker_icon_png(color):
    # Pillow is already a runtime dependency. Import it only for KMZ rendering.
    from PIL import Image, ImageDraw
    scale = 3
    image = Image.new('RGBA', (64 * scale, 64 * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    def box(values):
        return tuple(value * scale for value in values)
    draw.ellipse(box((9, 9, 55, 55)), fill=color, outline=EXPORT_PALETTE['outline_color'], width=2 * scale)
    output = BytesIO()
    image.resize((64, 64), Image.Resampling.LANCZOS).save(output, format='PNG')
    return output.getvalue()
