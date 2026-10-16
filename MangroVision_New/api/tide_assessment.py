"""Public planting validation using the same coastal graph rule as plantingTides.js.

Use the complete provider series to estimate the graph's reference, then check
every segment of the visit. Missing or stale data stays unknown for LGU review.
"""
import math
import time
from datetime import datetime


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def assess_graph_window(payload, start: datetime, end: datetime, *, now=None):
    unknown = {'status': 'unknown'}
    now = time.time() if now is None else now
    if not isinstance(payload, dict):
        return unknown
    fetched = _number(payload.get('fetched_at'))
    if (payload.get('available') is not True or payload.get('stale') is not False
            or fetched is None or fetched > now + 60 or now - fetched >= 6 * 3600
            or payload.get('datum') != 'MSL' or not payload.get('datum_reference')
            or payload.get('series_available') is not True):
        return unknown
    raw_rows = payload.get('heights')
    if not isinstance(raw_rows, list) or len(raw_rows) < 2:
        return unknown
    rows = []
    for row in raw_rows:
        if not isinstance(row, dict):
            return unknown
        at, height = _number(row.get('timestamp')), _number(row.get('height_m'))
        if (at is None or height is None
                or (rows and not 0 < at - rows[-1][0] <= 3600)):
            return unknown
        rows.append((at, height))
    start_at, end_at = start.timestamp(), end.timestamp()
    if (end_at <= start_at or start_at < rows[0][0] or end_at > rows[-1][0]):
        return unknown

    def inundation(elevation):
        underwater = 0
        total = 0
        for (a, ah), (b, bh) in zip(rows, rows[1:]):
            low, high = min(ah, bh), max(ah, bh)
            fraction = 1 if low > elevation else 0 if high <= elevation else (high - elevation) / (high - low)
            underwater += (b - a) * fraction
            total += b - a
        return underwater / total * 100

    low, high = min(row[1] for row in rows), max(row[1] for row in rows)
    for _ in range(60):
        middle = (low + high) / 2
        if inundation(middle) > 30:
            low = middle
        else:
            high = middle
    threshold = max(0, high)
    maximum = -math.inf
    for (a, ah), (b, bh) in zip(rows, rows[1:]):
        if b <= start_at or a >= end_at:
            continue
        for at in (max(start_at, a), min(end_at, b)):
            maximum = max(maximum, ah + (bh - ah) * (at - a) / (b - a))
    if not math.isfinite(maximum):
        return unknown
    return {'status': 'safe' if maximum <= threshold else 'unsafe',
            'threshold_m': threshold, 'maximum_m': maximum}
