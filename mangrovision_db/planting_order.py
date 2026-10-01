"""Geographic point selection, separate from the field app's visit order."""
import math


def _axis_indices(values):
    """Recover lattice indices, retaining parity when a row/column is missing."""
    keys = sorted(set(values))
    if len(keys) < 2:
        return {key: 0 for key in keys}
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    # Average the smallest step family to absorb centimetre-scale rounding.
    smallest = min(gaps)
    short_gaps = [gap for gap in gaps if gap <= smallest * 1.5]
    step = sum(short_gaps) / len(short_gaps)
    return {key: math.floor((key - keys[0]) / step + 0.5) for key in keys}


def zigzag_assignment_points(points):
    """Select geographic zigzag strips before dividing participant shares.

    Pair adjacent latitude levels. Within each pair, alternate the selected
    level at every longitude column. On the staggered hex grid this joins the
    offset points into a W-shaped strip; on a rectangular grid it makes two
    complementary W-shaped strips. Taking a participant's slice thus changes
    which locations they own, not just the visiting order of a rectangular block.

    Coordinates are never moved. Survey boundaries and gaps remain gaps; short
    strips at clipped edges continue into the next strip to keep shares balanced.
    """
    grids, missing = {}, []
    for point in points:
        lat, lon = point.get('latitude'), point.get('longitude')
        if lat is None or lon is None or not math.isfinite(float(lat)) or not math.isfinite(float(lon)):
            missing.append(point)
            continue
        grids.setdefault(int(point.get('analysis_id') or 0), []).append(point)
    tie = lambda p: (int(p.get('point_num') or 0), int(p['id']))
    ordered = []
    for grid_id in sorted(grids):
        grid = grids[grid_id]
        quantize = lambda value: math.floor(float(value) * 10_000_000 + 0.5)
        rows = _axis_indices([quantize(p['latitude']) for p in grid])
        columns = _axis_indices([quantize(p['longitude']) for p in grid])

        def strip_key(point):
            row = rows[quantize(point['latitude'])]
            column = columns[quantize(point['longitude'])]
            return (row // 2, (row + column) % 2, column, *tie(point))

        ordered.extend(sorted(grid, key=strip_key))
    return ordered + sorted(missing, key=tie)


def zigzag_points(points):
    """Keep each survey grid together and snake through its geographic rows.

    GPS coordinates, not image pixels, define rows: source photographs can be
    rotated. Seven decimal places absorb coordinate noise at centimetre scale
    without merging the half-metre staggered rows used by the planting grid.
    """
    grids = {}
    missing = []
    for point in points:
        lat, lon = point.get('latitude'), point.get('longitude')
        if lat is None or lon is None or not math.isfinite(float(lat)) or not math.isfinite(float(lon)):
            missing.append(point)
            continue
        grid = grids.setdefault(int(point.get('analysis_id') or 0), {})
        row = round(float(lat) * 10_000_000)
        grid.setdefault(row, []).append(point)
    ordered = []
    for grid_id in sorted(grids):
        rows = grids[grid_id]
        for index, row in enumerate(sorted(rows, reverse=True)):
            direction = -1 if index % 2 else 1
            ordered.extend(sorted(rows[row], key=lambda p: (
                direction * float(p['longitude']), int(p.get('point_num') or 0), int(p['id']),
            )))
    ordered.extend(sorted(missing, key=lambda p: (int(p.get('point_num') or 0), int(p['id']))))
    return ordered
