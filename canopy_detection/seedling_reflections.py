"""Reject colored sun-glint fringes without treating pale leaves as glare."""
import cv2
import numpy as np

try:
    from .seedling_leaf_evidence import _compact_leaf_pixels, _has_compact_leaf
except ImportError:
    from seedling_leaf_evidence import _compact_leaf_pixels, _has_compact_leaf


def filter_seedling_reflections(image, entries, gsd):
    """Check every candidate against repeated highlights and local plant evidence.

    Camera clipping can leave a yellow/green rim around a white reflection.
    Its apparent color and contrast are not independent evidence of foliage.
    Reject it only near a clipped highlight on a weakly green background with
    another separate glint, and no compact non-clipped green leaf or connected
    dark plant structure. All distances/areas use the source image's GSD.
    """
    metadata = {'seedling_reflection_rejected_count': 0,
                'seedling_reflection_rejected_centers': [],
                'seedling_reflection_rejected_evidence': []}
    if not entries or gsd is None or not np.isfinite(gsd) or gsd <= 0:
        return list(entries), metadata
    blue, green, red = cv2.split(image)
    clipped = ((green >= 245) & (red >= 245)
               & (np.abs(green.astype(np.int16) - red.astype(np.int16)) <= 10))
    if not np.any(clipped):
        return list(entries), metadata
    proximity = max(1, int(np.ceil(.10 / gsd)))
    near_highlight = cv2.dilate(clipped.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*proximity+1, 2*proximity+1),
    ))
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    fringe_radius = max(1, int(np.ceil(.04 / gsd)))
    highlight_fringe = cv2.dilate(clipped.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*fringe_radius+1, 2*fringe_radius+1),
    ))
    leaf_margin = max(1, int(np.ceil(.02 / gsd)))
    immediate_fringe = cv2.dilate(clipped.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*leaf_margin+1, 2*leaf_margin+1),
    ))
    height, width = image.shape[:2]
    radius = max(3, int(np.ceil(.60 / gsd)))
    min_highlight = max(2, int(np.ceil(.00025 / gsd**2)))
    kept = []
    for entry in entries:
        cx, cy = entry['center']
        x, y = int(round(cx)), int(round(cy))
        if not (0 <= x < width and 0 <= y < height) or not near_highlight[y, x]:
            kept.append(entry)
            continue
        x0, x1 = max(0, x-radius), min(width, x+radius+1)
        y0, y1 = max(0, y-radius), min(height, y+radius+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance2 = ((xx-cx)**2 + (yy-cy)**2) * gsd**2
        ring = (distance2 >= .16**2) & (distance2 <= .40**2)
        if np.count_nonzero(ring) < 24:
            kept.append(entry)
            continue
        b, g, r = [channel[y0:y1, x0:x1].astype(np.float32) for channel in (blue, green, red)]
        h, s, v = [channel[y0:y1, x0:x1] for channel in (hue, saturation, value)]
        exg = 2*g-r-b
        chroma = (g-np.maximum(r, b)) / np.maximum(g+r+b, 1)
        ring_chroma = float(np.median(chroma[ring]))
        ring_exg = float(np.median(exg[ring]))
        # Green foliage and strongly green backgrounds keep their existing path.
        if ring_chroma > .012 or ring_exg > 30:
            kept.append(entry)
            continue
        # Roots, mud relief and stems can surround genuinely pale leaves.
        # Infer open water only from a smooth remaining background, after
        # removing clipped glints and their immediate camera-color fringes.
        # Glints must not inflate their own background texture measurement.
        background = ring & (highlight_fringe[y0:y1, x0:x1] == 0)
        if np.count_nonzero(background) < 24:
            kept.append(entry)
            continue
        low, high = np.percentile(v[background], (25, 75))
        background_spread = float(high-low)
        if background_spread >= 12:
            kept.append(entry)
            continue
        bright = clipped[y0:y1, x0:x1] & (distance2 <= .60**2)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), 8)
        own_labels = set(np.unique(labels[distance2 <= .12**2]))
        # Separate glints must lie beyond the candidate's own colored fringe;
        # threshold gaps within one bright leaf cannot establish a glare field.
        other_glints = sum(
            label not in own_labels and stats[label, cv2.CC_STAT_AREA] >= min_highlight
            and np.any((labels == label) & (distance2 >= .16**2))
            for label in range(1, count)
        )
        if not other_glints:
            kept.append(entry)
            continue
        separate_color = ((distance2 <= .25**2) & (s >= 55) & (v >= 50) & (v < 235)
                          & (exg >= 35) & (immediate_fringe[y0:y1, x0:x1] == 0))
        independent_green = separate_color & (h >= 34) & (h <= 80) & (chroma >= .018)
        if _has_compact_leaf(independent_green, gsd):
            kept.append(entry)
            continue
        # Yellow leaves need not be strongly green. A substantial compact
        # colored body away from clipping is independent evidence; scattered
        # fringes cannot be summed into an artificial leaf.
        _, yellow_areas = _compact_leaf_pixels(
            separate_color & (h >= 26) & (h <= 72) & (chroma >= .008), gsd,
        )
        if max(yellow_areas, default=0) >= max(3, int(np.ceil(.002 / gsd**2))):
            kept.append(entry)
            continue
        background_value = float(np.median(v[ring]))
        dark = ((distance2 <= .20**2)
                & (v <= min(background_value-25, background_value*.75)))
        _, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
        nearby_labels = np.unique(dark_labels[distance2 <= .12**2])
        nearby_labels = nearby_labels[nearby_labels > 0]
        dark_area = int(max(dark_stats[nearby_labels, cv2.CC_STAT_AREA], default=0))
        if dark_area >= max(3, int(np.ceil(.002 / gsd**2))):
            kept.append(entry)
            continue
        metadata['seedling_reflection_rejected_centers'].append([float(cx), float(cy)])
        metadata['seedling_reflection_rejected_evidence'].append({
            'center': [float(cx), float(cy)], 'reason': 'specular_water_highlight',
            'other_highlights': int(other_glints), 'background_chroma': ring_chroma,
            'background_exg': ring_exg, 'connected_dark_area_m2': dark_area * gsd**2,
            'background_value_iqr': background_spread,
        })
    metadata['seedling_reflection_rejected_count'] = len(metadata['seedling_reflection_rejected_centers'])
    return kept, metadata
