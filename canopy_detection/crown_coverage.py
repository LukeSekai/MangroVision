"""Recover attached foliage and pale interiors of validated mature AI crowns."""
import cv2
import numpy as np


def recover_canopy_crown_coverage(image, canopy_mask, gsd):
    """Extend coverage without deleting AI pixels or detecting new crowns.

    Only validated components of at least 1.2 m² can anchor recovery. Attached
    green foliage can extend up to 1.5 m from that anchor. Bright, low-saturation
    crown material additionally needs to be inside its convex envelope and
    within 0.75 m of accepted pixels; isolated pale wood/water is not a seed.
    A connected group's added area is capped at 35% of its mature anchor area.
    These are ground-scale appearance checks, not a new species classifier.
    """
    metadata = {"canopy_crown_recovered_pixels": 0,
                "canopy_crown_recovered_area_m2": 0.0,
                "canopy_crown_recovery_limited_groups": 0}
    recovered = canopy_mask.copy()
    if gsd is None or not np.isfinite(gsd) or gsd <= 0:
        return recovered, metadata
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (canopy_mask > 0).astype(np.uint8), 8,
    )
    anchors = np.zeros_like(canopy_mask)
    envelopes = np.zeros_like(canopy_mask)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area * gsd**2 < 1.2:
            continue
        component = (labels[y:y+height, x:x+width] == label).astype(np.uint8)
        anchors[y:y+height, x:x+width][component > 0] = 255
        hull = cv2.convexHull(cv2.findNonZero(component))
        cv2.fillConvexPoly(envelopes[y:y+height, x:x+width], hull, 255)
    if not np.any(anchors):
        return recovered, metadata

    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    b, g, r = cv2.split(image.astype(np.float32))
    exg = 2*g-r-b
    chroma = (g-np.maximum(r, b)) / np.maximum(g+r+b, 1)
    foliage = (((hue >= 28) & (hue <= 96) & (sat >= 55) & (val >= 12)
                & (exg >= 10) & (chroma >= .025))
               | ((hue >= 18) & (hue < 28) & (sat >= 70) & (val >= 35)
                  & (exg >= 35)))
    distance_m = cv2.distanceTransform((anchors == 0).astype(np.uint8), cv2.DIST_L2, 5) * gsd
    pale_interior = ((val >= 170) & (sat <= 100) & (exg >= -8)
                     & (r <= g+20) & (b <= g+30)
                     & (envelopes > 0) & (distance_m <= .75))
    eligible = (foliage & (distance_m <= 1.5)) | pale_interior | (anchors > 0)
    # Join only gaps of about 2 cm, so attached leaves do not turn into
    # hundreds of isolated canopy specks. This cannot fill a broad mud/shadow
    # gap, and all original AI pixels remain in the returned mask.
    link_radius = max(1, min(7, round(.01 / gsd)))
    linked = cv2.morphologyEx(eligible.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((2*link_radius+1, 2*link_radius+1), np.uint8))
    _, regions = cv2.connectedComponents(linked, 8)
    anchor_groups = np.unique(regions[anchors > 0])
    anchor_groups = anchor_groups[anchor_groups > 0]
    additions = (linked > 0) & (canopy_mask == 0)
    group_anchor_areas = np.bincount(regions[anchors > 0])
    group_added_areas = np.bincount(regions[additions], minlength=len(group_anchor_areas))
    accepted_groups = []
    for group in anchor_groups:
        if group_added_areas[group] > .35 * group_anchor_areas[group]:
            metadata["canopy_crown_recovery_limited_groups"] += 1
        else:
            accepted_groups.append(group)
    added = additions & np.isin(regions, accepted_groups)
    recovered[added] = 255
    metadata["canopy_crown_recovered_pixels"] = int(np.count_nonzero(added))
    metadata["canopy_crown_recovered_area_m2"] = float(np.count_nonzero(added) * gsd**2)
    return recovered, metadata
