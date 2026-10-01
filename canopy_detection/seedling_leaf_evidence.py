"""Conservative recovery of compact yellow leaves missed by green-only pixels."""
import cv2
import numpy as np


def _principal_aspect(mask):
    y, x = np.nonzero(mask)
    if len(x) < 3:
        return float('inf')
    eigenvalues = np.linalg.eigvalsh(np.cov(np.array([x, y])))
    return float(np.sqrt((eigenvalues[-1] + 0.25) / (eigenvalues[0] + 0.25)))


def recover_yellow_leaf_clusters(image, excluded_mask, centers, gsd):
    """Return extra centers only for yellow-dominated vegetation with leaf shape.

    The caller must first establish independent green-leaf evidence in the
    image. Yellow wood alone must never enable this recovery. A green-dominated
    image retains its existing detection path. Shapes are measured in metres;
    principal-axis aspect rejects diagonal sticks as well as upright ones.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    exg = 2 * g - r - b
    chroma = (g - np.maximum(r, b)) / np.maximum(g + r + b, 1)
    yellow = ((hue >= 23) & (hue <= 65) & (sat >= 60)
              & (val >= 60) & (exg >= 30)).astype(np.uint8)
    pixel_count = int(np.count_nonzero(yellow))
    green_fraction = float(np.count_nonzero((yellow > 0) & (chroma >= .012))) / max(pixel_count, 1)
    metadata = {'seedling_yellow_green_fraction': green_fraction,
                'seedling_yellow_recovery_count': 0}
    if not pixel_count or green_fraction >= .5:
        return [], metadata

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, 8)
    radius = max(3, round(.27 / gsd))
    min_area, max_area = max(8, round(.0035 / gsd**2)), max(8, round(.15 / gsd**2))
    max_bbox = max(3, round(.65 / gsd))
    min_thickness = max(2., .03 / gsd)
    min_leaf_span = max(3, int(np.ceil(.09 / gsd)))
    duplicate_distance = max(2., .20 / gsd)
    occupied = list(centers)
    extra = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if (not min_area <= area <= max_area or max(width, height) > max_bbox
                or min(width, height) < min_leaf_span):
            continue
        component = labels[y:y+height, x:x+width] == label
        if np.any(component & (excluded_mask[y:y+height, x:x+width] > 0)):
            continue
        cx, cy = centroids[label]
        if any((cx-ox)**2 + (cy-oy)**2 <= duplicate_distance**2 for ox, oy in occupied):
            continue
        if _principal_aspect(component) > 2.5:
            continue
        thickness = cv2.distanceTransform(np.pad(component.astype(np.uint8), 1), cv2.DIST_L2, 5).max()
        if thickness < min_thickness:
            continue
        x0, x1 = max(0, x-radius), min(image.shape[1], x+width+radius)
        y0, y1 = max(0, y-radius), min(image.shape[0], y+height+radius)
        ring = yellow[y0:y1, x0:x1] == 0
        if np.count_nonzero(ring) < 24:
            continue
        object_sat = float(np.mean(sat[y:y+height, x:x+width][component]))
        object_exg = float(np.mean(exg[y:y+height, x:x+width][component]))
        if (object_sat - np.median(sat[y0:y1, x0:x1][ring]) < 25
                or object_exg - np.median(exg[y0:y1, x0:x1][ring]) < 25):
            continue

        # A rounded yellow tip may belong to a long pale stick. Include its
        # surrounding bright material before making the final shape decision.
        patch_hue, patch_val = hue[y0:y1, x0:x1], val[y0:y1, x0:x1]
        bright = ((patch_val >= np.median(patch_val[ring]) + 12)
                  & (sat[y0:y1, x0:x1] >= 20)
                  & (patch_hue >= 15) & (patch_hue <= 65)).astype(np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, bright_labels = cv2.connectedComponents(bright, 8)
        votes = bright_labels[y-y0:y-y0+height, x-x0:x-x0+width][component]
        votes = votes[votes > 0]
        if not votes.size or _principal_aspect(bright_labels == np.bincount(votes).argmax()) > 2.5:
            continue
        entry = {'center': (float(cx), float(cy)), 'is_micro': False,
                 'contextual_only': False, 'cluster_rescue': False,
                 'yellow_recovery': True, 'water_context': False}
        extra.append(entry)
        occupied.append(entry['center'])
    metadata['seedling_yellow_recovery_count'] = len(extra)
    return extra, metadata
