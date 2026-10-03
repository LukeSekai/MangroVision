"""Leaf evidence checks for supplemental seedlings and yellow-leaf recovery."""
import cv2
import numpy as np


def filter_seedling_ground_artifacts(image, entries, gsd):
    """Require local leaf evidence for every supplemental candidate.

    Color alone admits bright wood tips as well as weak shadow fringes. Verify
    connected leaf cores, including yellow leaves supported by a nearby green
    core, then check small ambiguous cores for surrounding linear wood. Larger
    leaf clusters and strongly green leaves can protect a real plant's stem.
    Distances and support areas use GSD rather than a fixed image resolution.
    """
    if not entries or gsd is None or not np.isfinite(gsd) or gsd <= 0:
        return list(entries), {"seedling_ground_artifact_rejected_count": 0}
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    b, g, r = cv2.split(image.astype(np.float32))
    exg = 2 * g - r - b
    chroma = (g - np.maximum(r, b)) / np.maximum(g + r + b, 1)
    radius = max(3, int(np.ceil(.40 / gsd)))
    retained = []
    rejected_centers = []
    rejected_reasons = {}
    min_core_pixels = max(3, int(np.ceil(.0005 / gsd**2)))
    height, width = image.shape[:2]
    for entry in entries:
        cx, cy = entry["center"]
        x, y = int(round(cx)), int(round(cy))
        x0, x1 = max(0, x-radius), min(width, x+radius+1)
        y0, y1 = max(0, y-radius), min(height, y+radius+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance_m2 = ((xx-cx)**2 + (yy-cy)**2) * gsd**2
        ring = (distance_m2 >= .16**2) & (distance_m2 <= .40**2)
        patch_sat, patch_val = sat[y0:y1, x0:x1], val[y0:y1, x0:x1]
        patch_exg, patch_chroma = exg[y0:y1, x0:x1], chroma[y0:y1, x0:x1]
        patch_hue = hue[y0:y1, x0:x1]
        ground_val = float(np.median(patch_val[ring] if np.any(ring) else patch_val))
        broad_leaf = ((distance_m2 <= .25**2)
                      & (patch_hue >= 23) & (patch_hue <= 72)
                      & (patch_sat >= 55) & (patch_exg >= 35)
                      & ((patch_val >= ground_val+8) | (patch_chroma >= .08)))
        green_core = (broad_leaf & (patch_hue >= 28)
                      & (g[y0:y1, x0:x1] >= r[y0:y1, x0:x1]-2))
        leaf_pixels, _ = _compact_leaf_pixels(green_core, gsd)
        leaf_area = int(np.count_nonzero(leaf_pixels))
        strong_leaf = _has_compact_leaf(leaf_pixels & (patch_chroma >= .08), gsd)
        _, broad_areas = _compact_leaf_pixels(broad_leaf, gsd)
        # Several yellow leaves can belong to one seedling; a single pale
        # highlight must not borrow unrelated color elsewhere in the image.
        if len(broad_areas) >= 2 and leaf_area >= min_core_pixels:
            leaf_area = max(leaf_area, sum(broad_areas))

        reason = None
        if not strong_leaf and leaf_area < np.ceil(.002 / gsd**2):
            reason = "insufficient_leaf_support"
            # Sunlit leaves can have almost-white centers. Grow a compact
            # pale region only when it has substantial, independently green
            # pixels. Neutral cream wood and a few colored edge pixels cannot
            # provide that support.
            pale = ((distance_m2 <= .25**2) & (patch_hue >= 28) & (patch_hue <= 72)
                    & (patch_sat >= 22) & (patch_exg >= 12)
                    & (g[y0:y1, x0:x1] >= r[y0:y1, x0:x1]-2)
                    & (patch_val >= ground_val+8))
            green_anchor = ((patch_hue >= 33) & (patch_hue <= 72)
                            & (patch_sat >= 35) & (patch_exg >= 25)
                            & (g[y0:y1, x0:x1] >= r[y0:y1, x0:x1]+3)
                            & (patch_val >= ground_val+8))
            pale_pixels, _ = _compact_leaf_pixels(pale, gsd)
            count, pale_labels = cv2.connectedComponents(pale_pixels.astype(np.uint8), 8)
            supported_pale = np.zeros_like(pale_pixels)
            for label in range(1, count):
                component = pale_labels == label
                if np.count_nonzero(component & green_anchor) >= max(3, np.ceil(.001 / gsd**2)):
                    supported_pale |= component
            pale_area = int(np.count_nonzero(supported_pale))
            if pale_area >= np.ceil(.002 / gsd**2):
                leaf_pixels = supported_pale
                leaf_area = pale_area
                reason = None
        if not strong_leaf and reason is None:
            bright = ((patch_val >= ground_val+12) & (patch_sat >= 20)
                      & (patch_hue >= 15) & (patch_hue <= 75)).astype(np.uint8)
            bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            count, bright_labels = cv2.connectedComponents(bright, 8)
            if leaf_area < np.ceil(.006 / gsd**2):
                for label in range(1, count):
                    component = bright_labels == label
                    if np.count_nonzero(component & leaf_pixels) >= leaf_area * .5:
                        if (_principal_aspect(component) > 3.5
                                and _principal_aspect(component & leaf_pixels) > 1.8):
                            reason = "elongated_wood_highlight"
                            break
            if (reason is None and leaf_area < np.ceil(.0032 / gsd**2)
                    and len(broad_areas) < 2):
                lines = cv2.HoughLinesP(
                    bright * 255, 1, np.pi / 180,
                    threshold=max(10, round(.18 / gsd)),
                    minLineLength=max(1, round(.25 / gsd)),
                    maxLineGap=max(1, round(.03 / gsd)),
                )
                if lines is not None:
                    for ax, ay, bx, by in lines[:, 0]:
                        dx, dy = float(bx-ax), float(by-ay)
                        length2 = dx*dx + dy*dy
                        if length2 == 0:
                            continue
                        ox, oy = cx-x0, cy-y0
                        t = np.clip(((ox-ax)*dx + (oy-ay)*dy) / length2, 0, 1)
                        distance = np.hypot(ox-(ax+t*dx), oy-(ay+t*dy)) * gsd
                        if distance < .04:
                            reason = "linear_wood_context"
                            break
        if reason:
            rejected_centers.append([float(cx), float(cy)])
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
        else:
            retained.append(entry)
    return retained, {
        "seedling_ground_artifact_rejected_count": len(rejected_centers),
        "seedling_ground_artifact_rejected_centers": rejected_centers,
        "seedling_ground_artifact_rejected_reasons": rejected_reasons,
    }


def _compact_leaf_pixels(mask, gsd):
    """Require a connected leaf core, rather than a few wood-colored pixels.

    Do not close/dilate this mask: doing so joins scattered stick highlights
    into artificial leaves. Minimum physical area scales with the GSD.
    Strong green cores can protect shaded leaves without a brightness gain.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    accepted = np.zeros(mask.shape, dtype=bool)
    areas = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < max(3, int(np.ceil(.0005 / gsd**2))):
            continue
        component = labels[y:y+height, x:x+width] == label
        if _principal_aspect(component) > 3.5:
            continue
        accepted[y:y+height, x:x+width] |= component
        areas.append(int(area))
    return accepted, areas


def _has_compact_leaf(mask, gsd):
    return bool(_compact_leaf_pixels(mask, gsd)[1])


def _principal_aspect(mask):
    y, x = np.nonzero(mask)
    if len(x) < 3:
        return float('inf')
    eigenvalues = np.linalg.eigvalsh(np.cov(np.array([x, y])))
    return float(np.sqrt((eigenvalues[-1] + 0.25) / (eigenvalues[0] + 0.25)))


def filter_small_canopy_ground_artifacts(image, canopy_mask, gsd):
    """Validate small AI crowns against local leaf evidence before buffering.

    Olive ground patches can pass the model's broad vegetation color gate.
    For isolated crowns up to 1.2 m², require a compact bright or strongly
    green leaf core inside the actual prediction. Long, thin predictions also
    need substantial compact leaf support to avoid accepting colored wood.
    Larger crowns retain the
    existing mature-canopy checks. This is a conservative appearance gate,
    not an algae species classifier.
    """
    metadata = {"canopy_ground_artifact_rejected_count": 0,
                "canopy_ground_artifact_rejected_centers": [],
                "canopy_ground_artifact_rejected_reasons": {}}
    if gsd is None or not np.isfinite(gsd) or gsd <= 0:
        return canopy_mask.copy(), metadata
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        (canopy_mask > 0).astype(np.uint8), 8,
    )
    filtered = canopy_mask.copy()
    if count <= 1 or not np.any(stats[1:, cv2.CC_STAT_AREA] * gsd**2 <= 1.2):
        return filtered, metadata
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    b, g, r = cv2.split(image.astype(np.float32))
    exg = 2 * g - r - b
    chroma = (g - np.maximum(r, b)) / np.maximum(g + r + b, 1)
    radius = max(3, int(np.ceil(.25 / gsd)))
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area * gsd**2 > 1.2:
            continue
        # An edge fragment may belong to a larger crown beyond this photo.
        # Its size and surrounding ground cannot be established here.
        if x == 0 or y == 0 or x+w == image.shape[1] or y+h == image.shape[0]:
            continue
        x0, x1 = max(0, x-radius), min(image.shape[1], x+w+radius)
        y0, y1 = max(0, y-radius), min(image.shape[0], y+h+radius)
        sl = np.s_[y0:y1, x0:x1]
        component = labels[sl] == label
        ground = labels[sl] == 0
        if np.count_nonzero(ground) < 24:
            continue
        ground_value = float(np.median(val[sl][ground]))
        leaves = (component & (hue[sl] >= 23) & (hue[sl] <= 72)
                  & (sat[sl] >= 55) & (exg[sl] >= 35)
                  & ((val[sl] >= ground_value + 8) | (chroma[sl] >= .08)))
        compact_leaves, leaf_areas = _compact_leaf_pixels(leaves, gsd)
        reason = None
        if not leaf_areas:
            reason = "insufficient_leaf_support"
        elif _is_thin_woody_prediction(component, compact_leaves, gsd):
            reason = "elongated_wood_prediction"
        if reason:
            patch = filtered[sl]
            patch[component] = 0
            metadata["canopy_ground_artifact_rejected_centers"].append(
                centers[label].tolist()
            )
            reasons = metadata["canopy_ground_artifact_rejected_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
    metadata["canopy_ground_artifact_rejected_count"] = len(
        metadata["canopy_ground_artifact_rejected_centers"]
    )
    return filtered, metadata


def _is_thin_woody_prediction(component, compact_leaves, gsd):
    """Reject pole-shaped crowns supported only by small colored flecks.

    Require all shape conditions together, in ground units. Long leafy branches
    with a wider crown or substantial compact leaf support remain eligible.
    This check applies before canopy expansion/merging and danger buffering.
    """
    if _principal_aspect(component) < 5.0:
        return False
    y, x = np.nonzero(component)
    _, dimensions, _ = cv2.minAreaRect(np.column_stack((x, y)).astype(np.float32))
    length, width = sorted(dimensions, reverse=True)
    if length * gsd < .60 or width * gsd > .25:
        return False
    return np.count_nonzero(compact_leaves) < .10 * len(x)


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
    min_area, max_area = max(8, int(np.ceil(.0005 / gsd**2))), max(8, round(.15 / gsd**2))
    max_bbox = max(3, round(.65 / gsd))
    min_thickness = max(1.5, .012 / gsd)
    min_leaf_span = max(3, int(np.ceil(.025 / gsd)))
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
        if (object_exg < 40
                or object_sat - np.median(sat[y0:y1, x0:x1][ring]) < 25
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
        if not votes.size:
            continue
        bright_component = bright_labels == np.bincount(votes).argmax()
        # A real leaf cluster can connect to a thin bright stem. Reject a
        # long highlight only when the compact leaf core is a minor part of
        # it, instead of discarding the whole seedling with its stem.
        leaf_fraction = area / max(1, np.count_nonzero(bright_component))
        if _principal_aspect(bright_component) > 2.5 and leaf_fraction < .40:
            continue
        entry = {'center': (float(cx), float(cy)), 'is_micro': False,
                 'contextual_only': False, 'cluster_rescue': False,
                 'yellow_recovery': True, 'water_context': False}
        extra.append(entry)
        occupied.append(entry['center'])
    metadata['seedling_yellow_recovery_count'] = len(extra)
    return extra, metadata
