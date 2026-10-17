"""Bounded image-based refinement when discrete landmarks cannot be matched.

This is an approximate fallback, not a replacement for validated SIFT fits.
Water is excluded from the fitting mask so glint and different tide textures
cannot dominate the correction. No detection masks or planting rules change.
"""
import cv2
import numpy as np

from .registration import _geometry_valid


def _land_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image.astype(np.int16))
    selected = ((hsv[:, :, 0] >= 24) & (hsv[:, :, 0] <= 95)
                & (hsv[:, :, 1] >= 42) & (hsv[:, :, 2] >= 35)
                & (2 * green - red - blue >= 18))
    return cv2.morphologyEx(selected.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))


def _project(points, matrix):
    return cv2.perspectiveTransform(np.float64(points).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def _correlation(first, second, mask):
    a, b = first[mask].astype(np.float64), second[mask].astype(np.float64)
    if len(a) < 128 or min(float(a.std()), float(b.std())) < .015:
        return None
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1.e-12))


def refine_photometric_alignment(image, reference, nominal_h, ortho_gsd_xy):
    """Return an affine correction only after geometry and image checks pass.

    Two smoothing levels, initialized from a coarse fit, must agree at all corners.
    Checks use fixed photo-side regions, including per-cell comparisons against
    the uncorrected projection. Fits cannot turn, resize or shift freely to
    chase an unrelated shoreline. The caller must retain approximate status.
    """
    rejected = {"photometric_refined": False}
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8
            or not isinstance(reference, np.ndarray) or reference.ndim != 3
            or reference.shape[2] != 3 or min(reference.shape[:2]) < 64
            or reference.dtype != np.uint8
            or not isinstance(nominal_h, np.ndarray) or nominal_h.shape != (3, 3)
            or not np.all(np.isfinite(nominal_h))):
        return None, rejected
    height, width = image.shape[:2]
    gsd_xy = np.asarray(ortho_gsd_xy, dtype=np.float64)
    if (min(height, width) < 64 or gsd_xy.shape != (2,)
            or not np.all(np.isfinite(gsd_xy)) or np.any(gsd_xy <= 0)):
        return None, rejected
    scale = min(1., 700. / max(width, height))
    size = (round(width * scale), round(height * scale))
    if min(size) < 32:
        return None, rejected
    resize = np.diag([size[0] / width, size[1] / height, 1.])
    try:
        inverse = np.linalg.inv(nominal_h)
    except np.linalg.LinAlgError:
        return None, rejected
    photo = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    prior = cv2.warpPerspective(reference, resize @ inverse, size)
    # Leave a margin around vegetation for trunks and bank texture. Only the
    # alignment matcher sees these masks; seedling/canopy detection is untouched.
    kernel = np.ones((25, 25), np.uint8)
    fit_mask = cv2.dilate(_land_mask(prior), kernel)
    photo_mask = cv2.dilate(_land_mask(photo), kernel) > 0
    if (np.count_nonzero(fit_mask) < 1000 or np.count_nonzero(photo_mask) < 1000
            or np.count_nonzero(fit_mask) / fit_mask.size > .85):
        return None, rejected
    gray_a = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    gray_b = cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    probes = np.float64([(0, 0), (width, 0), (width, height), (0, height), (width / 2, height / 2)])
    nominal_points = _project(probes, nominal_h)
    nominal_scale = float(np.sqrt(abs(np.linalg.det(nominal_h[:2, :2]))))
    fits = []
    seed = np.eye(2, 3, dtype=np.float32)
    for sigma in (6., 1., 3.):
        a, b = [cv2.GaussianBlur(frame, (0, 0), sigma) for frame in (gray_a, gray_b)]
        try:
            coefficient, affine = cv2.findTransformECC(
                a, b, seed.copy(), cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1.e-6), fit_mask, 5,
            )
        except cv2.error:
            return None, {**rejected, "photometric_rejection": "optimizer"}
        matrix = nominal_h @ np.linalg.inv(resize) @ np.vstack((affine, [0., 0., 1.])) @ resize
        if not _geometry_valid(matrix, width, height, nominal_scale):
            return None, {**rejected, "photometric_rejection": "geometry", "sigma": sigma,
                          "coefficient": float(coefficient), "matrix": matrix.tolist(),
                          "nominal_scale": nominal_scale}
        relative = np.linalg.inv(nominal_h[:2, :2]) @ matrix[:2, :2]
        scales = np.linalg.svd(relative, compute_uv=False)
        angle = abs(np.degrees(np.arctan2(relative[1, 0] - relative[0, 1],
                                           relative[0, 0] + relative[1, 1])))
        points = _project(probes, matrix)
        shifts = np.linalg.norm((points - nominal_points) * gsd_xy, axis=1)
        if (not np.isfinite(coefficient) or coefficient < .40
                or scales.min() < .85 or scales.max() > 1.15 or angle > 8.
                or shifts.max() > 8. or shifts[-1] > 5.5):
            return None, {**rejected, "photometric_rejection": "correction_limits",
                          "coefficient": float(coefficient), "scales": scales.tolist(),
                          "angle": float(angle), "shifts_m": shifts.tolist()}
        if sigma == 6.:
            seed = affine.copy()
        else:
            fits.append((matrix, affine, float(coefficient), points, shifts))
    stability = float(np.max(np.linalg.norm((fits[0][3] - fits[1][3]) * gsd_xy, axis=1)))
    if stability > .75:
        return None, {**rejected, "photometric_stability_m": stability}
    # The less smoothed fit independently corroborates the smoother estimate;
    # use the latter to reduce dependence on orthomosaic sharpening and grain.
    matrix, affine, coefficient, _, shifts = fits[1]
    aligned = cv2.warpAffine(gray_b, affine, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    valid = cv2.warpAffine((np.any(prior > 0, axis=2)).astype(np.uint8), affine, size,
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    mask = photo_mask & valid & np.any(prior > 0, axis=2)
    a, base, corrected = [cv2.GaussianBlur(frame, (0, 0), 3.) for frame in (gray_a, gray_b, aligned)]
    baseline = _correlation(a, base, mask)
    score = _correlation(a, corrected, mask)
    cells = []
    for row in range(4):
        for col in range(4):
            area = np.s_[row * size[1] // 4:(row + 1) * size[1] // 4,
                         col * size[0] // 4:(col + 1) * size[0] // 4]
            before = _correlation(a[area], base[area], mask[area])
            after = _correlation(a[area], corrected[area], mask[area])
            if before is not None and after is not None:
                cells.append((row, col, before, after))
    improved = [cell for cell in cells if cell[3] >= .30 and cell[3] - cell[2] >= .05]
    diagnostics = {"photometric_refined": False, "photometric_score": score,
                   "photometric_baseline_score": baseline,
                   "photometric_fit_score": coefficient,
                   "photometric_stability_m": stability,
                   "photometric_improved_cells": len(improved),
                   "photometric_cells": cells}
    if (score is None or baseline is None or score < .45 or score - baseline < .08
            or len(improved) < 4 or len({c[1] for c in improved}) < 3
            or len({c[0] for c in improved}) < 2):
        return None, diagnostics
    return matrix, {**diagnostics, "photometric_refined": True}
