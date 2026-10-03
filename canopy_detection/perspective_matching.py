"""Conservative retry for photos whose texture differs from the orthomosaic."""
import cv2
import numpy as np

from .registration import select_registration


def match_perspective_patch(image, patch, offset, nominal_h, nominal_scale, ortho_gsd):
    """Collect structural landmarks, then validate similarity/affine/perspective.

    Root-normalized SIFT and several smoothing levels reduce sensitivity to
    orthomosaic texture and sharpening. The GPS/heading prior only removes
    implausible correspondences; it does not dictate the fitted corner shape.
    Existing held-out, spatial-support and stability gates remain authoritative.
    """
    height, width = image.shape[:2]
    if nominal_scale <= 0 or ortho_gsd <= 0:
        return None, {"registration_validated": False}
    work_scale = min(1., 1400. / max(width * nominal_scale, height * nominal_scale))
    size = (round(width * nominal_scale * work_scale), round(height * nominal_scale * work_scale))
    if min(size) < 32 or patch.size == 0:
        return None, {"registration_validated": False}
    drone = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    reference = cv2.resize(patch, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    drone, reference = [cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) for value in (drone, reference)]
    sift = cv2.SIFT_create(nfeatures=18000, contrastThreshold=.01)
    expected_rotation = np.degrees(np.arctan2(nominal_h[1, 0], nominal_h[0, 0]))
    sources, targets = [], []
    for level, sigma in ((1., 0.), (1., 1.), (1., 2.), (1., 3.), (.75, 0.), (.5, 0.)):
        a, b = drone, reference
        if level != 1.:
            a = cv2.resize(a, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
        if sigma:
            a = cv2.GaussianBlur(a, (0, 0), sigma)
            b = cv2.GaussianBlur(b, (0, 0), sigma)
        keys_a, desc_a = sift.detectAndCompute(a, None)
        keys_b, desc_b = sift.detectAndCompute(b, None)
        if desc_a is None or desc_b is None or len(desc_b) < 2:
            continue
        desc_a = np.sqrt(desc_a / (desc_a.sum(axis=1, keepdims=True) + 1e-7))
        desc_b = np.sqrt(desc_b / (desc_b.sum(axis=1, keepdims=True) + 1e-7))
        pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .8 * pair[1].distance]
        # Repeated crowns can have similar descriptors at unrelated rotations.
        # Use a generous heading window; perspective is still fitted freely.
        matches = [m for m in matches if abs(
            (keys_b[m.trainIdx].angle - keys_a[m.queryIdx].angle - expected_rotation + 180) % 360 - 180
        ) <= 35.]
        if not matches:
            continue
        source = np.float64([keys_a[m.queryIdx].pt for m in matches]) / [a.shape[1]/width, a.shape[0]/height]
        target = np.float64([keys_b[m.trainIdx].pt for m in matches]) / [b.shape[1]/patch.shape[1], b.shape[0]/patch.shape[0]] + offset
        prior = cv2.perspectiveTransform(source.reshape(-1, 1, 2), nominal_h).reshape(-1, 2)
        nearby = np.linalg.norm(target-prior, axis=1) * ortho_gsd <= 6.0
        sources.extend(source[nearby])
        targets.extend(target[nearby])
    matrix, diagnostics = select_registration(sources, targets, width, height, nominal_scale)
    diagnostics['perspective_retry_matches'] = len(sources)
    return matrix, diagnostics
