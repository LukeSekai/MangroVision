"""Select a camera-to-orthophoto transform using spatially held-out landmarks."""

from typing import Any

import cv2
import numpy as np


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, np.float64).reshape(-1, 1, 2), matrix,
    ).reshape(-1, 2)


def _fit_once(kind: str, source: np.ndarray, target: np.ndarray) -> Any:
    if len(source) < 8:
        return None
    if kind == "projective":
        matrix, _ = cv2.findHomography(
            source, target, cv2.RANSAC, 3.0, maxIters=10000, confidence=0.999,
        )
        return matrix
    fit = cv2.estimateAffine2D if kind == "affine" else cv2.estimateAffinePartial2D
    matrix, _ = fit(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=10000, confidence=0.999, refineIters=20,
    )
    return None if matrix is None else np.vstack((matrix, [0.0, 0.0, 1.0]))


def _fit(kind, source, target, cells):
    # Several planes can coexist: roof features may form a larger consensus
    # than the ground. Evaluate subsequent consensuses spatially as well.
    remaining = np.arange(len(source))
    best, best_score = None, float("inf")
    for _ in range(3):
        matrix = _fit_once(kind, source[remaining], target[remaining])
        if matrix is None:
            break
        errors = np.linalg.norm(_project(source, matrix) - target, axis=1)
        score = float(np.mean([np.mean(np.minimum(errors[cells == cell], 12.0))
                               for cell in np.unique(cells)]))
        if score < best_score:
            best, best_score = matrix, score
        keep = errors[remaining] > 3.0
        if np.all(keep):
            break
        remaining = remaining[keep]
    return best


def _geometry_valid(matrix, width, height, nominal_scale):
    """Reject folds, horizon crossings, extreme perspective and scale changes."""
    if matrix is None or not np.all(np.isfinite(matrix)):
        return False
    grid = np.float64([(x, y) for y in (0, height / 2, height)
                      for x in (0, width / 2, width)])
    denominator = grid @ matrix[2, :2] + matrix[2, 2]
    if np.any(denominator <= 0) or np.ptp(denominator) > .5 * np.median(denominator):
        return False
    projected = _project(grid, matrix)
    step_x = _project(grid + [1, 0], matrix) - projected
    step_y = _project(grid + [0, 1], matrix) - projected
    for dx, dy in zip(step_x, step_y):
        jacobian = np.column_stack((dx, dy))
        scales = np.linalg.svd(jacobian, compute_uv=False)
        if (np.linalg.det(jacobian) <= 0 or scales[0] / scales[-1] > 1.4
                or scales[-1] < .65 * nominal_scale or scales[0] > 1.45 * nominal_scale):
            return False
    return True


def select_registration(source, target, width, height, nominal_scale):
    """Return a validated visual transform, or None when support is too local.

    Fit on a bounded sample per image cell and reserve different landmarks in
    each cell for validation. A detailed roof cannot dominate the score merely
    by supplying hundreds of similar corners. Extra affine/perspective freedom
    must improve the held-out score over the simpler model.
    """
    source = np.asarray(source, np.float64).reshape(-1, 2)
    target = np.asarray(target, np.float64).reshape(-1, 2)
    finite = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source, target = source[finite], target[finite]
    if len(source) < 40 or width <= 0 or height <= 0 or nominal_scale <= 0:
        return None, {"registration_validated": False}
    # SIFT can emit multiple orientations for the same landmark. Keep one
    # observation per source/target pixel at orthophoto resolution so copies
    # of a training landmark cannot masquerade as independent validation.
    order = np.lexsort((target[:, 1], target[:, 0], source[:, 1], source[:, 0]))
    source, target = source[order], target[order]
    _, unique = np.unique(np.round(source * nominal_scale), axis=0, return_index=True)
    source, target = source[unique], target[unique]
    _, unique = np.unique(np.round(target), axis=0, return_index=True)
    source, target = source[unique], target[unique]
    if len(source) < 40:
        return None, {"registration_validated": False}
    cell_x = np.clip((source[:, 0] * 4 / width).astype(int), 0, 3)
    cell_y = np.clip((source[:, 1] * 4 / height).astype(int), 0, 3)
    cells = cell_y * 4 + cell_x
    training, validation, balanced = [], [], []
    rng = np.random.default_rng(42)
    for cell in np.unique(cells):
        indices = np.flatnonzero(cells == cell)
        # Fix ordering before shuffling so OpenCV's keypoint enumeration cannot
        # decide which observations become validation data.
        indices = indices[np.lexsort((target[indices, 1], target[indices, 0],
                                     source[indices, 1], source[indices, 0]))]
        indices = rng.permutation(indices)
        held = indices[::4] if len(indices) >= 4 else np.array([], dtype=int)
        fit = indices[~np.isin(indices, held)][:40]
        training.extend(fit.tolist())
        validation.extend(held[:20].tolist())
        balanced.extend(indices[:60].tolist())
    training, validation, balanced = map(np.asarray, (training, validation, balanced))
    if len(validation) < 16 or len(np.unique(cells[validation])) < 5:
        return None, {"registration_validated": False}

    def support(indices):
        points = source[indices]
        return {
            "inliers": int(len(points)),
            "hull_ratio": float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))) / (width * height) if len(points) >= 3 else 0.,
            "span_x_ratio": float(np.ptp(points[:, 0])) / width if len(points) else 0.,
            "span_y_ratio": float(np.ptp(points[:, 1])) / height if len(points) else 0.,
            "grid_cells": int(len(np.unique(cells[indices]))),
        }

    def broad(metrics):
        return (metrics["hull_ratio"] >= .20 and metrics["span_x_ratio"] >= .5
                and metrics["span_y_ratio"] >= .5 and metrics["grid_cells"] >= 6)

    candidates = []
    diagnostics = {}
    sample_grid = np.float64([(x, y) for y in (0, height / 2, height)
                             for x in (0, width / 2, width)])
    for kind in ("similarity", "affine", "projective"):
        held_matrix = _fit(kind, source[training], target[training], cells[training])
        matrix = _fit(kind, source[balanced], target[balanced], cells[balanced])
        if not all(_geometry_valid(m, width, height, nominal_scale) for m in (held_matrix, matrix)):
            diagnostics[kind] = {"valid": False, "reason": "geometry"}
            continue
        errors = np.linalg.norm(_project(source, matrix) - target, axis=1)
        held_errors = np.linalg.norm(_project(source[validation], held_matrix) - target[validation], axis=1)
        inliers = np.flatnonzero(errors <= 5.0)
        evidence = support(inliers)
        held_inliers = held_errors <= 5.0
        held_evidence = support(validation[held_inliers])
        # Equal weight for every occupied validation cell, with bounded
        # contributions from bad descriptor matches.
        cell_scores = [float(np.mean(np.minimum(held_errors[cells[validation] == cell], 12.0)))
                       for cell in np.unique(cells[validation])]
        score = float(np.mean(cell_scores))
        stability = float(np.max(np.linalg.norm(_project(sample_grid, matrix) - _project(sample_grid, held_matrix), axis=1)))
        valid = (len(inliers) >= 40 and len(inliers) / len(source) >= .25
                 and broad(evidence) and np.count_nonzero(held_inliers) >= 12
                 and float(np.mean(held_inliers)) >= .25
                 and broad(held_evidence)
                 and stability <= 15.0)
        stats = {**evidence, "valid": bool(valid), "validation_score_px": score,
                 "validation_inliers": int(np.count_nonzero(held_inliers)),
                 "validation_hull_ratio": held_evidence["hull_ratio"],
                 "validation_grid_cells": held_evidence["grid_cells"],
                 "stability_px": stability,
                 "median_error_px": float(np.median(errors[inliers])) if len(inliers) else None}
        diagnostics[kind] = stats
        if valid:
            candidates.append((kind, matrix, stats))

    if not candidates:
        return None, {"registration_validated": False, "registration_candidates": diagnostics}
    chosen = candidates[0]
    for candidate in candidates[1:]:
        if candidate[2]["validation_score_px"] < chosen[2]["validation_score_px"] * .90:
            chosen = candidate
    kind, matrix, stats = chosen
    return matrix, {"registration_validated": True, "registration_model": kind,
                    "registration_candidates": diagnostics, **{
                        f"registration_{key}": value for key, value in stats.items()
                    }}
