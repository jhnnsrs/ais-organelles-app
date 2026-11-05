"""The path: arclength-uniform samples through the drawn vertices, the blur, and the simplification back to a shape."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter


def resample_path(vertices: np.ndarray, spacing: np.ndarray, step: float) -> np.ndarray:
    """``(m, 3)`` physical points every `step` along a cubic spline through `vertices` (index units).

    The spline is fitted in physical space, so a step along a coarse axis counts for the
    distance it really covers.
    """
    points = np.asarray(vertices, dtype=np.float64) * spacing
    # A repeated vertex has no direction and makes the spline fit singular.
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9])
    points = points[keep]
    if len(points) < 2:
        raise ValueError("A path needs at least two distinct vertices.")
    chord = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    u = np.linspace(0.0, 1.0, int(max(2000, 8 * chord / step)))
    try:
        curve = np.column_stack(splev(u, splprep(list(points.T), s=0, k=min(3, len(points) - 1))[0]))
    except Exception:
        # The polyline itself, parametrised by chord length.
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        curve = np.column_stack([np.interp(u * along[-1], along, points[:, i]) for i in range(3)])
    length = arclength(curve)
    targets = np.linspace(0.0, length[-1], max(2, int(round(length[-1] / step)) + 1))
    return np.column_stack([np.interp(targets, length, curve[:, i]) for i in range(3)])


def arclength(points: np.ndarray) -> np.ndarray:
    """``(m,)`` cumulative distance along `points`."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def smooth(volume: np.ndarray, spacing: np.ndarray, sigma_um: float) -> np.ndarray:
    """The volume as float32, blurred physically round (anisotropic voxels taken into account)."""
    values = np.asarray(volume, dtype=np.float32)
    if sigma_um <= 0:
        return values
    return gaussian_filter(values, sigma=sigma_um / np.asarray(spacing), mode="nearest", truncate=3.0)


def simplify(points: np.ndarray, spacing: np.ndarray, tolerance: float) -> np.ndarray:
    """Ramer-Douglas-Peucker on index points, with `tolerance` measured physically.

    The path is sampled every finest voxel, hundreds of vertices for an AIS: too many to
    drag around as a shape, and no more accurate than the few dozen within `tolerance`.
    """
    physical = np.asarray(points, dtype=np.float64) * spacing
    keep = np.zeros(len(physical), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(physical) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        chord = physical[last] - physical[first]
        span = np.linalg.norm(chord)
        between = physical[first + 1 : last] - physical[first]
        distance = np.linalg.norm(np.cross(between, chord / span), axis=1) if span > 0 else np.linalg.norm(between, axis=1)
        worst = int(np.argmax(distance))
        if distance[worst] > tolerance:
            keep[first + 1 + worst] = True
            stack += [(first, first + 1 + worst), (first + 1 + worst, last)]
    return np.asarray(points)[keep]
