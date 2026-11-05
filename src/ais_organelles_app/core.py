"""Segment the AIS and the cisternal organelle inside it, along a drawn path: the numerics, no server.

Everything works on voxel masks of the AIS crop, nothing is straightened:

    tube         ->  the voxels within diameter / 2 of the drawn path, each with its nearest
                     path sample (its row along the path)
    AIS          ->  Otsu on the AIS channel in the tube, per arclength window (AnkG fades
                     distally); the parts above it that touch the path, holes filled
    adjusted     ->  the path through each row's AIS centroid, smoothed: what the
                     measurements run along
    organelle    ->  Otsu on the organelle channel inside the AIS, floored at the diffuse
                     level b + noise_k·σ of the AIS core (Otsu always splits: with no
                     organelle it splits the noise). Pieces must peak at b + seed_k·σ;
                     pieces within `max_same_distance_um` are one organelle
    measure      ->  a row per organelle, a row per arclength bin

Geometry is in the AIS crop's local coordinates, ``(z, y, x)``: physical (µm) where it
says so, index otherwise. The self-checks (`_checks`, on `synthetic` volumes) run as
``tests/test_core.py``. In code the organelle is still called the "cluster".
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt, find_objects, gaussian_filter1d, label, map_coordinates
from scipy.ndimage import maximum as label_maximum
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes

from ais_organelles_app.centreline import arclength, resample_path, smooth

#: Physical crop-local points (N, 3) -> indices (N, 3) of the organelle field.
ToIndex = Callable[[np.ndarray], np.ndarray]
Mesh = tuple[np.ndarray, np.ndarray]

#: The noise floor's reference: the AIS core, voxels within this fraction of their row's
#: widest AIS voxel from the adjusted path (the blurred edge's ramp would read as noise),
#: past neither end. Fewer than `_MIN_REFERENCE`: the whole AIS.
_CORE = 0.5
_MIN_REFERENCE = 100
#: How smooth the adjusted path is along itself.
_ADJUST_SMOOTH_UM = 1.0


@dataclass(frozen=True)
class Params:
    diameter_um: float = 3.0  # the tube round the drawn path that holds the AIS
    sigma_um: float = 0.15  # the blur both channels get first
    window_um: float = 5.0  # the AIS Otsu's arclength window
    ais_scale: float = 1.0  # multiplies the AIS threshold
    organelle_scale: float = 1.0  # scales the organelle threshold's height above b
    noise_k: float = 3.0  # the organelle threshold's floor, in σ above b
    seed_k: float = 5.0  # an organelle must peak this many σ above b
    max_same_distance_um: float = 0.5  # pieces closer than this are one organelle
    min_volume_um3: float = 0.1
    bin_um: float = 1.0


@dataclass
class Result:
    drawn: np.ndarray  # (m, 3) the drawn path, physical
    line: np.ndarray  # (m, 3) the adjusted path, physical
    ais_threshold: np.ndarray  # (m,) t(s)
    ais_mask: np.ndarray  # crop-shaped bool
    ais_mesh: Mesh  # crop INDEX vertices, (z, y, x)
    labels: np.ndarray  # crop-shaped organelle ids, 1..n in path order
    meshes: dict[int, Mesh]  # one per organelle id
    threshold: float  # the organelle threshold
    threshold_source: str  # "otsu" or "noise floor"
    clusters: pd.DataFrame
    bins: pd.DataFrame
    summary: dict[str, float]

    @property
    def n_clusters(self) -> int:
        return len(self.meshes)


def _otsu(values: np.ndarray) -> float:
    lo, hi = float(values.min()), float(values.max())
    return float(threshold_otsu(values)) if hi > lo else lo


def otsu_along(values: np.ndarray, rows: np.ndarray, m: int, step: float, window_um: float, scale: float) -> np.ndarray:
    """``(m,)`` Otsu threshold t(s): one per half window over a window of rows, interpolated
    and smoothed along s. A window with too few values takes its neighbours'."""
    half = max(1, int(round(0.5 * window_um / step)))
    centres = np.unique(np.r_[np.arange(0, m, half), m - 1])
    order = np.argsort(rows, kind="stable")
    ordered = rows[order]
    found = np.full(len(centres), np.nan)
    for i, c in enumerate(centres):
        lo, hi = np.searchsorted(ordered, c - half, "left"), np.searchsorted(ordered, c + half, "right")
        if hi - lo >= 16:
            found[i] = _otsu(values[order[lo:hi]])
    good = np.isfinite(found)
    if not good.any():
        return np.full(m, scale * _otsu(values))
    t = np.interp(np.arange(m), centres[good], found[good])
    return scale * gaussian_filter1d(t, sigma=0.5 * half, mode="nearest")


def noise_floor(values: np.ndarray) -> tuple[float, float]:
    """``(b, σ)``: the histogram's mode, and the MAD (×1.4826) of the values below it.
    Organelles only add an upper tail, so they move neither (the median and the plain MAD
    both climb with them)."""
    lo, hi = np.percentile(values, [0.5, 99.5])
    if not hi > lo:
        return float(lo), 1e-6 * max(abs(float(lo)), 1.0)
    counts, edges = np.histogram(values, bins=128, range=(lo, hi))
    i = int(np.argmax(gaussian_filter1d(counts.astype(np.float64), 2.0)))
    b = float(0.5 * (edges[i] + edges[i + 1]))
    below = b - values[values < b]
    return b, max(1.4826 * float(np.median(below)) if len(below) else 0.0, 1e-6 * max(abs(b), 1.0))


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = (verts[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def surface(level: np.ndarray, mask: np.ndarray) -> Mesh:
    """Marching cubes of `level` with its sign forced to `mask`: subvoxel where the level
    crosses zero, on the voxel where the mask decided (a fill, the tube's edge)."""
    found = find_objects(mask.astype(np.uint8))[0]
    box = tuple(slice(max(b.start - 1, 0), min(b.stop + 1, n)) for b, n in zip(found, mask.shape))
    eps = 1e-6 * max(float(np.abs(level[box]).max()), 1.0)
    part = np.where(mask[box], np.maximum(level[box], eps), np.minimum(level[box], -eps))
    verts, faces, _, _ = marching_cubes(np.pad(part, 1, constant_values=-eps), level=0.0)
    return verts + (np.array([b.start for b in box], dtype=np.float64) - 1.0), faces


def _keep_touching(above: np.ndarray, path: np.ndarray) -> np.ndarray:
    """The components of `above` the path runs through; none: the largest."""
    pieces, n = label(above)
    if not n:
        raise ValueError("The AIS channel stays below its threshold along the whole path.")
    touched = np.unique(pieces[path])
    touched = touched[touched > 0]
    if not len(touched):
        touched = [int(np.argmax(np.bincount(pieces.ravel())[1:])) + 1]
    return np.isin(pieces, touched)


def _fill(mask: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Holes filled, with the tube's ends sealed by `caps` while filling: a ring-like stain
    (dim centre) is a hollow tube open at its ends, which a plain fill leaves hollow."""
    return mask | (binary_fill_holes(mask | caps) & ~caps)


def _adjusted(points: np.ndarray, rows: np.ndarray, m: int, step: float) -> np.ndarray:
    """``(m, 3)``: each row's centroid, rows without voxels interpolated, smoothed along s."""
    counts = np.bincount(rows, minlength=m)
    have = np.flatnonzero(counts)
    centre = np.column_stack([np.bincount(rows, weights=points[:, i], minlength=m)[have] / counts[have] for i in range(3)])
    s = np.arange(m)
    line = np.column_stack([np.interp(s, have, centre[:, i]) for i in range(3)])
    return gaussian_filter1d(line, sigma=_ADJUST_SMOOTH_UM / step, axis=0, mode="nearest")


def segment(
    ais_field: np.ndarray,
    organelle_field: np.ndarray,
    vertices: np.ndarray,
    spacing: np.ndarray,
    params: Params = Params(),
    organelle_to_index: ToIndex | None = None,
    low_um: np.ndarray | None = None,
) -> Result:
    """The AIS and its organelles from two blurred fields. `vertices` are AIS crop indices;
    `organelle_to_index` takes AIS crop physical points to the organelle field's indices
    (left out, both share the crop's grid); `low_um` is the crop's origin, for centroids."""
    spacing = np.asarray(spacing, dtype=np.float64)
    shape, step, voxel = ais_field.shape, float(spacing.min()), float(np.prod(spacing))
    drawn = resample_path(vertices, spacing, step)
    m = len(drawn)

    # The tube, and each of its voxels' row along the drawn path.
    on_path = np.rint(drawn / spacing).astype(int)
    on_path = on_path[np.all((on_path >= 0) & (on_path < shape), axis=1)]
    if not len(on_path):
        raise ValueError("The path leaves the AIS crop.")
    path = np.zeros(shape, dtype=bool)
    path[tuple(on_path.T)] = True
    tube = distance_transform_edt(~path, sampling=spacing) <= 0.5 * params.diameter_um
    points = np.argwhere(tube) * spacing
    _, rows = cKDTree(drawn).query(points)
    to_index = organelle_to_index or (lambda p: p / spacing)
    organelle = map_coordinates(organelle_field, to_index(points).T, order=1, mode="nearest").astype(np.float32)

    # The AIS: above its windowed Otsu, touching the path, filled.
    ais = ais_field[tube]
    t_ais = otsu_along(ais, rows, m, step, params.window_um, params.ais_scale)
    above = np.zeros(shape, dtype=bool)
    above[tube] = ais >= t_ais[rows]
    caps = np.zeros(shape, dtype=bool)
    caps[tube] = (rows == 0) | (rows == m - 1)
    ais_mask = _fill(_keep_touching(above, path), caps) & tube
    inside = ais_mask[tube]
    ais_level = np.full(shape, -1.0, dtype=np.float32)
    ais_level[tube] = ais - t_ais[rows]

    # The adjusted path; from here on rows and arclength are along it.
    line = _adjusted(points[inside], rows[inside], m, step)
    radial, rows = cKDTree(line).query(points)
    s = arclength(line)

    # The organelle threshold: Otsu inside the AIS, floored at the AIS core's diffuse level.
    depth = distance_transform_edt(ais_mask, sampling=spacing)[tube]
    values = organelle[inside]
    widest = np.zeros(m)
    np.maximum.at(widest, rows[inside], radial[inside])
    core = (radial[inside] <= _CORE * widest[rows[inside]]) & (rows[inside] > 0) & (rows[inside] < m - 1)
    reference = values[core] if core.sum() >= _MIN_REFERENCE else values
    b, sigma = noise_floor(reference)
    otsu, floor = _otsu(values), b + params.noise_k * sigma
    source = "otsu" if otsu >= floor else "noise floor"
    t = b + params.organelle_scale * (max(otsu, floor) - b)

    # The organelles: above t inside the AIS, filled, seeded, grouped, big enough, in path order.
    field = np.zeros(shape, dtype=np.float32)
    field[tube] = organelle
    mask = np.zeros(shape, dtype=bool)
    mask[tube] = inside & (organelle >= t)
    mask = binary_fill_holes(mask) & ais_mask
    pieces, n = label(mask)
    found = n
    if params.seed_k > 0 and n:
        peaks = np.asarray(label_maximum(field, pieces, np.arange(1, n + 1)))
        mask = np.r_[False, peaks >= b + params.seed_k * sigma][pieces]
        pieces, n = label(mask)
    n_pieces = n
    if params.max_same_distance_um > 0 and n > 1:
        groups, n = label(distance_transform_edt(~mask, sampling=spacing) <= 0.5 * params.max_same_distance_um)
        pieces = np.where(mask, groups, 0)
    on = pieces[tube]
    counts = np.bincount(on, minlength=n + 1)
    big = np.flatnonzero(counts * voxel >= params.min_volume_um3)
    big = big[big > 0]
    middle = np.bincount(on, weights=s[rows], minlength=n + 1) / np.maximum(counts, 1)
    lookup = np.zeros(n + 1, dtype=np.uint32)
    lookup[big[np.argsort(middle[big], kind="stable")]] = np.arange(1, len(big) + 1, dtype=np.uint32)
    labels, on = lookup[pieces], lookup[on]

    meshes = {i: surface(field - t, labels == i) for i in range(1, len(big) + 1)}
    low = np.zeros(3) if low_um is None else np.asarray(low_um, dtype=np.float64)
    clusters = _organelle_rows(on, s[rows], organelle, points + low, radial, depth, t, voxel)
    bins, summary = _bins(s, rows, inside, on, clusters, voxel, params.bin_um)
    summary.update(
        ais_threshold=float(t_ais.mean()),
        ais_threshold_range=float(np.ptp(t_ais)),
        threshold=t,
        organelle_background=b,
        organelle_noise=sigma,
        organelle_otsu=otsu,
        organelle_floor=floor,
        organelle_snr=(t - b) / sigma,
        cluster_unseeded=float(found - n_pieces),
        cluster_pieces=float(n_pieces),
        largest_adjust_um=float(np.linalg.norm(line - drawn, axis=1).max()),
    )
    return Result(
        drawn, line, t_ais, ais_mask, surface(ais_level, ais_mask), labels, meshes, t, source, clusters, bins, summary
    )


CLUSTER_COLUMNS = [
    "cluster_id", "arclength_um", "length_um", "volume_um3", "mean_intensity", "max_intensity",
    "centroid_z_um", "centroid_y_um", "centroid_x_um", "radial_um", "depth_um",
]
BIN_COLUMNS = [
    "bin", "arclength_um", "ais_volume_um3", "ais_radius_um", "cluster_volume_um3", "cluster_fraction",
    "n_clusters", "density_per_um",
]


def _organelle_rows(
    on: np.ndarray, s: np.ndarray, values: np.ndarray, points: np.ndarray, radial: np.ndarray, depth: np.ndarray,
    t: float, voxel: float,
) -> pd.DataFrame:
    """One row per organelle; its arclength weighted by how far each voxel stands above t."""
    rows = []
    for cluster_id in range(1, int(on.max(initial=0)) + 1):
        pick = on == cluster_id
        weight = np.clip(values[pick] - t, 1e-12, None)
        centroid = points[pick].mean(axis=0)
        rows.append({
            "cluster_id": cluster_id,
            "arclength_um": float(np.average(s[pick], weights=weight)),
            "length_um": float(np.ptp(s[pick])),
            "volume_um3": float(pick.sum() * voxel),
            "mean_intensity": float(values[pick].mean()),
            "max_intensity": float(values[pick].max()),
            "centroid_z_um": float(centroid[0]),
            "centroid_y_um": float(centroid[1]),
            "centroid_x_um": float(centroid[2]),
            "radial_um": float(np.average(radial[pick], weights=weight)),
            "depth_um": float(np.average(depth[pick], weights=weight)),
        })
    return pd.DataFrame(rows, columns=CLUSTER_COLUMNS)


def _bins(
    s: np.ndarray, rows: np.ndarray, inside: np.ndarray, on: np.ndarray, clusters: pd.DataFrame, voxel: float,
    bin_um: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per `bin_um` of the adjusted path. The rows at either end (which also hold the
    voxels past it) are left out of the volumes, so the end bins' radii are not inflated."""
    m, length = len(s), float(s[-1])
    n_bins = max(1, int(np.ceil(length / bin_um)))
    bin_of = np.minimum((s / bin_um).astype(int), n_bins - 1)
    alongside = (rows > 0) & (rows < m - 1)
    voxel_bin = bin_of[rows]
    ais = np.bincount(voxel_bin[inside & alongside], minlength=n_bins) * voxel
    organelle = np.bincount(voxel_bin[(on > 0) & alongside], minlength=n_bins) * voxel
    centres = clusters["arclength_um"].to_numpy()
    out = []
    for b in range(n_bins):
        start, end = b * bin_um, min((b + 1) * bin_um, length)
        span = max(end - start, 1e-12)
        count = int(((centres >= start) & ((centres <= end) if b == n_bins - 1 else (centres < end))).sum())
        out.append({
            "bin": b,
            "arclength_um": 0.5 * (start + end),
            "ais_volume_um3": float(ais[b]),
            "ais_radius_um": float(np.sqrt(ais[b] / (np.pi * span))),
            "cluster_volume_um3": float(organelle[b]),
            "cluster_fraction": float(organelle[b] / ais[b]) if ais[b] > 0 else 0.0,
            "n_clusters": count,
            "density_per_um": count / span,
        })
    total_ais, total_organelle = float(ais.sum()), float(organelle.sum())
    summary = {
        "length_um": length,
        "mean_radius_um": float(np.sqrt(total_ais / (np.pi * max(length, 1e-12)))),
        "clusters": float(len(clusters)),
        "density_per_um": len(clusters) / max(length, 1e-12),
        "ais_volume_um3": float(inside.sum() * voxel),
        "cluster_volume_um3": total_organelle,
        "cluster_fraction": total_organelle / total_ais if total_ais > 0 else 0.0,
    }
    return pd.DataFrame(out, columns=BIN_COLUMNS), summary


def run(
    ais_volume: np.ndarray, organelle_volume: np.ndarray, vertices: np.ndarray, spacing: np.ndarray,
    params: Params = Params(),
) -> Result:
    """The whole recipe on two raw volumes sharing one grid (the self-checks' case)."""
    blur = lambda v: smooth(v, spacing, params.sigma_um)
    return segment(blur(ais_volume), blur(organelle_volume), vertices, spacing, params)


# --------------------------------------------------------------------------- #
# Synthetic self-check
# --------------------------------------------------------------------------- #


@dataclass
class Synthetic:
    ais: np.ndarray  # (z, y, x) float32
    cluster: np.ndarray  # the organelle channel
    spacing: np.ndarray
    rough: np.ndarray  # the drawn path, indices
    centre: np.ndarray  # truth centreline every 0.02 µm, physical
    radius: np.ndarray  # truth AIS radius per `centre` sample, µm
    organelles: np.ndarray  # (n, 3) truth organelle centres, µm; none without contrast
    pieces: int  # how many separate pieces the organelles are in
    neighbour: np.ndarray | None  # the bright neighbour's centre, µm


def synthetic(
    seed: int = 0, fade: float = 0.0, neighbour: bool = False, organelle_contrast: float = 60.0, diffuse: float = 10.0
) -> Synthetic:
    """A curved AIS tapering 0.9 -> 0.5 µm. Inside it, on the organelle channel, a 0.2 µm
    thick tube 0.2 µm off the centre in three stretches 1.3 µm apart; each stretch is two
    pieces with a 0.4 µm break. Path drawn 0.3 µm off.

    `fade` dims the AIS signal linearly to (1 - fade) at its end; `neighbour` puts a blob
    brighter than the AIS 0.35 µm past its edge mid-way. `organelle_contrast` is the
    organelles' brightness above the axoplasm (0: none), `diffuse` the axoplasm's."""
    rng = np.random.default_rng(seed)
    shape, spacing = (30, 110, 140), np.array([0.3, 0.1, 0.1])
    u = np.linspace(0.0, 1.0, 800)
    curve = np.column_stack([14.0 + 2.0 * u, 55.0 + 15.0 * np.sin(1.2 * np.pi * u), 15.0 + 110.0 * u])
    centre = resample_path(curve, spacing, 0.02)
    s = arclength(centre)
    tangent = np.gradient(centre, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.array([1.0, 0.0, 0.0]) - tangent[:, :1] * tangent
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    binormal = np.cross(tangent, normal)
    radius = 0.9 - 0.4 * s / s[-1]
    grid = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1).reshape(-1, 3) * spacing
    distance, nearest = cKDTree(centre).query(grid)
    r_here = radius[nearest]
    inside = 1.0 / (1.0 + np.exp(-(r_here - distance) / 0.05))
    dim = 1.0 - fade * s[nearest] / s[-1]
    ais = 10.0 + 60.0 * dim * inside * (1.0 + 0.5 * np.clip(1.0 - (distance / r_here) ** 2, 0.0, None))
    blob = None
    if neighbour:
        mid = len(centre) // 2
        blob = centre[mid] + (radius[mid] + 0.6) * binormal[mid]
        ais = ais + 120.0 / (1.0 + np.exp(-(0.25 - np.linalg.norm(grid - blob, axis=1)) / 0.05))

    off = lambda rows, theta: centre[rows] + 0.2 * (np.cos(theta) * normal[rows] + np.sin(theta) * binormal[rows])
    axis_points, organelles = [], []
    for middle, theta in zip(np.array([0.2, 0.5, 0.8]) * s[-1], (0.5, 2.5, 4.5)):
        # Centre lines end 0.8 µm apart; their 0.2 µm round ends leave a 0.4 µm break.
        for lo, hi in ((middle - 1.0, middle - 0.4), (middle + 0.4, middle + 1.0)):
            axis_points.append(off(np.flatnonzero((s >= lo) & (s <= hi)), theta))
        organelles.append(off(np.array([np.searchsorted(s, middle)]), theta)[0])
    to_organelle, _ = cKDTree(np.concatenate(axis_points)).query(grid)
    organelle = 1.0 / (1.0 + np.exp(-(0.2 - to_organelle) / 0.04))
    cluster = 5.0 + diffuse * inside + organelle_contrast * organelle
    ais = ais.reshape(shape) + rng.normal(0.0, 1.5, shape)
    cluster = cluster.reshape(shape) + rng.normal(0.0, 1.5, shape)
    pick = np.linspace(0, len(curve) - 1, 10).astype(int)
    rough = curve[pick] + np.array([0.0, 3.0, 0.0]) + rng.normal(0.0, [0.1, 0.4, 0.4], (10, 3))
    some = organelle_contrast > 0
    return Synthetic(
        ais.astype(np.float32), cluster.astype(np.float32), spacing, rough, centre, radius,
        np.array(organelles) if some else np.empty((0, 3)), len(axis_points) if some else 0, blob,
    )


def _checks(seed: int = 0) -> Iterator[tuple[str, bool, str]]:
    synth = synthetic(seed)
    result = run(synth.ais, synth.cluster, synth.rough, synth.spacing)
    truth = cKDTree(synth.centre)
    off_drawn, _ = truth.query(result.drawn)
    off, _ = truth.query(result.line)
    middle = slice(int(0.05 * len(off)), int(0.95 * len(off)))
    yield (
        "the adjusted path lies on the AIS",
        bool(np.median(off[middle]) < 0.1),
        f"median {np.median(off[middle]):.3f} µm off (drawn: {np.median(off_drawn):.2f})",
    )
    step = float(np.linalg.norm(np.diff(synth.centre, axis=0), axis=1).mean())
    r = synth.radius
    truth_volume = float((np.pi * r**2).sum() * step + 2.0 / 3.0 * np.pi * (r[0] ** 3 + r[-1] ** 3))
    volume = result.summary["ais_volume_um3"]
    yield "the AIS has the right volume", abs(volume / truth_volume - 1.0) < 0.15, f"{volume:.2f} µm³ (truth {truth_volume:.2f})"

    def radius_error(found: Result) -> np.ndarray:
        """Per inner bin, |found / truth − 1| of the AIS radius."""
        _, at = truth.query(found.line)
        s = arclength(found.line)
        inner = found.bins.iloc[1:-1]
        want = np.array([synth.radius[at[np.argmin(np.abs(s - c))]] for c in inner["arclength_um"]])
        return np.abs(inner["ais_radius_um"].to_numpy() / want - 1.0)

    error = radius_error(result)
    yield "the AIS radius per bin", bool(np.median(error) < 0.1), f"median error {100 * np.median(error):.0f}%"
    hard = synthetic(seed, fade=0.6, neighbour=True)
    found = run(hard.ais, hard.cluster, hard.rough, hard.spacing)
    error = radius_error(found)
    tail = error[int(0.75 * len(error)) :]
    blob = tuple(np.rint(hard.neighbour / hard.spacing).astype(int))
    yield (
        "a fading AIS beside a brighter neighbour keeps its radius, without the neighbour",
        bool(np.median(error) < 0.1 and np.median(tail) < 0.15) and not found.ais_mask[blob],
        f"median error {100 * np.median(error):.0f}%, distal {100 * np.median(tail):.0f}%, neighbour in: {found.ais_mask[blob]}",
    )
    n = len(synth.organelles)
    ungrouped = run(synth.ais, synth.cluster, synth.rough, synth.spacing, Params(max_same_distance_um=0.0, min_volume_um3=0.02))
    yield (
        f"{n} organelles at the defaults (pieces 0.4 µm apart count as one), {synth.pieces} pieces ungrouped",
        result.n_clusters == n and ungrouped.n_clusters == synth.pieces,
        f"{result.n_clusters} at the defaults, {ungrouped.n_clusters} at max_same_distance 0",
    )
    if len(result.clusters):
        centres = result.clusters[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy()
        worst = float(cKDTree(centres).query(synth.organelles)[0].max())
        depth = result.clusters["depth_um"].to_numpy()
        yield (
            "each organelle is centred within 0.3 µm, inside the AIS",
            worst < 0.3 and bool((depth > 0).all()),
            f"worst {worst:.2f} µm, depths {np.round(depth, 2).tolist()}",
        )
    total = int(result.bins["n_clusters"].sum())
    yield "the bins count every organelle once", total == result.n_clusters, f"{total} in bins"
    yield (
        "with bright organelles, Otsu (not the noise floor) sets the organelle threshold",
        result.threshold_source == "otsu",
        f"Otsu {result.summary['organelle_otsu']:.2f}, floor {result.summary['organelle_floor']:.2f}",
    )
    for name, diffuse in (("diffuse axoplasm and noise, no organelle", 10.0), ("flat noise", 0.0)):
        empty = synthetic(seed, organelle_contrast=0.0, diffuse=diffuse)
        found = run(empty.ais, empty.cluster, empty.rough, empty.spacing)
        yield (
            f"{name}: no organelle is found",
            found.n_clusters == 0,
            f"{found.n_clusters} found at {found.threshold:.2f} (b {found.summary['organelle_background']:.2f} "
            f"± {found.summary['organelle_noise']:.3f}, {int(found.summary['cluster_unseeded'])} pieces below the seed)",
        )
    # 5 above the axoplasm is ~3x the raw noise (1.5). Thresholded this close, an organelle's
    # two halves shrink apart and may stay two pieces: what counts is that each is found,
    # and nothing else is.
    faint = synthetic(seed, organelle_contrast=5.0)
    found = run(faint.ais, faint.cluster, faint.rough, faint.spacing)
    centres = found.clusters[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy()
    hit = len(centres) > 0 and float(cKDTree(centres).query(faint.organelles)[0].max()) < 1.0
    stray = len(centres) > 0 and float(cKDTree(faint.organelles).query(centres)[0].max()) >= 1.0
    yield (
        f"faint organelles (contrast 5, noise 1.5): all {n} found, nothing else",
        hit and not stray,
        f"{found.n_clusters} pieces, threshold {found.summary['organelle_snr']:.1f} σ above b",
    )
