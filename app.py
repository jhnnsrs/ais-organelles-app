"""Segment the AIS and the cisternal organelle inside it, along a drawn PATH.

You trace a PATH along an axon and run **Segment AIS tube along path** on it, naming the AIS
channel (AnkyrinG, NaV) and the organelle channel (synaptopodin, ...; still called the
"cluster" channel in parameters and tables). The numerics are ``src/ais_organelles_app/core.py``:
a tube of voxels round the path, the AIS as its windowed-Otsu mask, the organelles as a
noise-floored threshold inside it.

It answers the adjusted path (rewritten in place on a re-run, found by its lineage), the
AIS surface, the organelles (one mesh object per organelle; its ``object_id`` is the table's
``cluster_id``), a table with one row per organelle and one with one row per arclength
bin. Every surface derives from the AIS lens and names the path as a parent.

The two channels need not share a grid: the organelle lens is read at the same world points
through the drawn space (``frame_ais.to_drawn`` then ``frame_cluster.to_index``). Only the
box round the path is read from each lens.

Run:  arkitekt run dev
Test: uv run pytest     # the numerics on synthetic volumes, and the action against a stubbed mikro
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

import numpy as np
import xarray as xr
from arkitekt import App, Description, Task, run
from mikro import Mikro, mikro_service
from mikro.api.schema import (
    Annotation,
    AnnotationDerivedFromInput,
    AxisType,
    ColumnInput,
    DimensionPartInput,
    MeshCollectionIdentifiesInput,
    MeshLens,
    PartKind,
    TableDataset,
    ValueRelation,
)
from mikro.arkitekt.specs import PathAnnotation, SingleChannelVolume
from mikro.arkitekt.widgets import overlapping

from ais_organelles_app import core
from ais_organelles_app.centreline import resample_path, simplify, smooth
from logos import declare_logos, logo_of

from ais_organelles_app.placement import (
    _SIMPLIFY,
    PathFrame,
    lens_label,
    lens_volume,
    path_frame,
    pinned_of,
    refuse_outside,
    spacing_of,
    write_refined_path,
)

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.0.1"

SPACE = ("z", "y", "x")
XYZ = ("x", "y", "z")

app = App(
    "ais-organelles-app",
    __version__,
    description="The AIS and the cisternal organelle inside it, segmented along a drawn path",
    author="jhnnsrs",
    services=[mikro_service],
)
declare_logos(app)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def open_channel(mikro: Mikro, task: Task, path: Any, lens: Any) -> tuple[np.ndarray, PathFrame, xr.DataArray]:
    """The path's vertices in `lens`' voxels, the frame between them, and the lens' volume, lazily."""
    if sorted(lens.axes_of_type(AxisType.SPACE)) != sorted(SPACE):
        raise ValueError(f"'{lens_label(lens)}' needs the space axes {list(SPACE)}.")
    _, calibrated = spacing_of(lens)
    if not calibrated:
        raise ValueError(
            f"'{lens_label(lens)}' is not calibrated: the diameter, the distances and the volumes are in µm, so "
            f"every space axis {list(SPACE)} needs a voxel size. Calibrate the image first."
        )
    vertices, frame = path_frame(mikro, path, lens, SPACE)
    if frame.warning:
        task.log(frame.warning)
    return vertices, frame, lens_volume(lens, SPACE, pinned_of(path, lens))


def crop_box(low: np.ndarray, high: np.ndarray, pad: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[low, high)`` voxel bounds holding the index box ``[low, high]`` plus `pad` and the
    voxel linear interpolation reads past it, inside the image."""
    start = np.maximum(np.floor(low).astype(int) - pad - 1, 0)
    stop = np.minimum(np.ceil(high).astype(int) + pad + 2, shape)
    return start, stop


def corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """The eight corners of the box ``[low, high]``: what any affine map of it is bounded by."""
    return np.array([[(low, high)[bit][axis] for axis, bit in enumerate(bits)] for bits in np.ndindex(2, 2, 2)], dtype=np.float64)


def read_blurred(data: xr.DataArray, low: np.ndarray, high: np.ndarray, spacing: np.ndarray, sigma_um: float) -> np.ndarray:
    """``[low, high)`` of the lazy volume, blurred: only the store chunks it touches are fetched."""
    crop = data.isel({axis: slice(int(lo), int(hi)) for axis, lo, hi in zip(SPACE, low, high)}).to_numpy()
    return smooth(crop, spacing, sigma_um)


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #


def to_lens_xyz(meshes: dict[int, core.Mesh], low: np.ndarray, shape: np.ndarray) -> dict[int, core.Mesh]:
    """Crop-local (z, y, x) index vertices -> lens voxels with (x, y, z) components.

    The reversal is a reflection and turns the marching-cubes winding outward. Clipped
    into the volume, since the octree wants non-negative cells.
    """
    extent = np.asarray(shape[::-1], dtype=np.float64) - 1.0
    out = {}
    for object_id, (verts, faces) in meshes.items():
        xyz = (np.asarray(verts, dtype=np.float64) + low)[:, ::-1]
        if core.signed_volume(xyz, np.asarray(faces)) <= 0:
            raise AssertionError(f"object {object_id} is inside out")
        out[int(object_id)] = (np.clip(xyz, 0.0, extent), np.asarray(faces))
    return out


def upload_surfaces(mikro: Mikro, lens: Any, path: Any, surfaces: dict[int, core.Mesh], name: str) -> Any:
    """One mesh collection in `lens`' voxels: BY_DIMENSION by axis name, and the path as a parent."""
    from fabriks import build_collection

    return mikro.create_mesh_collection(
        name=name,
        # `axes` names the vertex slots: left out, the viewer guesses and transposes.
        store=build_collection(surfaces, levels=2, axes=XYZ),
        axes=list(XYZ),
        derived_from=[
            lens.derive(
                kind="BY_DIMENSION",
                parts=[DimensionPartInput(kind=PartKind.IDENTITY, input_axes=XYZ, output_axes=XYZ)],
                value_relation=ValueRelation.TRANSFORMED,
            ),
            AnnotationDerivedFromInput(annotation=path.id, value_relation=ValueRelation.TRANSFORMED),
        ],
    )


#: Each column's long name and the power of µm it is in.
_CLUSTER_LONG = {
    "arclength_um": ("arclength along the adjusted path", 1),
    "length_um": ("extent along the path", 1),
    "volume_um3": ("volume", 3),
    "mean_intensity": ("mean organelle signal", 0),
    "max_intensity": ("max organelle signal", 0),
    "centroid_z_um": ("centroid z (image)", 1),
    "centroid_y_um": ("centroid y (image)", 1),
    "centroid_x_um": ("centroid x (image)", 1),
    "radial_um": ("distance from the adjusted path", 1),
    "depth_um": ("distance inside the AIS edge", 1),
}
_BIN_LONG = {
    "arclength_um": ("bin centre arclength", 1),
    "ais_volume_um3": ("AIS volume in the bin", 3),
    "ais_radius_um": ("AIS equivalent radius", 1),
    "cluster_volume_um3": ("organelle volume in the bin", 3),
    "cluster_fraction": ("organelle fraction of the AIS volume", 0),
    "n_clusters": ("organelles (by centre)", 0),
    "density_per_um": ("organelles per length", -1),
}


def _columns(index: str, long: dict[str, tuple[str, int]], identified_by=None) -> list[ColumnInput]:
    unit = lambda power: None if power == 0 else "micrometer" if power == 1 else f"micrometer**{power}"
    # Unset, not None: the field is `[IdentificationInput!]! = []`, and an explicit None is sent as null.
    extra = {"identified_by": identified_by} if identified_by else {}
    columns = [ColumnInput(name=index, axis_type=AxisType.INDEX, long_name=index.replace("_", " "), **extra)]
    return columns + [ColumnInput(name=name, long_name=text, unit=unit(power)) for name, (text, power) in long.items()]


# --------------------------------------------------------------------------- #
# The model the action answers
# --------------------------------------------------------------------------- #


@app.model
@dataclass
class AisClusterRow:
    """One organelle, rounded for reading."""

    cluster: int
    label: str
    arclength: float
    volume: float
    depth: float
    intensity: float


@app.model
@dataclass
class AisTubeRun:
    """What one run found: the AIS along the adjusted path and the organelles inside it, in path order."""

    path: str
    length: float
    mean_radius: float
    count: int
    density_per_length: float
    ais_threshold: float
    threshold: float
    cluster_background: float
    cluster_noise: float
    cluster_threshold_source: str
    ais_channel: str = ""
    cluster_channel: str = ""
    clusters: list[AisClusterRow] = field(default_factory=list)


def tube_run(result: core.Result, name: str) -> AisTubeRun:
    s = result.summary
    return AisTubeRun(
        path=name,
        length=round(s["length_um"], 2),
        mean_radius=round(s["mean_radius_um"], 3),
        count=result.n_clusters,
        density_per_length=round(s["density_per_um"], 3),
        ais_threshold=round(s["ais_threshold"], 4),
        threshold=round(result.threshold, 4),
        cluster_background=round(s["organelle_background"], 4),
        cluster_noise=round(s["organelle_noise"], 4),
        cluster_threshold_source=result.threshold_source,
        clusters=[
            AisClusterRow(
                cluster=int(row.cluster_id),
                label=f"#{int(row.cluster_id)} at {row.arclength_um:.1f}",
                arclength=round(float(row.arclength_um), 2),
                volume=round(float(row.volume_um3), 3),
                depth=round(float(row.depth_um), 3),
                intensity=round(float(row.mean_intensity), 1),
            )
            for row in result.clusters.itertuples()
        ],
    )


# --------------------------------------------------------------------------- #
# The action
# --------------------------------------------------------------------------- #


@app.action(collections=["segment"], logo=logo_of("segment_ais_tube_along_path"))
def segment_ais_tube_along_path(
    path: Annotated[PathAnnotation, Description("The PATH along the axon")],
    ais_lens: Annotated[SingleChannelVolume, overlapping("path"), Description("The AIS channel (AnkyrinG, NaV)")],
    cluster_lens: Annotated[SingleChannelVolume, overlapping("path"), Description("The organelle channel (synaptopodin, ...)")],
    diameter_um: Annotated[float, Description("The tube round the path that holds the AIS, in µm")] = 3.0,
    threshold_scale: Annotated[float, Description("Scales the AIS threshold: below 1 grows the AIS, above 1 shrinks it")] = 1.0,
    cluster_threshold_scale: Annotated[float, Description("Scales the organelle threshold's height above the diffuse level inside the AIS: below 1 grows the organelles, above 1 shrinks them")] = 1.0,
    max_same_distance_um: Annotated[float, Description("Organelle pieces no further apart than this are one organelle; 0: only touching pieces")] = 0.5,
    min_volume_um3: Annotated[float, Description("Drop organelles smaller than this")] = 0.1,
    *,
    mikro: Mikro,
    task: Task,
) -> tuple[
    Annotated[Annotation, Description("The path adjusted onto the AIS")],
    Annotated[MeshLens, Description("The AIS surface")],
    Annotated[MeshLens | None, Description("The organelles, one mesh object per cluster_id; None when none were found")],
    Annotated[TableDataset | None, Description("One row per organelle; None when none were found")],
    Annotated[TableDataset, Description("One row per 1 µm of the adjusted path")],
    Annotated[AisTubeRun, Description("The AIS and the organelle inside it, in path order")],
]:
    """Segment AIS Tube Along Path

    Takes the voxels within diameter_um / 2 of the path, the AIS as the AIS channel above
    its Otsu (per 5 µm along the path, for a fading stain) where it touches the path, and
    the organelles as the organelle channel inside the AIS above its Otsu, but never below
    the diffuse level there plus 3 noise σ (so an AIS without an organelle yields none).
    Pieces within max_same_distance_um are one organelle. Reads only the box round the path
    """
    params = core.Params(
        diameter_um=diameter_um, ais_scale=threshold_scale, organelle_scale=cluster_threshold_scale,
        max_same_distance_um=max_same_distance_um, min_volume_um3=min_volume_um3,
    )
    name = path.name or f"path {path.id}"
    vertices, frame_a, data_a = open_channel(mikro, task, path, ais_lens)
    _, frame_c, data_c = open_channel(mikro, task, path, cluster_lens)
    shape_a = np.array([data_a.sizes[a] for a in SPACE])
    shape_c = np.array([data_c.sizes[a] for a in SPACE])
    refuse_outside(vertices, shape_a, path)
    spacing_a, spacing_c = spacing_of(ais_lens)[0], spacing_of(cluster_lens)[0]

    # The box round the path's spline (which can bow out past the vertices) out to the tube,
    # plus half a voxel diagonal (the path is rounded to voxels), then each blur's footprint.
    drawn = resample_path(vertices, spacing_a, float(spacing_a.min())) / spacing_a
    reach = (0.5 * diameter_um + float(spacing_a.max()) + 0.5 * float(np.linalg.norm(spacing_a))) / spacing_a
    read_low = np.minimum(drawn.min(axis=0), vertices.min(axis=0)) - reach
    read_high = np.maximum(drawn.max(axis=0), vertices.max(axis=0)) + reach
    blur = lambda spacing: (3.0 * params.sigma_um / spacing + 0.5).astype(int)
    low_a, high_a = crop_box(read_low, read_high, blur(spacing_a), shape_a)
    on_c = frame_c.to_index(frame_a.to_drawn(corners(read_low, read_high)))
    low_c, high_c = crop_box(on_c.min(axis=0), on_c.max(axis=0), blur(spacing_c), shape_c)

    task.progress(10, f"Reading round '{name}'")
    field_a = read_blurred(data_a, low_a, high_a, spacing_a, params.sigma_um)
    field_c = read_blurred(data_c, low_c, high_c, spacing_c, params.sigma_um)

    def cluster_index(points: np.ndarray) -> np.ndarray:
        return frame_c.to_index(frame_a.to_drawn(points / spacing_a + low_a)) - low_c

    task.progress(30, f"Segmenting along '{name}'")
    result = core.segment(field_a, field_c, vertices - low_a, spacing_a, params, cluster_index, low_um=low_a * spacing_a)
    s = result.summary
    task.log(
        f"AIS {s['length_um']:.1f} µm long, mean radius {s['mean_radius_um']:.2f} µm, threshold {s['ais_threshold']:.4g}; "
        f"{result.n_clusters} organelles, {s['cluster_volume_um3']:.3f} µm³ ({100 * s['cluster_fraction']:.1f}% of the AIS); "
        f"path adjusted up to {s['largest_adjust_um']:.2f} µm"
    )
    b, noise = s["organelle_background"], s["organelle_noise"]
    why = (
        f"the noise floor: Otsu {s['organelle_otsu']:.4g} fell within {params.noise_k:g} σ of the diffuse level"
        if result.threshold_source == "noise floor"
        else f"Otsu, above the noise floor {s['organelle_floor']:.4g}"
    )
    task.log(
        f"Organelle threshold {result.threshold:.4g} ({s['organelle_snr']:.1f} σ above the diffuse level "
        f"{b:.4g} ± {noise:.3g}) from {why}; {int(s['cluster_unseeded'])} pieces dropped below {params.seed_k:g} σ."
    )

    adjusted = simplify(result.line / spacing_a, spacing_a, _SIMPLIFY * float(spacing_a.min())) + low_a
    adjusted_path = write_refined_path(
        mikro, path, frame_a.to_drawn(adjusted), "Adjusted path",
        reason="moved onto the AIS (segment_ais_tube_along_path)", made_by="segment_ais_tube_along_path",
    )

    task.progress(60, "Uploading the surfaces")
    # Plain names: which path and lenses a collection comes from is its lineage.
    ais_mesh = upload_surfaces(mikro, ais_lens, path, to_lens_xyz({1: result.ais_mesh}, low_a, shape_a), "AIS")
    cluster_mesh = None
    if result.meshes:
        cluster_mesh = upload_surfaces(
            mikro, ais_lens, path, to_lens_xyz(result.meshes, low_a, shape_a), "Organelles"
        )

    task.progress(85, "Uploading the tables")
    settings = (
        f"diameter {diameter_um:g} µm, AIS Otsu ×{threshold_scale:g}, organelle threshold ×{cluster_threshold_scale:g} "
        f"above the diffuse level, pieces within {max_same_distance_um:g} µm one organelle"
    )
    # Measured along the path on both lenses: lineage without geometry (UNMAPPABLE), so the
    # plain names need not say which path or images a table is of.
    parent = [
        AnnotationDerivedFromInput(annotation=path.id, value_relation=ValueRelation.TRANSFORMED),
        *(
            lens.derive(kind="UNMAPPABLE", value_relation=ValueRelation.TRANSFORMED, reason="measured along the path")
            for lens in (ais_lens, cluster_lens)
        ),
    ]
    cluster_table = None
    if cluster_mesh is not None:
        identified = [
            MeshCollectionIdentifiesInput(kind="MESH_COLLECTION", mesh_collection=cluster_mesh.id, name="mesh object_id -> cluster_id")
        ]
        cluster_table = mikro.create_table_dataset(
            name="Organelles",
            data=result.clusters.reset_index(drop=True),
            columns=_columns("cluster_id", _CLUSTER_LONG, identified),
            description=(
                f"One row per organelle of {lens_label(cluster_lens)} inside the AIS of {lens_label(ais_lens)} along "
                f"'{name}' (adjusted), {settings}, threshold {result.threshold:.4g}. Centroids in the whole AIS image."
            ),
            derived_from=parent,
        )
    profile_table = mikro.create_table_dataset(
        name="AIS Profile",
        data=result.bins.reset_index(drop=True),
        columns=_columns("bin", _BIN_LONG),
        description=(
            f"Per {params.bin_um:g} µm of the adjusted '{name}': the AIS of {lens_label(ais_lens)} and the organelle of "
            f"{lens_label(cluster_lens)} inside it, {settings}. " + "; ".join(f"{k} = {v:.3g}" for k, v in s.items())
        ),
        derived_from=parent,
    )

    task.progress(100, f"{result.n_clusters} organelles on the AIS along '{name}'")
    found = tube_run(result, name)
    found.ais_channel, found.cluster_channel = lens_label(ais_lens), lens_label(cluster_lens)
    return adjusted_path, ais_mesh.lens(), cluster_mesh.lens() if cluster_mesh else None, cluster_table, profile_table, found


if __name__ == "__main__":
    run(app)
