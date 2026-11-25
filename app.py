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

The panel (the blok ``ais_organelles_workbench``, opened on a z-stack) does the same by
drawing: **Stage Channels** shows the two channels as a scene to trace in, a finished path
runs **Segment AIS Tube Into Scene** (the same run, shown in a 3D scene of its own: both
channels cut to the box round the path, the AIS, the organelles and the adjusted path, one
layer each), and **Export Organelles** writes every traced path's organelles to one CSV file.

Run:  arkitekt run dev
Test: uv run pytest     # the numerics on synthetic volumes, and the action against a stubbed mikro
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import xarray as xr
from arkitekt import App, Description, DescriptorOperator, Key, Provides, Requires, Task, run
from arkitekt.widgets import StructurePicker
from mikro import Mikro
from mikro.api.schema import (
    Annotation,
    AnnotationDerivedFromInput,
    AnnotationKind,
    ArrayDataset,
    AxisType,
    Blending,
    ColumnInput,
    CoordinateInput,
    DimensionPartInput,
    File,
    MeshCollectionIdentifiesInput,
    MeshLens,
    PartKind,
    PreferredView,
    ProjectionMode,
    Scene,
    SliceInput,
    TableDataset,
    TableDatasetExportOfInput,
    TransformationFilter,
    ValueRelation,
)
from mikro.arkitekt.specs import N_CHANNELS, N_SPACE_AXES, PathAnnotation, SingleChannelVolume
from mikro.arkitekt.widgets import overlapping

from ais_organelles_app import core
from ais_organelles_app.centreline import resample_path, simplify, smooth
from logos import declare_logos, logo_of

from ais_organelles_app.placement import (
    _REFINED_COLOR,
    _SIMPLIFY,
    _UREG,
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
__version__ = "0.1.1"

SPACE = ("z", "y", "x")
XYZ = ("x", "y", "z")

app = App(
    "ais-organelles-app",
    __version__,
    description="The AIS and the cisternal organelle inside it, segmented along a drawn path",
    author="jhnnsrs",
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
# The run
# --------------------------------------------------------------------------- #


@dataclass
class Segmented:
    """What one run stored, and where it read: the boxes are ``[low, high)`` voxels of each lens."""

    name: str
    adjusted_path: Any
    ais_mesh: Any
    cluster_mesh: Any | None
    cluster_table: Any | None
    profile_table: Any
    found: AisTubeRun
    ais_box: tuple[np.ndarray, np.ndarray]
    cluster_box: tuple[np.ndarray, np.ndarray]


def segment_along(mikro: Mikro, task: Task, path: Any, ais_lens: Any, cluster_lens: Any, params: core.Params) -> Segmented:
    """Segment the AIS and its organelles along `path` and store them: both actions' run."""
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
    reach = (0.5 * params.diameter_um + float(spacing_a.max()) + 0.5 * float(np.linalg.norm(spacing_a))) / spacing_a
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
        f"diameter {params.diameter_um:g} µm, AIS Otsu ×{params.ais_scale:g}, organelle threshold ×{params.organelle_scale:g} "
        f"above the diffuse level, pieces within {params.max_same_distance_um:g} µm one organelle"
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

    found = tube_run(result, name)
    found.ais_channel, found.cluster_channel = lens_label(ais_lens), lens_label(cluster_lens)
    return Segmented(
        name=name, adjusted_path=adjusted_path, ais_mesh=ais_mesh.lens(), cluster_mesh=cluster_mesh.lens() if cluster_mesh else None,
        cluster_table=cluster_table, profile_table=profile_table, found=found, ais_box=(low_a, high_a), cluster_box=(low_c, high_c),
    )


# --------------------------------------------------------------------------- #
# The actions
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
    out = segment_along(mikro, task, path, ais_lens, cluster_lens, params)
    task.progress(100, f"{out.found.count} organelles on the AIS along '{out.name}'")
    return out.adjusted_path, out.ais_mesh, out.cluster_mesh, out.cluster_table, out.profile_table, out.found


# --------------------------------------------------------------------------- #
# The panel's actions: the two channels as a scene, a run shown in 3D, the export
# --------------------------------------------------------------------------- #

#: The datasets the panel opens on: a z-stack with at least two channels to pick from.
ChannelVolume = Annotated[
    ArrayDataset,
    Requires(key=N_SPACE_AXES, operator=DescriptorOperator.EQUALS, value=3),
    Provides(key=N_SPACE_AXES, operator=DescriptorOperator.EQUALS, value=3),
    Requires(key=N_CHANNELS, operator=DescriptorOperator.GTE, value=2),
    Provides(key=N_CHANNELS, operator=DescriptorOperator.GTE, value=2),
]

#: The dataset search with the port's constraints handed on: mikro's own dataset widget
#: declares no `$requires`, so it would offer every dataset and leave the refusal to the run.
_CHANNEL_VOLUMES = """
query SearchChannelVolumes($search: String, $values: [ID!], $requires: [DescriptorConstraintInput!], $limit: Int, $offset: Int = 0) {
  options: arrayDatasets(
    filters: {search: $search, ids: $values, requires: $requires}
    pagination: {limit: $limit, offset: $offset}
  ) {
    value: id
    label: name
  }
}
"""

#: The lenses of the `dataset` argument that meet the port (`$requires`: one channel, a stack).
_CHANNELS_OF_DATASET = """
query SearchChannelsOfDataset($search: String, $values: [ID!], $requires: [DescriptorConstraintInput!], $dataset: ID, $limit: Int, $offset: Int = 0) {
  options: lenses(
    filters: {kind: ARRAY, search: $search, ids: $values, requires: $requires, dataset: $dataset}
    pagination: {limit: $limit, offset: $offset}
  ) {
    value: id
    label
  }
}
"""

VOLUME_PICKER = StructurePicker(_CHANNEL_VOLUMES, "mikro")
CHANNEL_OF_DATASET = StructurePicker(_CHANNELS_OF_DATASET, "mikro", ["dataset"])

#: How both scenes draw the pair: the AIS green, the organelle magenta, blended additively.
AIS_COLOR = [0, 255, 0, 255]
ORGANELLE_COLOR = [255, 0, 255, 255]

#: The scenes showing a lens, each with what its layers draw.
_SCENES_OF_LENS = """
query ScenesOfLens($lens: ID!) {
  scenes(filters: {lens: $lens}) {
    id
    layers {
      __typename
      ... on IntensityLayer {
        lens {
          id
        }
      }
    }
  }
}
"""

#: The world of the scene a path was drawn in: null for a path of a freestanding collection.
_DRAWN_IN = """
query DrawnIn($id: ID!) {
  annotation(id: $id) {
    collection {
      scene {
        worldCoordinateSystem {
          id
        }
      }
    }
  }
}
"""

#: A scene's paths, each with the tables computed from it.
_PATHS_OF_SCENE = """
query PathsOfScene($id: ID!) {
  scene(id: $id) {
    name
    annotationCollection {
      annotations(filters: {kind: PATH}, pagination: {limit: 1000}) {
        id
        name
        sourceIds
        derivedResidents {
          __typename
          ... on TableDataset {
            id
            columns {
              name
            }
          }
        }
      }
    }
  }
}
"""


def world_of(mikro: Mikro, dataset: Any) -> Any:
    """The space a scene of `dataset` adopts: its physical space if it has one, else its voxel grid.

    The physical space is the first one its voxel grid has an edge into whose axes carry a
    length unit: what ``calibrate`` authors.
    """
    grid = dataset.intrinsic_system
    if grid is None:
        raise ValueError(f"Dataset '{dataset.name}' was fetched without its voxel grid: nothing to stage over.")
    for edge in mikro.get_transformations(filters=TransformationFilter(input=str(grid.id))):
        axes = getattr(getattr(edge, "output", None), "axes", None) or ()
        if any(axis.unit and _UREG(str(axis.unit)).check("[length]") for axis in axes):
            return edge.output
    return grid


def staged_scene_of(mikro: Mikro, ais_lens: Any, cluster_lens: Any) -> str | None:
    """The scene that shows exactly these two channels, if one exists: where paths were already traced.

    Told by what its layers draw, never by its name. The scene of a run draws crops of
    them, which are other lenses, and surfaces beside.
    """
    pair = {str(ais_lens.id), str(cluster_lens.id)}
    result = mikro.rath.query(_SCENES_OF_LENS, {"lens": str(ais_lens.id)})
    for scene in (result.data or {}).get("scenes") or []:
        drawn = [layer for layer in scene.get("layers") or [] if layer.get("__typename") != "AnnotationLayer"]
        if (
            len(drawn) == 2
            and all(layer.get("__typename") == "IntensityLayer" for layer in drawn)
            and {str((layer.get("lens") or {}).get("id")) for layer in drawn} == pair
        ):
            return str(scene["id"])
    return None


def crop_of(mikro: Mikro, lens: Any, box: tuple[np.ndarray, np.ndarray], pinned: dict[str, int]) -> Any:
    """The lens over the box ``[low, high)`` of `lens`' own voxels, at `pinned`: a selection, no voxel is copied."""
    low, high = box
    slices = [SliceInput(axis=axis, start=int(lo), stop=int(hi), step=1) for axis, lo, hi in zip(SPACE, low, high)]
    slices += [SliceInput(axis=axis, start=value, stop=value + 1, step=1) for axis, value in pinned.items()]
    return mikro.create_lens(lens=lens.id, slices=slices)


def stage_run(mikro: Mikro, task: Task, path: Any, ais_lens: Any, cluster_lens: Any, out: Segmented) -> Any:
    """A 3D scene of one run: both channels cut to the box round the path, the AIS, the organelles and the adjusted path.

    One layer each, so each is switched on and off by itself. It stands over the world of
    the scene the path was drawn in, so everything lies where it lies there.
    """
    result = mikro.rath.query(_DRAWN_IN, {"id": str(path.id)})
    drawn_in = ((((result.data or {}).get("annotation") or {}).get("collection") or {}).get("scene") or {})
    world = (drawn_in.get("worldCoordinateSystem") or {}).get("id")
    if world is None:
        world = world_of(mikro, mikro.get_array_dataset(ais_lens.dataset.id)).id
    scene = mikro.create_scene(
        name=f"{out.name} · AIS and organelles",
        coordinate_system=world,
        blending=Blending.ADDITIVE,
        preferred_view=PreferredView.THREE_D,
    )
    channels = ((ais_lens, out.ais_box, AIS_COLOR), (cluster_lens, out.cluster_box, ORGANELLE_COLOR))
    for order, (lens, box, color) in enumerate(channels):
        mikro.create_volume_layer(
            lens=crop_of(mikro, lens, box, pinned_of(path, lens)).id, scene=scene.id, mode=ProjectionMode.MIP,
            color=color, blending=Blending.ADDITIVE, visible=True, order=order,
        )
    # The AIS see-through, so the organelles inside it show.
    mikro.create_mesh_layer(scene=scene.id, lens=out.ais_mesh.id, material_color=AIS_COLOR, opacity=0.3, visible=True, order=2)
    if out.cluster_mesh is not None:
        mikro.create_mesh_layer(scene=scene.id, lens=out.cluster_mesh.id, material_color=ORGANELLE_COLOR, visible=True, order=3)
    if drawn_in:
        # Drawn on the scene, into its own collection: that copies the world's axes, as the
        # collection the path was drawn into did, so the vertices are the same numbers.
        mikro.create_annotation(
            kind=AnnotationKind.PATH,
            vectors=[[float(v) for v in vertex] for vertex in out.adjusted_path.vectors],
            scene=scene.id,
            name=f"{out.name} · adjusted",
            coordinates=[CoordinateInput(name=c.name, value=int(c.value)) for c in path.coordinates or ()],
            stroke_color=list(_REFINED_COLOR),
        )
    else:
        task.log(f"'{out.name}' was not drawn in a scene: the 3D scene shows the AIS without its path.")
    return scene


@app.action(collections=["display"])
def stage_channels(
    dataset: Annotated[ChannelVolume, VOLUME_PICKER, Description("The z-stack the two channels are taken from")],
    ais_lens: Annotated[SingleChannelVolume, CHANNEL_OF_DATASET, Description("The AIS channel (AnkyrinG, NaV), a lens of the dataset")],
    cluster_lens: Annotated[SingleChannelVolume, CHANNEL_OF_DATASET, Description("The organelle channel (synaptopodin, ...), a lens of the dataset")],
    *,
    mikro: Mikro,
    task: Task,
) -> Annotated[Scene, Key("scene"), Description("A scene of the two channels, one layer each, to trace paths in")]:
    """Stage Channels

    Shows the AIS channel in green and the organelle channel in magenta, one layer each,
    in a 3D scene to trace paths in. The scene stands over the dataset's physical space when
    it has one, so paths drawn there are in micrometres. A pair that was staged before
    gets its scene back, with the paths traced in it
    """
    if str(ais_lens.id) == str(cluster_lens.id):
        raise ValueError(f"Both channels are {lens_label(ais_lens)}: pick two different channels.")
    for lens in (ais_lens, cluster_lens):
        if str(lens.dataset.id) != str(dataset.id):
            raise ValueError(f"{lens_label(lens)} is not a channel of '{dataset.name}'.")
    earlier = staged_scene_of(mikro, ais_lens, cluster_lens)
    if earlier is not None:
        task.log("These two channels were staged before: showing that scene, with the paths traced in it.")
        return mikro.get_scene(earlier)
    scene = mikro.create_scene(
        name=f"{dataset.name} · AIS and organelle channels",
        coordinate_system=world_of(mikro, dataset).id,
        blending=Blending.ADDITIVE,
        # Through z: an axon leaves any one slice, so its path is traced in 3D.
        preferred_view=PreferredView.THREE_D,
    )
    for order, (lens, color) in enumerate(((ais_lens, AIS_COLOR), (cluster_lens, ORGANELLE_COLOR))):
        mikro.create_volume_layer(
            lens=lens.id, scene=scene.id, mode=ProjectionMode.MIP, color=color, blending=Blending.ADDITIVE, visible=True, order=order,
        )
    return scene


@app.action(collections=["segment"], logo=logo_of("segment_ais_tube_along_path"))
def segment_ais_tube_into_scene(
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
    Annotated[AisTubeRun, Key("summary"), Description("The AIS and the organelle inside it, in path order")],
    Annotated[Scene, Key("scene"), Description("Both channels round the path in 3D, the AIS, the organelles and the adjusted path, a layer each")],
    Annotated[Annotation, Key("adjusted_path"), Description("The path adjusted onto the AIS")],
    Annotated[MeshLens, Key("ais"), Description("The AIS surface")],
    Annotated[MeshLens | None, Key("organelles"), Description("The organelles, one mesh object per cluster_id; None when none were found")],
    Annotated[TableDataset | None, Key("organelle_table"), Description("One row per organelle; None when none were found")],
    Annotated[TableDataset, Key("profile_table"), Description("One row per 1 µm of the adjusted path")],
]:
    """Segment AIS Tube Into Scene

    Segments the AIS and the organelles inside it along the path, as Segment AIS Tube
    Along Path does, and shows the run in a 3D scene of its own: both channels cut to the
    box round the path, the AIS surface, the organelles and the adjusted path, one layer
    each
    """
    params = core.Params(
        diameter_um=diameter_um, ais_scale=threshold_scale, organelle_scale=cluster_threshold_scale,
        max_same_distance_um=max_same_distance_um, min_volume_um3=min_volume_um3,
    )
    out = segment_along(mikro, task, path, ais_lens, cluster_lens, params)
    task.progress(92, "Building the 3D scene")
    scene = stage_run(mikro, task, path, ais_lens, cluster_lens, out)
    task.progress(100, f"{out.found.count} organelles on the AIS along '{out.name}'")
    return out.found, scene, out.adjusted_path, out.ais_mesh, out.cluster_mesh, out.cluster_table, out.profile_table


def latest_organelle_table(residents: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """Whether a path was segmented, and the organelle table of its latest run: None when that run found none.

    Every run stores a profile table, and before it an organelle table when it found any.
    Ids ascend, so the latest run's organelle table lies between the last two profile tables.
    """
    profiles, organelles = [], []
    for resident in residents:
        if resident.get("__typename") != "TableDataset":
            continue
        columns = {column["name"] for column in resident.get("columns") or []}
        if "ais_volume_um3" in columns:
            profiles.append(int(resident["id"]))
        elif {"cluster_id", "depth_um"} <= columns:
            organelles.append(int(resident["id"]))
    if not profiles:
        return False, None
    profiles.sort()
    floor = profiles[-2] if len(profiles) > 1 else -1
    of_run = [table for table in organelles if floor < table < profiles[-1]]
    return True, str(max(of_run)) if of_run else None


@app.action(collections=["export"])
def export_organelles(
    scene: Annotated[Scene, Description("The scene the paths were traced in")],
    *,
    mikro: Mikro,
    task: Task,
) -> Annotated[File, Key("file"), Description("One row per organelle of every segmented path of the scene, as CSV")]:
    """Export Organelles

    Gathers the organelles of every path traced in the scene, from the latest run of
    each, into one CSV file: a row per organelle, with the path it lies on. A path whose
    AIS holds no organelle adds no row
    """
    result = mikro.rath.query(_PATHS_OF_SCENE, {"id": str(scene.id)})
    found = (result.data or {}).get("scene") or {}
    paths = ((found.get("annotationCollection") or {}).get("annotations")) or []
    # A path made from a path is a run's adjusted one, not one that was traced.
    drawn = [path for path in paths if not path.get("sourceIds")]
    frames, tables, empty, unsegmented = [], [], 0, 0
    for index, path in enumerate(drawn):
        task.progress(int(90 * index / max(len(drawn), 1)), f"Reading '{path.get('name') or path['id']}'")
        segmented, table_id = latest_organelle_table(path.get("derivedResidents") or [])
        if not segmented:
            unsegmented += 1
        elif table_id is None:
            empty += 1
        else:
            rows = mikro.get_table_dataset(table_id).store.parquet_dataset.read_pandas().to_pandas()
            rows.insert(0, "path", path.get("name") or "")
            rows.insert(0, "path_id", path["id"])
            frames.append(rows)
            tables.append(table_id)
    if not frames:
        raise ValueError(
            f"Nothing to export: of the {len(drawn)} paths traced in '{found.get('name')}', {unsegmented} were never "
            f"segmented and {empty} hold no organelle."
        )
    organelles = pd.concat(frames, ignore_index=True)
    task.log(
        f"{len(organelles)} organelles on {len(frames)} paths; {empty} segmented paths hold none, "
        f"{unsegmented} paths were never segmented."
    )
    task.progress(95, "Uploading the CSV")
    name = re.sub(r"[^\w.-]+", "_", str(found.get("name") or f"scene {scene.id}")).strip("_")
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / f"{name}_organelles.csv"
        organelles.to_csv(target, index=False)
        return mikro.from_file_like(
            file=target,
            file_name=target.name,
            export_of=[TableDatasetExportOfInput(table_dataset=table) for table in tables],
        )


# --------------------------------------------------------------------------- #
# The panel: opened on a dataset
# --------------------------------------------------------------------------- #
# Left, the two channels as a scene to trace in: a path is segmented as soon as it is
# finished, and the selected one again with what the Tune tab holds. Right, the latest
# run: its 3D scene (both channels round the path, the AIS, the organelles and the adjusted
# path, each a layer to switch on and off) over its numbers and its organelles. The latest
# run is the action's latest task, whichever of the two calls started it, so it is still
# there when the panel is opened again. The header exports every path's organelles as CSV.
WORKBENCH = """
<Blok kind="agent" context="panel" catalog="https://arkitekt.live/catalogs/orkestrator/v1" chrome="none">
  <local name="ais" kind="STRUCTURE" identifier="@mikro/arraylens" nullable />
  <local name="organelle" kind="STRUCTURE" identifier="@mikro/arraylens" nullable />
  <local name="selected" nullable />
  <local name="diameter" like="self.segment_ais_tube_into_scene.diameter_um" />
  <local name="ais_scale" like="self.segment_ais_tube_into_scene.threshold_scale" />
  <local name="organelle_scale" like="self.segment_ais_tube_into_scene.cluster_threshold_scale" />
  <local name="same" like="self.segment_ais_tube_into_scene.max_same_distance_um" />
  <local name="smallest" like="self.segment_ais_tube_into_scene.min_volume_um3" />
  <Toast task={exported}>
    <done>The organelles are exported: the CSV is linked in the header.</done>
    <failed>{exported.error}</failed>
  </Toast>
  <div className="@container h-full">
    <div className="flex h-full flex-col gap-3 p-4">

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-brand-soft text-brand">
            <Icon name="activity" size={20} />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <p className="text-base font-semibold tracking-tight">AIS organelles</p>
            <p className="truncate text-sm text-muted-foreground">Trace an axon, look at its AIS and the organelles inside it, export the measurements.</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StructureDisplay value={args.dataset} variant="chip" link={true} />
          <Badge variant={self.tasks.segment_ais_tube_into_scene.failed ? "destructive" : self.tasks.segment_ais_tube_into_scene.running ? "default" : "outline"}>{self.tasks.segment_ais_tube_into_scene.status}</Badge>
          <if when={exported.returns.file != null}>
            <StructureDisplay value={exported.returns.file} variant="chip" link={true} />
          </if>
          <Button
            variant="outline"
            icon={exported.running ? "loader" : "download"}
            disabled={staged.returns.scene == null || exported.running}
            onClick={self.export_organelles(scene: staged.returns.scene) into exported}
          >{exported.running ? exported.message ?? "Exporting" : "Export CSV"}</Button>
        </div>
      </div>

      <Resizable direction="row" className="min-h-0 flex-1 rounded-2xl border">
        <ResizablePanel defaultSize={50} minSize={30}>
          <div className="flex h-full flex-col">
            <div className="flex flex-wrap items-end gap-2 border-b p-3">
              <div className="flex min-w-[10rem] flex-1 flex-col gap-1">
                <span className="text-xs text-muted-foreground">AIS channel</span>
                <StructurePicker identifier="@mikro/arraylens" ward="mikro" bind={ais} variables={logic.coalesce(values: [{dataset: args.dataset}])} placeholder="AnkyrinG, NaV" query='query AisChannels($search: String, $values: [ID!], $dataset: ID!, $limit: Int, $offset: Int = 0) { options: lenses(filters: {kind: ARRAY, dataset: $dataset, search: $search, ids: $values, requires: [{key: "@mikro/n_channels", operator: LTE, value: 1}, {key: "@mikro/n_space_axes", operator: EQUALS, value: 3}]}, pagination: {limit: $limit, offset: $offset}) { value: id label } }' />
              </div>
              <div className="flex min-w-[10rem] flex-1 flex-col gap-1">
                <span className="text-xs text-muted-foreground">Organelle channel</span>
                <StructurePicker identifier="@mikro/arraylens" ward="mikro" bind={organelle} variables={logic.coalesce(values: [{dataset: args.dataset}])} placeholder="Synaptopodin" query='query OrganelleChannels($search: String, $values: [ID!], $dataset: ID!, $limit: Int, $offset: Int = 0) { options: lenses(filters: {kind: ARRAY, dataset: $dataset, search: $search, ids: $values, requires: [{key: "@mikro/n_channels", operator: LTE, value: 1}, {key: "@mikro/n_space_axes", operator: EQUALS, value: 3}]}, pagination: {limit: $limit, offset: $offset}) { value: id label } }' />
              </div>
              <Button
                icon={staged.running ? "loader" : "layers"}
                disabled={ais == null || organelle == null || staged.running}
                onClick={self.stage_channels(dataset: args.dataset, ais_lens: ais, cluster_lens: organelle) into staged}
              >Show</Button>
            </div>
            <if when={staged.failed}>
              <div className="flex gap-2 border-b border-destructive/40 bg-destructive/10 p-2 text-destructive">
                <Icon name="alert-triangle" size={14} className="mt-0.5 shrink-0" />
                <p className="min-w-0 whitespace-pre-wrap break-words font-mono text-[11px]">{staged.error}</p>
              </div>
            </if>
            <div className="min-h-0 flex-1">
              <if when={staged.returns.scene != null}>
                <MikroScene
                  value={staged.returns.scene}
                  bind={selected}
                  className="h-full"
                  onAnnotation={self.segment_ais_tube_into_scene(path: event.annotation, ais_lens: staged.args.ais_lens, cluster_lens: staged.args.cluster_lens, diameter_um: diameter, threshold_scale: ais_scale, cluster_threshold_scale: organelle_scale, max_same_distance_um: same, min_volume_um3: smallest) into run}
                >
                  <div className="flex h-full flex-col">
                    <div className="min-h-0 flex-1">
                      <MikroSceneViewport height="100%" />
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-2 border-t p-2">
                      <p className="min-w-0 flex-1 text-xs text-muted-foreground">Draw a path along the axon: it is segmented as soon as you finish it. Click a path to select it.</p>
                      <Button
                        size="sm"
                        variant="secondary"
                        icon="refresh"
                        disabled={selected.annotation == null || self.tasks.segment_ais_tube_into_scene.running}
                        onClick={self.segment_ais_tube_into_scene(path: selected.annotation, ais_lens: staged.args.ais_lens, cluster_lens: staged.args.cluster_lens, diameter_um: diameter, threshold_scale: ais_scale, cluster_threshold_scale: organelle_scale, max_same_distance_um: same, min_volume_um3: smallest) into again}
                      >Segment the selected path again</Button>
                    </div>
                  </div>
                </MikroScene>
                <else>
                  <div className="flex h-full items-center justify-center p-6">
                    <Empty icon="layers" title="Pick the two channels" description="Choose the AIS channel and the organelle channel of this image and press Show. They open as a scene to trace axons in." />
                  </div>
                </else>
              </if>
            </div>
          </div>
        </ResizablePanel>
        <ResizableHandle withHandle={true} />
        <ResizablePanel defaultSize={50} minSize={25}>
          <div className="flex h-full flex-col">
            <if when={self.tasks.segment_ais_tube_into_scene.running}>
              <div className="flex flex-col gap-1.5 border-b p-3">
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-xs text-muted-foreground">{self.tasks.segment_ais_tube_into_scene.message ?? "Starting"}</p>
                  <span className="font-mono text-[11px] text-muted-foreground">{self.tasks.segment_ais_tube_into_scene.progress}%</span>
                </div>
                <Progress value={self.tasks.segment_ais_tube_into_scene.progress} max={100} />
              </div>
            </if>
            <if when={self.tasks.segment_ais_tube_into_scene.failed}>
              <div className="flex gap-2 border-b border-destructive/40 bg-destructive/10 p-3 text-destructive">
                <Icon name="alert-triangle" size={14} className="mt-0.5 shrink-0" />
                <p className="min-w-0 whitespace-pre-wrap break-words font-mono text-[11px]">{self.tasks.segment_ais_tube_into_scene.error}</p>
              </div>
            </if>
            <Resizable direction="column" className="min-h-0 flex-1">
              <ResizablePanel defaultSize={58} minSize={20}>
                <if when={self.tasks.segment_ais_tube_into_scene.returns.scene != null}>
                  <MikroScene value={self.tasks.segment_ais_tube_into_scene.returns.scene} className="h-full">
                    <div className="flex h-full">
                      <div className="min-w-0 flex-1">
                        <MikroSceneViewport height="100%" sliders={false} />
                      </div>
                      <div className="flex w-[15rem] shrink-0 flex-col gap-2 overflow-y-auto border-l p-2">
                        <p className="text-xs font-medium text-muted-foreground">Show</p>
                        <MikroSceneLayers height="100%" />
                      </div>
                    </div>
                  </MikroScene>
                  <else>
                    <div className="flex h-full items-center justify-center p-6">
                      <Empty icon="box" title="No AIS yet" description="Trace an axon on the left. Its AIS, the organelles inside it and the image round the path show up here in 3D, each a layer to switch on and off." />
                    </div>
                  </else>
                </if>
              </ResizablePanel>
              <ResizableHandle withHandle={true} />
              <ResizablePanel defaultSize={42} minSize={15}>
                <Tabs defaultValue="results" className="flex h-full flex-col overflow-y-auto p-3">
                  <TabsList>
                    <TabsTrigger value="results" label="Results" icon="bar-chart" />
                    <TabsTrigger value="organelles" label="Organelles" icon="table" />
                    <TabsTrigger value="tune" label="Tune" icon="sliders" />
                  </TabsList>

                  <TabsContent value="results" className="flex flex-col gap-3 pt-3">
                    <if when={self.tasks.segment_ais_tube_into_scene.returns.summary != null}>
                      <p className="truncate text-xs text-muted-foreground">Along {self.tasks.segment_ais_tube_into_scene.returns.summary.path}: {self.tasks.segment_ais_tube_into_scene.returns.summary.cluster_channel} inside {self.tasks.segment_ais_tube_into_scene.returns.summary.ais_channel}</p>
                      <div className="flex flex-wrap gap-2">
                        <div className="flex min-w-[8rem] flex-1 flex-col rounded-xl border bg-card p-3">
                          <p className="text-2xl font-semibold leading-none tracking-tight tabular-nums">{self.tasks.segment_ais_tube_into_scene.returns.summary.count}</p>
                          <p className="mt-1 text-[11px] text-muted-foreground">organelles</p>
                        </div>
                        <div className="flex min-w-[8rem] flex-1 flex-col rounded-xl border bg-card p-3">
                          <p className="text-2xl font-semibold leading-none tracking-tight tabular-nums">{fmt.number(value: self.tasks.segment_ais_tube_into_scene.returns.summary.length, digits: 1)}</p>
                          <p className="mt-1 text-[11px] text-muted-foreground">µm of AIS</p>
                        </div>
                        <div className="flex min-w-[8rem] flex-1 flex-col rounded-xl border bg-card p-3">
                          <p className="text-2xl font-semibold leading-none tracking-tight tabular-nums">{fmt.number(value: self.tasks.segment_ais_tube_into_scene.returns.summary.mean_radius, digits: 2)}</p>
                          <p className="mt-1 text-[11px] text-muted-foreground">µm mean radius</p>
                        </div>
                        <div className="flex min-w-[8rem] flex-1 flex-col rounded-xl border bg-card p-3">
                          <p className="text-2xl font-semibold leading-none tracking-tight tabular-nums">{fmt.number(value: self.tasks.segment_ais_tube_into_scene.returns.summary.density_per_length, digits: 2)}</p>
                          <p className="mt-1 text-[11px] text-muted-foreground">organelles per µm</p>
                        </div>
                      </div>
                      <if when={self.tasks.segment_ais_tube_into_scene.returns.summary.count > 0}>
                        <Chart type="bar" data={self.tasks.segment_ais_tube_into_scene.returns.summary.clusters} categoryKey="label" series={[{key: "volume", label: "Volume (µm³)"}]} height={180} />
                      </if>
                      <p className="text-xs text-muted-foreground">AIS threshold {self.tasks.segment_ais_tube_into_scene.returns.summary.ais_threshold}, organelle threshold {self.tasks.segment_ais_tube_into_scene.returns.summary.threshold} ({self.tasks.segment_ais_tube_into_scene.returns.summary.cluster_threshold_source}) over a diffuse level of {self.tasks.segment_ais_tube_into_scene.returns.summary.cluster_background}</p>
                      <div className="flex flex-wrap items-center gap-2">
                        <StructureDisplay value={self.tasks.segment_ais_tube_into_scene.returns.organelle_table} variant="chip" link={true} />
                        <StructureDisplay value={self.tasks.segment_ais_tube_into_scene.returns.profile_table} variant="chip" link={true} />
                        <StructureDisplay value={self.tasks.segment_ais_tube_into_scene.returns.scene} variant="chip" link={true} />
                      </div>
                      <else>
                        <Empty icon="bar-chart" title="No results yet" description="The numbers of the latest run show up here." />
                      </else>
                    </if>
                  </TabsContent>

                  <TabsContent value="organelles" className="pt-3">
                    <if when={self.tasks.segment_ais_tube_into_scene.returns.summary.count > 0}>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead text="#" />
                            <TableHead text="Along (µm)" align="end" />
                            <TableHead text="Volume (µm³)" align="end" />
                            <TableHead text="Inside the edge (µm)" align="end" />
                            <TableHead text="Intensity" align="end" />
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          <foreach items={self.tasks.segment_ais_tube_into_scene.returns.summary.clusters} let="row" key={row.cluster}>
                            <TableRow>
                              <TableCell text={row.cluster} />
                              <TableCell text={row.arclength} align="end" />
                              <TableCell text={row.volume} align="end" />
                              <TableCell text={row.depth} align="end" />
                              <TableCell text={row.intensity} align="end" />
                            </TableRow>
                          </foreach>
                        </TableBody>
                      </Table>
                      <else>
                        <Empty icon="table" title="No organelles" description="The organelles of the latest run are listed here, in path order." />
                      </else>
                    </if>
                  </TabsContent>

                  <TabsContent value="tune" className="flex flex-col gap-4 pt-3">
                    <p className="text-xs text-muted-foreground">Used by the next path you draw, and by Segment the selected path again.</p>
                    <Field label="Tube diameter (µm)" description="The tube round the path that holds the AIS.">
                      <Slider bind={diameter} min={1} max={8} step={0.25} />
                      <Text text={diameter} tone="muted" size="sm" mono={true} />
                    </Field>
                    <Field label="AIS threshold" description="Scales the AIS threshold: below 1 grows the AIS, above 1 shrinks it.">
                      <Slider bind={ais_scale} min={0.5} max={2} step={0.05} />
                      <Text text={ais_scale} tone="muted" size="sm" mono={true} />
                    </Field>
                    <Field label="Organelle threshold" description="Scales the organelle threshold above the diffuse level inside the AIS: below 1 grows the organelles, above 1 shrinks them.">
                      <Slider bind={organelle_scale} min={0.5} max={2} step={0.05} />
                      <Text text={organelle_scale} tone="muted" size="sm" mono={true} />
                    </Field>
                    <Field label="One organelle within (µm)" description="Pieces no further apart than this are one organelle; 0 joins only touching pieces.">
                      <Slider bind={same} min={0} max={3} step={0.1} />
                      <Text text={same} tone="muted" size="sm" mono={true} />
                    </Field>
                    <Field label="Smallest organelle (µm³)" description="Smaller organelles are dropped.">
                      <Slider bind={smallest} min={0} max={2} step={0.05} />
                      <Text text={smallest} tone="muted" size="sm" mono={true} />
                    </Field>
                  </TabsContent>
                </Tabs>
              </ResizablePanel>
            </Resizable>
          </div>
        </ResizablePanel>
      </Resizable>
    </div>
  </div>
</Blok>
"""

app.blok(
    "ais_organelles_workbench",
    WORKBENCH,
    description="Opened on a z-stack: show its AIS and organelle channels, trace axons, and each AIS shows up in 3D with its organelles, layer by layer, beside its measurements. Exports every path's organelles as CSV.",
    args={"dataset": Annotated[ChannelVolume, VOLUME_PICKER]},
    mode="strict",
)


if __name__ == "__main__":
    run(app)
