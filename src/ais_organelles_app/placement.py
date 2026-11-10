"""Placing a path on a lens and writing the adjusted path back: the server side of the geometry.

Copied from ``testing/path_segment_app.py``, ``testing/straighten_path_app.py`` and
``testing/cluster_segment_path_app.py`` (only what this app uses), verbatim apart from
import sorting and two changes: `lens_volume` returns the lazy volume itself, and
`spacing_of` reads the voxel size directly, since the app keeps no cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pint
import xarray as xr
from mikro import Mikro
from mikro.api.schema import (
    Annotation,
    AnnotationKind,
    AnnotationRelation,
    AnnotationSourceInput,
    AxisType,
    CoordinateInput,
)
from mikro.vocabulary import COMPOSABLE_KINDS

logger = logging.getLogger(__name__)

#: How far past a face, in voxels, a vertex is still read as lying on it.
_ON_A_FACE = 1e-6

#: The refined path keeps vertices within this fraction of the finest voxel of the centreline.
_SIMPLIFY = 0.5

#: Shown on the refined path, so it is told apart from the drawn one at a glance.
_REFINED_COLOR = (255, 170, 0, 255)

_MADE_FROM = """
query RefinedOf($id: ID!) {
  annotation(id: $id) {
    collection {
      id
    }
    derivedAnnotations {
      id
      derivedFrom {
        relation
        reason
        source {
          id
        }
      }
    }
  }
}
"""

_UREG = pint.UnitRegistry()

_PATH_KINDS = {"PATH", "LINE"}

#: The volume is always handled in this order, whatever order the lens stores it in.
SPACE = ("z", "y", "x")

_DATASET_NAME_DOCUMENT = """
query DatasetNameOf($id: ID!) {
  lens(id: $id) {
    ... on ArrayLens {
      dataset {
        name
      }
    }
  }
}
"""


def _value(kind: Any) -> str:
    """An enum or its value, as the bare string."""
    return str(getattr(kind, "value", kind))


def voxel_size_um(lens: Any) -> dict[str, float]:
    """The lens' voxel size in micrometres per SPACE axis, as the server resolves it: ``{}`` when uncalibrated.

    The server reads it along the edges to the first space with units, BY_DIMENSION parts
    and stepped crops included, and picks one calibration by the edges' validity. A lens
    without the field came from a query that does not select it, which is refused rather
    than read as "uncalibrated".
    """
    if not hasattr(lens, "voxel_size"):
        raise TypeError(f"{type(lens).__name__} carries no `voxel_size`: fetch the lens through the ArrayLens fragment.")
    sizes: dict[str, float] = {}
    for entry in lens.voxel_size or ():
        if str(getattr(entry.type, "value", entry.type)) == "SPACE":
            sizes[entry.axis] = float((entry.size * _UREG(entry.unit)).to("micrometer").magnitude)
    return sizes


def dataset_name_of(mikro: Mikro, lens: Any) -> str:
    """The name of the dataset a lens looks at; not in the generated `ArrayLens` fragment either."""
    try:
        result = mikro.rath.query(_DATASET_NAME_DOCUMENT, {"id": str(lens.id)})
    except Exception:
        return f"lens {lens.id}"
    dataset = ((result.data or {}).get("lens") or {}).get("dataset") or {}
    return dataset.get("name") or f"lens {lens.id}"


def named_vertices(annotation: Any) -> tuple[np.ndarray, list[str]]:
    """The path's vertices as ``(n, k)`` and the axis each column is along.

    The columns of an annotation's vectors follow its coordinate system's axes.
    """
    if _value(annotation.kind) not in _PATH_KINDS:
        raise ValueError(
            f"Annotation '{annotation.name}' is a {_value(annotation.kind)}, not a PATH."
        )
    vectors = np.asarray(annotation.vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError(f"Annotation '{annotation.name}' has fewer than two vertices.")

    axes = sorted(annotation.coordinate_system.axes, key=lambda a: a.order)
    # A vertex has three components whatever it is drawn over. In a three-axis
    # space those are its axes; in a wider one (c, z, y, x) they are its space
    # axes, and the rest is said by the annotation's pinned coordinates.
    if vectors.shape[1] == len(axes):
        columns = [axis.name for axis in axes]
    else:
        columns = [axis.name for axis in axes if _value(axis.type) == "SPACE"]
    if vectors.shape[1] != len(columns):
        raise ValueError(
            f"Annotation '{annotation.name}' has {vectors.shape[1]} values per vertex, which "
            f"matches neither the axes {[axis.name for axis in axes]} it is drawn over nor "
            "the space axes among them."
        )
    return vectors, columns


@dataclass(frozen=True)
class PathFrame:
    """The affine from a path's stored vertices to the lens' voxel indices: ``index = A v + b``.

    Kept as a matrix rather than applied once, because the refined path goes back
    through its inverse. The round trip is then exact, and the refined path is
    stored in the same units and column order as the one that was drawn.
    """

    columns: tuple[str, ...]
    matrix: np.ndarray
    offset: np.ndarray
    warning: str | None = None

    def to_index(self, vertices: np.ndarray) -> np.ndarray:
        return np.asarray(vertices, dtype=np.float64) @ self.matrix.T + self.offset

    def to_drawn(self, index: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.matrix, (np.asarray(index) - self.offset).T).T


def path_frame(
    mikro: Mikro, annotation: Any, lens: Any, space_axes: tuple[str, ...]
) -> tuple[np.ndarray, PathFrame]:
    """The path's vertices as lens voxel indices (in `space_axes` order), and the frame that took them there.

    The same placement as ``straighten_path_app.path_vertices``, including its rules:
    a freestanding collection is trusted as pixel indices and the frame says so, and
    an UNMAPPABLE edge does not count as a route.
    """
    drawn_in = annotation.coordinate_system
    target = lens.coordinate_system
    if drawn_in is None or target is None:
        raise ValueError(
            "The path or the image has no coordinate system to place the one on the other."
        )
    vertices, columns = named_vertices(annotation)

    def as_they_are(warning: str | None = None) -> PathFrame:
        missing = [axis for axis in space_axes if axis not in columns]
        if missing:
            raise ValueError(
                f"Annotation '{annotation.name}' is drawn over axes {columns}, which do not name "
                f"the image's space axes {missing}."
            )
        matrix = np.zeros((3, 3))
        for row, axis in enumerate(space_axes):
            matrix[row, columns.index(axis)] = 1.0
        return PathFrame(tuple(columns), matrix, np.zeros(3), warning)

    if str(drawn_in.id) == str(target.id):
        frame = as_they_are()
    else:
        graph = drawn_in.graph(mikro=mikro)
        if not any(
            str(drawn_in.id) in (str(edge.input.id), str(edge.output.id))
            for edge in graph.transformations
            if _value(getattr(edge, "kind", "")) in COMPOSABLE_KINDS
            and getattr(edge, "input", None) is not None
            and getattr(edge, "output", None) is not None
        ):
            frame = as_they_are(
                f"The collection of '{annotation.name}' is freestanding: nothing records which "
                f"image it was drawn on. Its vertices are read as pixel indices of '{target.name}'."
            )
        else:
            try:
                onto = drawn_in.axis_map_to(target, graph=graph, mikro=mikro)
                # The affine read off the map itself: the image of the origin and of each basis vector.
                probes = np.vstack([np.zeros(3), np.eye(3)])
                mapped = onto.apply(probes, input_axes=columns, output_axes=space_axes)
            except ValueError as error:
                raise ValueError(
                    f"Annotation '{annotation.name}' cannot be placed on this image: its space "
                    f"'{drawn_in.name}' does not map onto '{target.name}' along {list(space_axes)}. "
                    f"{error} Pass an image that shares a space with the path."
                ) from error
            frame = PathFrame(tuple(columns), (mapped[1:] - mapped[0]).T, mapped[0])
    if abs(np.linalg.det(frame.matrix)) < 1e-12:
        raise ValueError(
            f"Annotation '{annotation.name}' maps onto the image degenerately; its path has no inverse."
        )
    return frame.to_index(vertices), frame


def lens_volume(lens: Any, space_axes: tuple[str, ...], pinned: dict[str, int]) -> xr.DataArray:
    """The 3D volume a lens shows at `pinned`, lazily: selecting and transposing reads no voxels.

    Everything beside the space axes has to be a single position once `pinned`
    is applied, or the volume is refused rather than silently cut to its first plane.
    """
    data = lens.xarray_data
    data = data.isel({name: value for name, value in pinned.items() if name in data.dims})
    wide = {str(d): int(data.sizes[d]) for d in data.dims if d not in space_axes and data.sizes[d] > 1}
    if wide:
        raise ValueError(
            f"The image still spans {wide} beside its space axes. Pass a lens of one "
            "channel, or draw the path at one timepoint."
        )
    return data.isel({d: 0 for d in data.dims if d not in space_axes}).transpose(*space_axes)


def write_refined_path(mikro: Mikro, path: Any, vectors: np.ndarray, name: str, reason: str, made_by: str) -> Annotation:
    """Put the refined path beside the drawn one, made from it, replacing the one an earlier run made.

    The refined path names the drawn one in ``derivedFrom`` (REFINED: both are paths), so the
    scene hides it beside its source and draws it over the source's card. Which earlier shape
    is this run's is read off the lineage, never the name: the drawn path's REFINED children
    whose derivation `reason` mentions `made_by`. A re-run rewrites the first of them (vectors
    and name) and deletes the rest -- which mikro refuses while something was made from one,
    and then it is left as it is.
    """
    result = mikro.rath.query(_MADE_FROM, {"id": str(path.id)})
    annotation = (result.data or {}).get("annotation") or {}
    earlier = [
        child["id"]
        for child in annotation.get("derivedAnnotations") or []
        if any(
            str((link.get("source") or {}).get("id")) == str(path.id)
            and _value(link.get("relation")) == "REFINED"
            and made_by in (link.get("reason") or "")
            for link in child.get("derivedFrom") or []
        )
    ]
    # A list per vertex: `ThreeDVector` refuses a tuple.
    rows = [[float(v) for v in row] for row in vectors]
    if earlier:
        for extra in earlier[1:]:
            try:
                mikro.delete_annotation(id=extra)
            except Exception as error:
                logger.info("Kept an earlier '%s' (%s): %s", name, extra, error)
        return mikro.update_annotation(id=earlier[0], vectors=rows, name=name)
    return mikro.create_annotation(
        kind=AnnotationKind.PATH,
        vectors=rows,
        collection=(annotation.get("collection") or {})["id"],
        name=name,
        coordinates=[CoordinateInput(name=c.name, value=int(c.value)) for c in path.coordinates or ()],
        stroke_color=list(_REFINED_COLOR),
        # Every caller hands a PATH, and a path made from a path is REFINED.
        derived_from=[AnnotationSourceInput(annotation=path.id, relation=AnnotationRelation.REFINED, reason=reason)],
    )


def pinned_of(path: Any, lens: Any) -> dict[str, int]:
    """The path's positions on the lens' axes beside space and channel (its timepoint, say)."""
    skip = set(lens.axes_of_type(AxisType.SPACE)) | set(lens.axes_of_type(AxisType.CHANNEL))
    lens_axes = tuple(lens.axis_names)
    return {str(c.name): int(c.value) for c in path.coordinates or () if c.name in lens_axes and c.name not in skip}


def spacing_of(lens: Any) -> tuple[np.ndarray, bool]:
    """The voxel size in µm along `SPACE`, and whether it is one: ones when uncalibrated."""
    spacing_um = voxel_size_um(lens)
    calibrated = all(axis in spacing_um for axis in SPACE)
    return (np.array([spacing_um[axis] for axis in SPACE]) if calibrated else np.ones(3)), calibrated


def refuse_outside(vertices: np.ndarray, shape: np.ndarray, path: Any) -> None:
    if (vertices < -0.5 - _ON_A_FACE).any() or (vertices > shape - 0.5 + _ON_A_FACE).any():
        raise ValueError(
            f"Annotation '{path.name}' leaves the image: its vertices span "
            f"{vertices.min(axis=0).round(1).tolist()} to {vertices.max(axis=0).round(1).tolist()} "
            f"over {list(SPACE)}, the image is {shape.tolist()}."
        )


def lens_label(lens: Any) -> str:
    return getattr(lens, "label", None) or f"lens {lens.id}"
