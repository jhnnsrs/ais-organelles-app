"""The action against a stubbed mikro: two lenses on two grids, joined through the drawn space.

Each lens records the chunks dask fetches from it, so the test sees that only a crop
round the path is downloaded.

The coordinate graph is mikro's own model classes (so `path_frame` walks it with
mikro's `axis_map_to`): each lens' voxel space is calibrated into one physical space,
which the scene's drawing space maps onto by axis name. The organelle channel is on a
grid twice as coarse in xy as the AIS channel.

Run these with `uv run pytest`; the release workflow does too.
"""

from __future__ import annotations

from types import SimpleNamespace as ns
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
from arkitekt import Task
from arkitekt.runtime import Runtime
from mikro.api.schema import (
    AnnotationRelation,
    Axis,
    AxisType,
    CoordinateSystem,
    TransformationByDimensionTransformation,
    TransformationPart,
    TransformationScaleTransformation,
)

from ais_organelles_app import core
from app import SPACE, XYZ, app, segment_ais_tube_along_path


def system(names: str, system_id: str, unit: str | None = None) -> CoordinateSystem:
    return CoordinateSystem(
        id=system_id,
        name=system_id,
        axes=tuple(
            Axis(id=f"{system_id}-{n}", order=i, name=n, unit=None if n == "c" else unit, type="CHANNEL" if n == "c" else "SPACE")
            for i, n in enumerate(names)
        ),
    )


PHYSICAL = system("czyx", "physical", "micrometer")
DRAWING = system("zyx", "drawing")


def calibration(system_id: str, spacing: np.ndarray) -> tuple[CoordinateSystem, Any]:
    image = system("czyx", system_id)
    edge = TransformationScaleTransformation(
        id=f"calibration-{system_id}", kind="SCALE", version=1, input=image, output=PHYSICAL, scale=(1.0, *spacing)
    )
    return image, edge


ON_SCENE = TransformationByDimensionTransformation(
    id="scene", kind="BY_DIMENSION", version=1, input=DRAWING, output=PHYSICAL,
    inputAxes=tuple("zyx"), outputAxes=tuple("zyx"),
    parts=(TransformationPart(kind="IDENTITY", inputAxes=tuple("zyx"), outputAxes=tuple("zyx"), scale=(), translation=(), affine=()),),
)


class Recorded:
    """A (c, z, y, x) volume that counts the voxels dask fetches from it: a store's chunks."""

    def __init__(self, volume: np.ndarray):
        self.volume = volume[None]
        self.shape, self.dtype, self.ndim = self.volume.shape, self.volume.dtype, self.volume.ndim
        self.fetched = 0

    def __getitem__(self, key):
        out = self.volume[key]
        self.fetched += out.size
        return out


class StubLens:
    axis_names = ("c", "z", "y", "x")

    def __init__(self, volume, coordinate_system, lens_id, spacing):
        self.store, self.coordinate_system, self.id, self.label = Recorded(volume), coordinate_system, lens_id, lens_id
        # What the server resolves along the calibration edge: `ArrayLens.voxelSize`.
        self.voxel_size = tuple(ns(axis=axis, size=float(size), unit="micrometer", type="SPACE") for axis, size in zip(SPACE, spacing))

    @property
    def xarray_data(self):
        return xr.DataArray(da.from_array(self.store, chunks=(1, 16, 32, 32)), dims=list("czyx"))

    def axes_of_type(self, axis_type):
        return {"SPACE": SPACE, "CHANNEL": ("c",)}.get(str(getattr(axis_type, "value", axis_type)), ())

    def derive(self, **kwargs):
        return ns(**kwargs)


class StubMikro:
    """Answers what the action asks, and records what it creates."""

    def __init__(self, *edges):
        self.edges = edges
        self.rath = ns(query=self.query)
        self.annotations: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []

    def query(self, document, variables):
        if "RefinedOf" in document:
            # As the server answers it: the shapes made from this one, each with its derivations.
            children = [
                {
                    "id": f"annotation-{i + 1}",
                    "derivedFrom": [
                        {"relation": link.relation.value, "reason": link.reason, "source": {"id": str(link.annotation)}}
                        for link in a.get("derived_from") or []
                    ],
                }
                for i, a in enumerate(self.annotations)
                if any(str(link.annotation) == variables["id"] for link in a.get("derived_from") or [])
            ]
            return ns(data={"annotation": {"collection": {"id": "collection-a"}, "derivedAnnotations": children}})
        return ns(data={"lens": {"dataset": {"name": "Synthetic"}}})

    def get_transformations(self, filters):
        return [edge for edge in self.edges if str(edge.input.id) == str(filters.input)]

    def get_coordinate_graph(self, coordinate_system, max_depth=None):
        return ns(transformations=self.edges)

    def create_annotation(self, **kwargs):
        self.annotations.append(kwargs)
        return ns(id=f"annotation-{len(self.annotations)}", **kwargs)

    def update_annotation(self, id, **changes):
        self.annotations[int(str(id).split("-")[1]) - 1].update(changes)
        return ns(id=id)

    def create_mesh_collection(self, **kwargs):
        self.meshes.append(kwargs)
        number = len(self.meshes)
        return ns(id=f"collection-{number}", lens=lambda: ns(id=f"collection-{number}"))

    def create_table_dataset(self, **kwargs):
        self.tables.append(kwargs)
        return ns(id="table", axis_names=tuple(c.name for c in kwargs["columns"] if c.axis_type == AxisType.INDEX))


def test_two_channels_on_two_grids_read_only_round_the_path():
    synth = core.synthetic()
    # Background round the synthetic volume, so the image is much larger than the path's box.
    pad = ((16, 16), (64, 64), (64, 64))
    ais, cluster = np.pad(synth.ais, pad, mode="edge"), np.pad(synth.cluster, pad, mode="edge")
    rough = synth.rough + np.array([p[0] for p in pad])
    image_a, edge_a = calibration("image-ais", synth.spacing)
    spacing_c = synth.spacing * np.array([1.0, 2.0, 2.0])
    image_c, edge_c = calibration("image-cluster", spacing_c)
    mikro = StubMikro(edge_a, edge_c, ON_SCENE)
    path = ns(
        id="path-1", name="axon", kind="PATH", coordinates=[], coordinate_system=DRAWING,
        vectors=[list(row) for row in rough * synth.spacing],
    )
    ais_lens = StubLens(ais, image_a, "ais", synth.spacing)
    cluster_lens = StubLens(cluster[:, ::2, ::2].copy(), image_c, "cluster", spacing_c)

    *_, first = segment_ais_tube_along_path(path, ais_lens, cluster_lens, mikro=mikro, task=Task.local())

    # Only the chunks round the path were fetched, never the whole image.
    for lens in (ais_lens, cluster_lens):
        assert 0 < lens.store.fetched < 0.5 * lens.store.volume.size, f"{lens.id}: {lens.store.fetched} of {lens.store.volume.size}"
    # And the crop held everything the flight reads: every organelle is found, as on the unpadded volume.
    assert first.count == len(synth.organelles), f"{first.count} found, mean radius {first.mean_radius} µm"
    # The AIS surface and the organelles: meshes in xyz, BY_DIMENSION to the AIS lens.
    assert len(mikro.meshes) == 2
    assert all(m["axes"] == list(XYZ) and m["derived_from"][0].kind == "BY_DIMENSION" for m in mikro.meshes)
    assert set(mikro.tables[0]["data"]["cluster_id"]) == set(range(1, first.count + 1))
    # The organelle table is keyed by cluster_id, identified by the mesh collection.
    key = mikro.tables[0]["columns"][0]
    assert key.name == "cluster_id" and [s.kind for s in key.identified_by] == ["MESH_COLLECTION"]
    assert sorted(c.name for c in mikro.tables[0]["columns"]) == sorted(mikro.tables[0]["data"].columns)
    # The adjusted path is made from the drawn one (REFINED), so the scene shows it over the drawn path's card.
    links = mikro.annotations[0].get("derived_from") or []
    assert [(str(link.annotation), link.relation) for link in links] == [("path-1", AnnotationRelation.REFINED)]

    # A re-run finds the adjusted path by its lineage, not its name, and rewrites it in place.
    mikro.annotations[0]["name"] = "axon · adjusted"  # as runs before plain names left it
    segment_ais_tube_along_path(path, ais_lens, cluster_lens, cluster_threshold_scale=1.1, mikro=mikro, task=Task.local())
    assert len(mikro.annotations) == 1 and mikro.annotations[0]["name"] == "Adjusted path"
    # The tables name the path and both lenses they were measured on.
    kinds = [type(link).__name__ for link in mikro.tables[-1]["derived_from"]]
    assert kinds[0] == "AnnotationDerivedFromInput" and len(kinds) == 3, kinds


def test_an_organelle_channel_without_organelles_yields_none():
    synth = core.synthetic(organelle_contrast=0.0)
    image_a, edge_a = calibration("image-ais", synth.spacing)
    image_c, edge_c = calibration("image-cluster", synth.spacing)
    mikro = StubMikro(edge_a, edge_c, ON_SCENE)
    path = ns(
        id="path-1", name="axon", kind="PATH", coordinates=[], coordinate_system=DRAWING,
        vectors=[list(row) for row in synth.rough * synth.spacing],
    )
    ais_lens = StubLens(synth.ais, image_a, "ais", synth.spacing)
    cluster_lens = StubLens(synth.cluster, image_c, "cluster", synth.spacing)

    _, ais_mesh, organelle_mesh, organelle_table, profile, run = segment_ais_tube_along_path(
        path, ais_lens, cluster_lens, mikro=mikro, task=Task.local()
    )

    # Diffuse axoplasm and noise: the floor, not Otsu, sets the threshold, and nothing passes it.
    assert run.count == 0 and run.cluster_threshold_source == "noise floor", run
    assert ais_mesh is not None and organelle_mesh is None and organelle_table is None and profile is not None
    assert len(mikro.meshes) == 1


def test_the_app_starts_and_offers_its_action(local_runtime: Runtime[None]):
    assert local_runtime.app is app
    assert "segment_ais_tube_along_path" in set(app.registry.implementations)
