# ais-organelles-app

The axon initial segment (AIS) and the cisternal organelle inside it, segmented along
a drawn path. You trace a rough PATH along an axon in the orkestrator 3D scene and run
**Segment AIS Tube Along Path** from the shape's card. You name two channels: the AIS
(AnkyrinG, NaV) and the organelle (synaptopodin, ...).

In parameters and tables the organelle channel is still called the "cluster" channel
(`cluster_lens`, `cluster_id`, `cluster_threshold_scale`).

## How it works

Everything works on voxel masks around the path. Nothing is straightened.

1. **Tube.** The voxels within `diameter_um` / 2 of the drawn path. Each voxel belongs
   to its nearest path sample, which is its row along the path.
2. **AIS.** Otsu on the AIS channel in the tube, per 5 µm along the path, so it follows a
   stain that fades distally. The AIS is what lies above that threshold and touches the
   path, which drops a bright neighbour. Holes are filled, including the dim centre of a
   ring-like stain.
3. **Adjusted path.** The path through each row's AIS centroid, smoothed. It is written
   beside yours as "Adjusted path", and the measurements run along it.
4. **Organelle.** Otsu on the organelle channel inside the AIS. Otsu always splits, so
   with no organelle present it would cut the noise in two. To prevent that, the threshold
   is never below the diffuse level b + 3σ. Both are measured in the AIS core, the inner
   half of each cross-section, away from the blurred edge: b is the histogram's mode, and
   σ is the MAD of the values below it. A piece is kept only if its brightest voxel
   reaches b + 5σ. Pieces within `max_same_distance_um` of each other are one organelle.
   `cluster_threshold_scale` scales the threshold's height above b. The task log says
   whether Otsu or the floor set it.
5. **Measure.** One row per organelle: arclength, extent, volume, intensity, centroid,
   distance from the path and depth inside the AIS. One row per 1 µm bin: AIS volume and
   equivalent radius, organelle volume, fraction, count and density.

The two channels need not share a grid. The organelle lens is read at the same world
points through the drawn space. Only the box round the path, plus the blur's footprint,
is read from each lens.

## Outputs

- the adjusted path (rewritten in place on a re-run, found through its lineage)
- the AIS surface
- the organelles, one mesh object per `cluster_id`
- the per-organelle table (keyed by `cluster_id`, identified by the organelles' mesh
  collection) and the per-bin AIS profile

Names are plain ("AIS", "Organelles", "AIS Profile", "Adjusted path"): what each is of is
its lineage. Every surface derives from the AIS lens and the path; the tables from the path
and both lenses (measured, so without geometry). Both lenses must
be calibrated (a µm voxel size on z, y and x).

## Layout

- `app.py`: the action, reading the crops, the uploads and the tables
- `src/ais_organelles_app/core.py`: the numerics, server-free, with synthetic volumes
  and their checks
- `src/ais_organelles_app/centreline.py`: the path's spline, the blur, the simplification
- `src/ais_organelles_app/placement.py`: placing the path on a lens and writing the
  adjusted path back

## Develop

```sh
uv sync
uv run arkitekt run dev
uv run pytest          # the numerics on synthetic volumes; the action against a stubbed mikro
```
