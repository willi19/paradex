# Articulated Object Capture Pipeline

This folder contains the first offline pipeline for articulated/separable object state capture.

> **Redesign note:** the mask-acquisition layer of this pipeline is planned to move to
> SAM3-generated masks, followed by a staged restructuring of `calc_states.py` into a
> package. See [REDESIGN_PLAN.md](REDESIGN_PLAN.md) for the target architecture, the
> round-by-round migration plan, and what gets deleted vs kept.

## Current Staged Pipeline

The active path for SAM3 sessions is `generate_masks_sam3.py -> register_states.py ->
segment_parts.py`. The legacy `calculate.py` documentation below remains for the older
pipeline; do not mix its segmentation outputs with the staged path.

After mask generation and body registration, run part decomposition as follows:

```bash
python src/dataset_acquisition/articulated_object/segment_parts.py \
  --session-dir ~/shared_data/capture/articulated_object/frame_oak/sam3_centered_01 \
  --mesh-path ~/shared_data/mesh_blender/frame_oak/frame_oak.obj \
  --min-fit-iou 0.25 \
  --support-drop 0.25 \
  --grow-support-drop 0.10
```

Stage C is enabled by default. It now gates each body/part observation before using it:

- body registration IoU must reach `--min-fit-iou` unless
  `--stage-c-min-body-fit-iou` overrides it;
- an individual part residual fit must reach `--stage-c-min-part-fit-iou` (default
  `0.08`);
- a camera that leaves nearly the entire object mask unexplained by the body is
  rejected by `--stage-c-max-residual-object-fraction` (default `0.85`);
- a state whose motion evidence covers nearly the entire mesh is rejected by
  `--stage-c-max-state-motion-fraction` (default `0.75`);
- a vertex needs support from two placement groups by default. Use
  `--stage-c-min-motion-groups 1 --stage-c-min-static-groups 1` only for a
  single-placement capture.

Distance-based mesh-piece bridging is now disabled by default. Do not restore the old
`--stage-c-bridge-dist-frac 0.02` unless the Stage-C mesh diagnostics prove that the
pieces should belong to one part; spatial proximity alone can join frame, hinge, and
unrelated metal geometry.

Stage B now uses `--stage-b-residual-source full_mesh` by default: it subtracts the
entire known rest mesh from the object mask before fitting a moving candidate. This
prevents a static face that Stage A accidentally labeled movable from becoming its own
residual evidence. The legacy Stage-A-body residual is still written beside it for
comparison, under `segmentation/stage_b/residual_overlays_stage_a_body/`. Use
`--stage-b-residual-source stage_a_body` only to reproduce the legacy behavior.

The default Stage-B fit evidence is now `--stage-b-evidence-source hybrid`. It first
computes positive RGB-change maps within each placement group, gates them to the
dilated group-union SAM support, then admits residual pixels only near that image
change. This allows a handle that stays inside the outer silhouette to contribute
evidence, while a SAM3 false negative or an occluded view is unknown rather than a
hard static vote. RGB change within
`--stage-b-image-motion-body-edge-band` pixels (default `4`) of the registered body
silhouette boundary is suppressed: a fixed dark body on light cloth flickers at its
high-contrast edges, and those false thin bands both drag the part-pose fit toward
the frame and hand rim-shaped static patches their "explained" pixels. A real part
sweeps an area, so losing a thin rim of it is cheap. Suppressed pixels are shown as
red in `image_motion_overlays/` and reported as `edge_suppressed` fractions. After a
whole Stage-A candidate supplies a motion-pose hypothesis, `patch_refined` splits
that candidate along welded mesh connectivity and sharp normal changes.

Patch certification is now a JOINT selection
(`--stage-b-patch-score-mode composite_joint`, the default), not a set of independent
per-patch tests. Independent testing cannot reject a static surface that the mover
occludes or reveals: the disocclusion is real image change at that surface's own
footprint, and hinge-adjacent geometry sweeps into the mover's change region under
the shared part transform, so such patches score high precision on their own. The
joint pass composes the z-buffered scene for a patch SUBSET moved together and adds
patches greedily by marginal gain (newly explained RGB change minus
`--stage-b-patch-joint-spurious-weight` times predicted change outside the
evidence). The mover alone already predicts the revealed surfaces' change, so their
marginal gain collapses and they drop out without any duplicate-claim penalty. After
selection the part pose is refit on the selected subset only
(`--stage-b-patch-joint-refit-rounds`, default 1) and the selection repeated under
the new pose. The refit round is ADOPTED only when its joint score (explained minus
weighted spurious pixels, comparable across rounds because the evidence is fixed)
beats the previous round; the subset-vs-evidence fit IoU is size-biased and is only
reported (`subset_fit`), never used for adoption or for downgrading the candidate's
moving status (a downgrade would silently remove the candidate from Stage C).
Every selected patch must still pass the independent recurrence gates:
`--stage-b-patch-min-groups` placement groups (default 2), pixel-weighted
`--stage-b-patch-min-pooled-change-precision` (default `0.12`), and the composite
floor `--stage-b-patch-composite-min-change-pixels` (default `4`) after downscaling.

The default joint mode is now also **cross-candidate**: every patch of every
moving candidate competes in one pool under every candidate's fitted `T_world_part`
track. A patch may choose one track only. Patch sources and pose-track anchors are
separate: all moving candidates contribute source patches, but only candidates whose
Stage-B fit reaches `--stage-b-cross-track-min-fit-ratio` (default `0.90`) of the
best fit can become an independent output track. A lower-fit source patch may still
borrow an admitted track, which is the desired handle-fragment behavior; it cannot
turn an unstable hinge-like fit into another output part. The objective is normalized
within placement groups so a track fitted in more cameras/states does not win only by
having more pixels. Selected faces are labeled by their assigned track ID, and Stage C
consumes only those selected tracks. `--skip-stage-b-cross-candidate-selection`
restores the previous per-candidate joint pass for comparison. This remains generic:
one selected track can absorb several articulated mesh pieces, while distinct,
similarly reliable tracks remain available for separable components. The default
`--stage-b-cross-max-tracks 2` caps only pose anchors and limits runtime growth;
raise it (or use `0`) when testing more separable components.

Cross selection also requires an absolute anchor quality of
`--stage-b-cross-min-track-fit-iou` (default `0.08`) and a final explained RGB
coverage of `--stage-b-cross-min-coverage` (default `0.05`). A low-fit pin that
matches only a tiny piece of a much larger blue unexplained region is therefore a
rejected result, not a movable part. When a broad Stage-A candidate contains both
the frame and the handle, `--stage-b-cross-probe-max-patches N` can independently
fit its `N` largest-area local patches to RGB motion before cross selection. The
report records those probe fits and promotes one only when it improves the broad
candidate by `--stage-b-cross-probe-min-fit-gain` (default `0.01`); this is the
diagnostic for whether a viable handle patch remains before cross composition.
When such a probe supplies a pose anchor, cross refit keeps that anchor patch in
the refit geometry and rejects a refit whose per-state pose drifts farther than
`--stage-b-cross-refit-max-anchor-rotation-deg` (default `15`) or
`--stage-b-cross-refit-max-anchor-translation-frac` (default `0.03` of the mesh
diagonal). This prevents a tiny selected patch from replacing a good handle pose
with an unrelated local RGB match. The JSON report records each round's actual
track fit and anchor drift; `--no-stage-b-cross-refit-anchor-lock` is available
only for the legacy ablation.

Each scored patch is exported to `patch_refined/patches/` (per-patch OBJ plus one
patch-colored PLY per candidate). The cross-candidate selected subset's per-camera
explained/spurious/unexplained maps go to `patch_refined/cross_overlays/`
(green = predicted change confirmed by RGB, red = predicted where nothing changed,
blue = evidence the subset does not explain).
`--stage-b-patch-score-mode composite_depth` keeps the prior independent per-patch
test; `raster_change` and `centroid` remain older ablations. Set
`--stage-b-evidence-source residual` or `--skip-stage-b-patch-refine` only to
reproduce older behavior.

Inspect these outputs after each run:

```text
<session>/processed/segmentation/stage_b/part_observations.json
<session>/processed/segmentation/stage_b/residual_overlays/                  # primary full-mesh residual
<session>/processed/segmentation/stage_b/residual_overlays_stage_a_body/     # legacy comparison residual
<session>/processed/segmentation/stage_b/image_motion_overlays/              # cyan: within-placement RGB change
<session>/processed/segmentation/stage_b/motion_refined/mesh_labeled_motion_refined.ply
<session>/processed/segmentation/stage_b/patch_refined/mesh_labeled_patch_refined.ply # primary Stage-B shell
<session>/processed/segmentation/stage_b/patch_refined/patch_refinement.json
<session>/processed/segmentation/stage_b/patch_refined/patches/                # per-patch OBJs + patch-colored PLY
<session>/processed/segmentation/stage_b/patch_refined/joint_overlays/         # explained/spurious/unexplained maps
<session>/processed/segmentation/stage_c/part_solids.json
<session>/processed/segmentation/stage_c/mesh_labeled_solid.ply
<session>/processed/segmentation_report.html
```

`[STAGEC:state]` lines state exactly why each candidate/state was used or dropped;
`res_obj` is residual area divided by object-mask area, not by full image area.

1. Capture raw synchronized images for each static object state.

```powershell
python src/dataset_acquisition/articulated_object/capture.py --object-name OBJECT_NAME
```

Press `c` for each state, and `q` to finish.

2. Preprocess the latest session with the input mesh.

```powershell
python src/dataset_acquisition/articulated_object/calculate.py --object-name OBJECT_NAME --mesh-path ~/shared_data/mesh_blender/OBJECT_NAME/OBJECT_NAME.obj
```

If `--mesh-path` is omitted, the calculation step infers the input mesh as:

```text
~/shared_data/mesh_blender/OBJECT_NAME/OBJECT_NAME.obj
```

The mesh is an input to calculation, not a capture output. The current calculation stage:

- undistorts raw images into each state `images/` directory
- checks image/camera/input-mesh completeness
- detects and matches multiview image features per state
- triangulates a calibrated sparse point cloud per state
- groups states by object placement (hybrid capture) and, with 2+ placements, recovers per-camera
  median background plates and per-state full object silhouette masks; with
  `--object-mask-source external` (or a SAM3 manifest + `auto`), externally generated masks
  from `generate_masks_sam3.py` replace the plate masks
- carves a per-state visual hull from those masks; the hull surface becomes the preferred
  whole-object registration target
- estimates a whole-object input-mesh registration per state
  - robust object-ROI registration by default: PCA orientation candidates (resolving the
    flat/symmetric axis-flip ambiguity) each refined by iterative object-crop + trimmed ICP,
    so scene background/outliers in the sparse cloud do not drag the fit off the object
  - legacy path (single coarse PCA centroid alignment + optional trimmed ICP on the full
    cloud) is still available with `--registration-object-crop-distance 0`
- segments the whole input mesh into planned canonical part meshes
  - mask-based sparse evidence when part masks exist
  - motion-residual clustering fallback across registered states
- estimates candidate per-part poses when part meshes are provided
- accumulates per-part sparse residual points as hidden-surface completion candidates
- exports part pose tracks and relative part-pair motion observations
- fits candidate kinematic joints from relative part motion
- exports a COLMAP text camera model for later COLMAP/MVS integration
- writes manifests under `<session>/processed/`

Main outputs:

```text
<session>/processed/
  pipeline_manifest.json
  articulated_object_model.json
  states/<state_id>.json
  multiview/<state_id>/
    cameras.json
    pairs.json
    sparse_points.ply
    object_points_roi.ply    # object-only cloud after auto ROI + ground removal (preferred downstream)
    object_roi_removed.ply   # points removed as floor/ground (inspection)
    multiview_manifest.json
    colmap_text/
      cameras.txt
      images.txt
      points3D.txt
  background_plates/
    <serial>.png             # per-camera median over all states = empty background (hybrid capture)
    trust_<serial>.png       # per-camera trust map: cloth-like plate pixels where mask evidence is reliable
    trust_overlay_<serial>.jpg # plate with untrusted region darkened (first cameras only)
  object_masks/
    mask_stats.csv           # per (state, serial) mask stats (plate fractions or SAM scores)
    masks_manifest.json      # present when an external provider (generate_masks_sam3.py) wrote the masks
    <state_id>/
      <serial>.png           # per-view full object silhouette mask (|image - plate| or external SAM3)
      overlay_<serial>.jpg   # [image | mask overlay] inspection panel (first cameras only)
  visual_hull/
    hull_<state_id>.ply      # hull surface voxels from multi-view mask voting (world frame)
  registration/<state_id>/
    registration.json
    coarse_aligned_mesh_sample.ply
    aligned_mesh_sample.ply
  segmentation/
    segmentation.json
    vertex_labels.csv
    parts/
      <part_id>.obj
  part_registration/<state_id>/
    part_registration.json
    <part_id>/
      part_registration.json
      target_masked_sparse_points.ply   # if part masks are available
      target_segmented_sparse_points.ply # if generated/provided part meshes can assign sparse points
      target_whole_sparse_points.ply    # fallback when masks are not available
      initial_aligned_part_mesh_sample.ply
      aligned_part_mesh_sample.ply
  completion/
    completion.json
    parts/<part_id>/
      completion.json
      completion_points.ply
      completed_surfel_mesh.obj
  motion_observations/
    part_pose_tracks.json
    motion_observations.json
    <parent_part_id>__<child_part_id>.json
  kinematic_model/
    kinematic_model.json
    <parent_part_id>__<child_part_id>.json
```

Use `--skip-multiview` when only raw undistortion and metadata checks are needed.
Use `--skip-registration` when sparse point clouds are needed but mesh alignment should be skipped.
Use `--registration-refine-method none` to keep only the coarse PCA initialization.

The registration output records both `coarse_T_world_object` and final `T_world_object`.
When ICP is accepted, final `T_world_object` is the refined transform. When ICP fails or
is rejected, final `T_world_object` falls back to the coarse transform and the reason is
recorded under `refinement` / `warnings`.

Whole-object registration is still point-cloud alignment. Per-part registration can use
mask-filtered sparse points when masks are available, but it is not differentiable
silhouette alignment or final articulated tracking initialization yet.

Automatic background removal front-end (default, runs before registration):

- Multiview triangulation reconstructs the whole scene (object + floor cloth + background +
  photographer + outliers). Before any registration, `_build_object_roi_clouds` cleans every
  per-state cloud to object-only points, fully automatically and independent of registration
  quality, so no known coordinates, per-view masks, or empty-background capture are required.
- World-volume ROI crop: the working-volume center is estimated from camera geometry alone
  (least-squares intersection of the camera optical axes, which converge on the object). This is
  frame-agnostic, so re-calibrating between sessions is fine. Points farther than
  `--roi-radius-factor` x mesh-bbox-diagonal from that center are dropped (removes walls,
  photographer, distant clutter). If the sphere would keep too little, it is skipped (safe no-op).
- Ground-plane removal: the dominant supporting plane (the floor cloth) is fit by RANSAC on the
  ROI points (`--roi-ground-plane-fraction` tolerance). Its normal is oriented toward the camera
  centroid ("up"); plane inliers and the far side are removed, keeping the object side. Removal is
  only applied when the plane is large (`--roi-ground-min-inlier-fraction`) and is a genuine
  supporting plane (`--roi-ground-side-fraction` of off-plane points on the camera side), so a
  large flat object face is not mistaken for the floor. A textureless (unreconstructed) floor is
  simply absent and safely skipped.
- Output: `multiview/<state>/object_points_roi.ply` (object-only cloud) and
  `object_roi_removed.ply` (points removed as ground) plus per-state stats under
  `pipeline_manifest.json > object_roi_crop`. `_find_state_pointcloud` prefers this cloud, so
  registration, canonical loading, and segmentation all consume the cleaned cloud automatically.
- This works when the whole object is re-positioned arbitrarily between states (each state's cloud
  is cleaned independently in its own frame; the mesh is the common anchor at registration).
  Disable with `--roi-crop-mode off` to fall back to the raw whole-scene cloud.

Placement groups + background plates (hybrid capture, default on):

- The recommended hybrid capture places the object at several different spots on the cloth and
  articulates the moving part inside each spot ("placement"), with cameras and cloth fixed for the
  whole session. Two states are only image-diff-comparable when they share a placement, and
  differing placements are exactly what makes background recovery possible.
- `_detect_placement_groups` annotates every state with a `placement_group_id`. Auto mode clusters
  the per-state object-cloud centroids: states within `--placement-group-distance-fraction` (default
  0.35) x mesh-bbox-diagonal share a placement (the moving part shifts the centroid far less than a
  re-position). No state labels are required. Use `--placement-groups '000,001;002,003'` with
  `--placement-group-mode manual` to force groups, or `--placement-group-mode single` for the legacy
  fixed-body behavior. Groups are printed as `[GROUP] ...` and recorded under
  `pipeline_manifest.json > placement_groups`.
- With 2+ placements, `_build_background_plates_and_masks` computes, per camera, the per-pixel
  median over all states = the empty background plate (each pixel is covered by the object in only
  a minority of states). Each state's full object silhouette mask is then
  `|image - plate| >= --object-mask-threshold`, followed by shadow suppression (pixels darker than
  the plate but with the plate's chromaticity, i.e. shadow on the white cloth; tune
  `--object-mask-shadow-*`, disable with `--object-mask-suppress-shadows off`), morphological
  cleanup, and small-component removal. Outputs: `background_plates/<serial>.png`,
  `object_masks/<state>/<serial>.png`, `[image | overlay]` panels for the first
  `--object-mask-overlay-cameras` cameras, and `object_masks/mask_stats.csv`. Prints `[PLATE] ...`.
- Shadow suppression is restricted to pixels whose *plate* looks like the cloth (bright and
  unsaturated: `--object-mask-shadow-plate-min-brightness`, `--object-mask-shadow-plate-max-chroma`).
  Against non-cloth backgrounds (walls, equipment) the darker-but-same-chroma rule would match
  object pixels and punch holes in the mask, so it is simply not applied there. Per-view masks are
  still imperfect off the cloth (a pixel whose object color matches the background behind it can
  never be separated by differencing); the visual hull below absorbs those per-view defects.
- Trust-gated mask evidence (`--mask-trust-mode auto`, default): off-cloth mask regions are not
  just noisy, they are *structurally* unreliable (holes where the object matches the dark room,
  background-structure-shaped blobs where it occludes equipment), and both hull voting and mask
  scoring previously treated those pixels as hard evidence. Each camera now gets a trust map
  (`background_plates/trust_<serial>.png`): plate pixels passing the same cloth-likeness test as
  shadow suppression (`--object-mask-shadow-plate-*`), eroded by `--mask-trust-erode` to keep the
  soft cloth boundary out. Untrusted pixels become "unknown" instead of negative evidence:
  - Hull voting counts a camera's opportunity/vote for a voxel only when it projects onto trusted
    pixels, so an off-cloth mask hole can no longer carve real object voxels (`[HULL]` prints
    `trust=<gated>/<used>`).
  - Registration mask scoring intersects both the object mask and the rendered silhouette with the
    trust map; a camera whose trust-gated mask falls below `--registration-mask-min-trusted-px`
    pixels is dropped from mask scoring entirely (fragmentary masks mis-rank poses). A silhouette
    scored as empty after gating gets the worst finite score instead of dropping out of the mean,
    so poses cannot escape a camera's judgement by leaving the trusted zone.
  - `image_refine_<serial>.png` overlays darken the untrusted region; `mask_stats.csv` gains
    `mask_trusted_fraction` (fraction of each mask on trusted pixels) to identify which cameras
    were judging poses from off-cloth evidence.
  What is deliberately NOT gated: the within-placement image-difference channels (image_diff
  vertex votes, silhouette evidence, moving-part carve) compare two images of the same placement
  directly, so the static background cancels regardless of what it is. Side cameras keep their full
  articulation evidence there — trust gating only limits where *absolute* silhouette claims are
  believed, not where *change* is detected. `--mask-trust-mode off` restores the old behavior. If
  the metal rig passes the cloth-likeness test (bright, unsaturated), check `trust_overlay_*.jpg`
  and tighten `--object-mask-shadow-plate-max-chroma` if its specular reflections cause trouble.

External object masks (SAM3 provider, REDESIGN_PLAN Round 0):

- `generate_masks_sam3.py` (this folder) is a standalone mask provider. It runs in the SAM3
  environment (Python 3.12+, torch; the script is paradex-import-free) and writes the same disk
  contract as the plate path, plus `object_masks/masks_manifest.json` declaring the provider:

  ```bash
  conda run -n sam3 python src/dataset_acquisition/articulated_object/generate_masks_sam3.py \
    --session-dir <capture_root>/<object>/<session> \
    --output-dir <output_dir> \
    --prompt "wooden picture frame"
  ```

  It reads the undistorted `<state>/images/` views (so the masks match the calibration; run the
  calc step once first for a fresh session) and, per image: one PCS text-prompt call returns
  concept instances; the top-scoring instance is kept (`--instance-mode union` ORs everything
  above `--min-score` if SAM ever splits the object); *enclosed* background holes below
  `--fill-holes-max-frac` of the mask area are filled (pinholes vanish; real openings such as
  the fold-out handle gap stay open because they are larger or touch the outside background);
  masks, `[image | mask]` overlay panels, `mask_stats.csv` (with per-mask SAM confidence) and
  the manifest are written. Prompt resolution: `--prompt` > `--prompt-json[object]` > object
  name with underscores as spaces (a convenience that only works for natural noun phrases).
  Machine-specific locations are arguments; only `--sam3-repo-root ~/sam3` has a default.
- ROI-aware generation (default when available): the per-camera "where is the object" box
  comes from the state's *triangulated object cloud*
  (`multiview/<state>/object_points_roi.ply`, left behind by any calc run,
  mask-independent). The cloud is best-effort object-only - for off-center placements the
  calc ROI sphere crop may have been skipped, leaving a scene-wide cloud whose plain bbox
  projects to the whole image (observed) - so the box is *median-anchored* (robust to <50%
  contamination) and *size-bounded by the input-mesh diagonal* from
  `pipeline_manifest.json > input_mesh > mesh_stats` (the object cannot outgrow its own
  mesh), expanded by `--object-box-expand` (default 0.2 of the mesh diagonal, covering
  texture-poor extremities such as the handle), its corners projected with the per-state
  `cam_param/*.json`. The per-state `[SAM3]` line prints the box extent in world units -
  sanity-check it against the object size. The ROI *sphere* under
  `pipeline_manifest.json > object_roi_crop` is only a loose visibility fallback when the
  cloud is missing: it is centered on the camera convergence point - not the object - with
  radius 1.5x the mesh diagonal, so prompting with it makes SAM segment cloth/background
  chunks (observed), and it keeps object-absent views alive because part of the big sphere
  is almost always in frame. Effects per camera:
  - cameras whose projected object box misses the image (or sits behind it) are skipped as
    `object_out_of_view` - no inference, no false positives on views that cannot see the
    object, and the coverage stats stop counting them as SAM failures;
  - when the concept prompt returns several instances, candidates must have
    `--roi-min-overlap` of their pixels inside the projected box (drops rig/cloth
    detections) before top-score/union selection;
  - when the concept prompt finds nothing although the box is visible - the observed failure:
    close-up top views showing only wood grain no longer *look like* a "picture frame", and
    those maskless top views let the visual hull extrude into a pillar - the view is first
    left blank. After every camera in the state has run PCS, the successful SAM masks carve
    a coarse state-local visual hull inside the object box. The projected hull becomes a
    location guide: its tight bbox is combined with `--fallback-prompt` (default
    `"foreground object"`) for one more SAM3 call, and the returned candidate must overlap
    and cover the guide before it is accepted. These recovered masks are recorded as
    `prompt_mode=hull_guided_sam3` in `mask_stats.csv`. The projected hull itself is not used
    as the mask because it is a coarse voxel projection; it only localizes the fallback
    prompt. Overlays draw the used fallback box in magenta; `mask_stats.csv` records
    `box_kind` (object/sphere),
  `fallback_source_masks`, `hull_occupied_voxels`, `hull_projected_voxels`,
  `hull_guide_mask_fraction`, `fallback_prompt`, `hull_guided_overlap`,
  `hull_guided_coverage`, and `hull_rejection_reason`. Without any ROI data the generator
  behaves as plain PCS for the main prompt and simply cannot run the hull-guided retry.
- `--object-mask-source` in the calc step selects the provider: `auto` (default) prefers
  external masks whenever `masks_manifest.json` declares a non-plate provider, `external`
  requires them, `plate` forces the legacy median-plate path. With external masks, plate
  building is skipped and plate trust gating is force-disabled (trust maps describe where
  *plate* content is reliable, which says nothing about a learned mask; stale trust maps from
  an earlier plate run in the same output dir must not gate the new masks). Hull voting and
  mask-scored registration consume the masks unchanged - including on off-cloth placements,
  which is exactly what the plate path could not provide. Prints `[MASKS] external ...`.
- Round 0 validation tip: point both the generator and the calc step at a fresh `--output-dir`
  so the SAM-mask run stays side-by-side comparable with the previous plate-mask run.

Visual hull + hull-target registration (default on when masks exist):

- `_build_visual_hull_clouds` fills each state's camera-derived working volume (ROI sphere, above
  the ground plane) with voxels and keeps those that project inside the object mask in at least
  `--hull-min-view-fraction` (default 0.7) of the cameras that see them (`--hull-min-opportunities`
  minimum, `--hull-grid-resolution` voxels per axis). Majority voting across views is what makes the
  imperfect per-view masks usable: a hole or false blob in one view is vetoed by the other views.
  The hull surface (occupied voxels with an empty 6-neighbor) is written to
  `visual_hull/hull_<state>.ply` in world coordinates; stats print as `[HULL] ...` and live under
  `pipeline_manifest.json > visual_hull`.
- Whole-object registration then uses the hull surface as its target cloud
  (`--registration-target auto`, printed as `target=hull` in `[REG]` lines). Unlike the sparse
  triangulated cluster (planar, texture-biased, ~200 points), the hull is volumetric and covers the
  whole object including textureless parts, so the PCA axes/centroid of target and mesh finally
  correspond and trimmed ICP has in-plane gradients; the moving-part protrusion in the hull also
  breaks the flat-object orientation ambiguity. The mask-scored image refinement still runs
  afterwards as a polish with a now-sane initial pose. Use `--registration-target sparse` to compare
  against the old behavior, and `--registration-hull-min-points` to guard against degenerate hulls.
- Hull bloat and residual pose slack: a visual hull is always a superset of the object, and at the
  default 96 voxels/axis a thin slab is only 1-2 voxels thick, so a thin mesh can tilt ~20 degrees
  inside the bloated hull with little ICP cost (symptom: per-state orientation errors about varying
  axes). Counter it with a finer grid (`--hull-grid-resolution 160`), stricter voting
  (`--hull-min-view-fraction 0.75+`), and `--hull-mask-erode 2` which erodes each mask before voting
  to cancel the systematic blur/close silhouette bloat (mask holes are absorbed by the vote, bloat
  is not). The residual is then removed by the image refinement polish below.
- Multi-round image refinement polish: `--registration-image-refine-rounds 3` turns the candidate
  search into a greedy coordinate descent — each round applies the best improving delta about one
  axis and searches again, so rotations about different axes compose and an initial error about an
  arbitrary axis becomes correctable. `--registration-image-tilt-multiples 3` adds tilt candidates
  at +/-8/16/24 degrees about all three mesh axes (including fine in-plane steps about the normal).
  Applied deltas are printed per round and recorded under `image_refinement.applied_rounds`;
  `candidates.csv` gains a `round` column.

Direct mask-silhouette registration (`--registration-coarse-method mask_silhouette`):

- The hull path reconstructs a 3D visual hull and fits the mesh to it. For a flat, thin, untextured
  object with limited good views this is weak: the visual hull over-estimates exactly the thin
  (thickness) dimension whose axis the orientation depends on, adds phantom blobs and depth smear,
  and the mesh then fits an already-corrupted target (symptom: `target=hull` with hundreds of hull
  points but only ~20 crop/tight inliers, orange and green both off the mask). When the masks are
  good, the stronger use of them is the reverse direction: fit the *known* metric mesh directly to
  the 2D masks (analysis-by-synthesis), never building a hull.
- The method: **translation** from a robust IRLS intersection of the per-view mask-centroid rays
  (a minority of false-positive masks whose centroid rays miss the object are down-weighted out);
  **orientation** from a coarse `SO(3)` grid — `--registration-silhouette-normal-dirs` face-normal
  directions (Fibonacci sphere) x `--registration-silhouette-inplane-steps` in-plane angles — scored
  by a **trimmed-mean** multiview silhouette-vs-mask cost; then a local greedy refine of rotation and
  translation. Scale is fixed (the mesh is metric), so there is no scale ambiguity. `[MASKREG]`
  prints `masks= rays=(in=) cand= coarse= refined=` per state.
- Scoring cost (**chamfer / distance transform**, replaces per-candidate rasterization): the old
  scorer rasterized the whole mesh silhouette (fill-poly over thousands of triangles) *for every*
  candidate pose and view, which dominated the runtime (hundreds of thousands of renders across the
  coarse grid, the local refine, and the group-share pass). Instead, each view precomputes a mask
  distance field once (`cv2.distanceTransform`); a candidate is scored by projecting a fixed set of
  `--registration-silhouette-score-points` (default 2000) mesh surface points and (a) precision = the
  **fraction of points outside the mask** (scale-free, so an inflated or twisted pose that spills
  points out cannot be cheapened by the recall term) plus a small smooth outside-distance term for a
  convergence basin, with a ~1.5 px boundary tolerance so point-vs-rasterized-mask discretization does
  not create a noise floor; and (b) recall = uncovered mask area, from splatting the points into an
  occupancy grid. One point projection plus array lookups per view, no triangle rasterization — orders
  of magnitude cheaper and a smoother objective, so the refine also walks a finer step tail to close
  the last few degrees. Translation placement is unchanged from the previous scorer (mesh vertex-mean
  centroid at the mask-centroid ray intersection); only the orientation/refine scoring changed.
- Robust to the real mask environment (not every mask is a true positive): the multiview cost drops
  the worst `--registration-silhouette-trim-fraction` (default 0.3) of views, so a minority of FP/FN
  masks — and the view of an **articulated part that moved** away from the rest-pose mesh — do not
  drag the body pose. The body dominates the silhouette; the moving part is a small, out-voted
  minority. (Fitting the mesh to the masks per-part to *recover* the articulation directly is a
  natural follow-on, not done here.)
- Falls back to `hull_pca` automatically when a state has fewer than
  `--registration-silhouette-min-cameras` (default 3) usable masks. It does not need
  `--registration-image-refine` (that flag drives the hull path's polish); the coarse+refine here is
  self-contained and writes the same `image_refine/` overlays (orange = coarse-best, green =
  refined, yellow = mask boundary). Cost knobs: `--registration-silhouette-coarse-cameras`,
  `--registration-silhouette-coarse-downscale`, `--registration-silhouette-refine-cameras`,
  `--registration-silhouette-refine-seeds`, `--registration-silhouette-refine-rounds`,
  `--registration-silhouette-score-points`.
- Placement-group shared orientation (`--registration-group-share-pose`, default on for
  `mask_silhouette`): a single flat, near-square state can still lock onto a 90-degree in-plane flip
  — the object body is nearly symmetric and the one asymmetric feature (the articulated part) has
  *moved* away from the rest-pose mesh in some joint states, so per-view precision actively favors
  the wrong flip there. But within a placement the **body pose is identical across the group's states**
  (only the part moves), so orientation is a single shared unknown. After the per-state fits, this
  post-pass takes each group state's orientation plus its +/-90/180-degree in-plane variants as
  candidates and scores every candidate against the **whole group's pooled masks** (worst views
  trimmed). Scoring uses each state's existing translation with the cheap chamfer cost — no
  per-candidate translation refine (the old pass re-ran a full greedy refine for every candidate,
  which dominated `[GROUPREG]` runtime; the discrete 0/90/180/270-degree choice does not need it) —
  and only the adopted group-best orientation gets one final per-state translation refine. The
  group-best orientation — resolved by the state whose part *does* match the mesh and by pooling three
  states' body evidence — is adopted by every state (status `mask_silhouette_group_shared`, prints
  `[GROUPREG] group=...`, records
  `group_shared_pose` in `registration.json` with the pre-share pose). This is the payoff of the
  hybrid multi-placement capture. Disable with `--no-registration-group-share-pose`; it is a no-op
  under `--placement-group-mode single` (one lumped group would wrongly mix placements). Recovering
  the articulated part itself by the per-state mesh-vs-mask *mismatch* is a natural follow-on.
- These masks are texture-independent full silhouettes in every view (not just top-down ones) and
  need no registration, no empty-background capture, and no manual annotation. They are the primary
  evidence for mask-scored registration below; later stages (visual hull, hull-difference part
  evidence) can consume them too. With a single placement the stage safely skips itself (a median
  over one placement would still contain the object) — that is the one thing the old fixed-body
  sessions cannot provide.
- All state-to-reference image-difference channels (image_diff segmentation evidence, silhouette
  evidence, moving-part carve) now only compare states inside the same placement group; cross-
  placement states are recorded as `skipped_cross_placement`. The carve runs once per placement
  group (per-group reference and ROI volume) and unions all groups into `carved_all.ply`; per-group
  stats live under `pipeline_manifest.json > moving_part_carve > groups`.

Robust object-ROI registration (default):

- Multiview triangulation reconstructs the whole scene, so the per-state sparse cloud mixes
  object points with table/background/outlier points. Plain PCA + ICP on that full cloud is
  dragged off the object (loose fit, mirrored/rotated pose for flat symmetric objects).
- The default path instead enumerates up to four PCA orientation candidates (principal-axis
  sign flips) and, for each, iterates: crop the target to points within
  `--registration-object-crop-distance` (fraction of the mesh bbox diagonal) of the currently
  aligned mesh surface, then run trimmed ICP on that object-only crop, shrinking the crop over
  `--registration-crop-iterations`. The candidate with the most tight object inliers
  (within `--registration-object-tight-fraction` of the surface) is selected.
- `registration/<state>/registration.json` records the selected candidate, per-candidate
  reports, and an `object_registration` block (`object_point_count`, `tight_inlier_count`,
  `median_object_surfdist_fraction`, `p90_object_surfdist_fraction`) computed on object points
  only, so registration quality is no longer diluted by background points.
- `registration/<state>/object_sparse_points.ply` exports the ROI-cropped object points for
  inspection. Inspect `sparse_vs_aligned_mesh_overlay.ply` to confirm the mesh (blue) hugs the
  sparse object (white); a good fit has `median_object_surfdist_fraction` around 0.02-0.05.
- Set `--registration-object-crop-distance 0` to fall back to the legacy full-cloud path.

Image-space registration refinement (optional):

- `--registration-image-refine` keeps sparse/ROI registration as the initial pose, then scores
  candidate poses by rendering the mesh silhouette into each single camera image and measuring how
  close its boundary lies to image edges. This is a single-frame score, not a reference-current
  image-difference score, so it is the registration path that can later support object-rotated
  capture and separable-object reconstruction.
- Score channel (`--registration-image-score-mode`, default `auto`): when background-plate object
  masks exist for the state (hybrid capture, 2+ placements), candidates are scored against the real
  silhouette instead of Canny edges: `precision` = fraction of the rendered silhouette inside the
  object mask, `recall` = fraction of the mask covered, combined as
  `--registration-mask-precision-weight * (1-precision) + --registration-mask-recall-weight *
  (1-recall)` (precision-weighted so residual shadow or the moving part in the mask does not drag
  the body). This is far more discriminative for in-plane rotation/flips than edge distance, which
  is clutter-dominated on the cloth/shadow/internal-picture edges. Scoring runs at
  `--registration-image-mask-downscale` (default 2) for speed; only mask-capable cameras are scored
  so the score scale stays consistent. With trust gating (default) both the mask and the rendered
  silhouette are intersected with the camera's trust map first, and cameras whose trusted mask is
  below `--registration-mask-min-trusted-px` are dropped (see the trust-gated mask evidence bullet
  above). `candidates.csv` gains a `score_channel` column; `image_refine_<serial>.png` overlays draw
  the object-mask boundary in yellow and darken the untrusted image region. `mask` forces the
  channel (falls back to edge with a warning when no masks exist); `edge` keeps the old behavior.
- The candidate set includes rotation about the mesh thin/normal axis
  (`--registration-image-normal-step-deg`), small broad-axis tilts
  (`--registration-image-tilt-deg`), and optional 180-degree front/back flips
  (`--no-registration-image-frontback-flips` disables those flips). The edge score uses only the
  closest fraction of silhouette-boundary pixels (`--registration-image-edge-trim-fraction`) so
  articulated or missing mesh regions can behave as outliers instead of forcing the whole mesh to
  fit the wrong pose.
- If both the initial and refined silhouettes miss the object by a visible offset, enable the
  object-local translation/scale search with `--registration-image-translation-fraction`,
  `--registration-image-translation-steps`, and `--registration-image-scale-sweep`. This handles
  initial poses whose orientation is plausible but whose projected center/depth/scale is wrong.
- By default the image score is not computed against the full image. It first projects the current
  object/ROI sparse points into each camera (`--registration-image-roi-mode projected_sparse`) and
  scores silhouette edges only inside that image ROI, with bbox center/size penalties
  (`--registration-image-bbox-weight`). In `image_refine_<serial>.png`, magenta is this projected
  sparse ROI, orange is the initial silhouette, and green is the best/refined silhouette. If magenta
  does not cover the real object, registration is being driven by the wrong sparse evidence before
  silhouette refinement starts.
- If the best candidate improves the initial score by
  `--registration-image-acceptance-ratio`, the per-state `T_world_object` is replaced. Diagnostics
  are written to `registration/<state>/image_refine/candidates.csv` plus
  `image_refine_<serial>.png`, where orange is the initial silhouette and green is the accepted/best
  candidate. These images are also embedded in `visual_debug_report.html`.

Mesh segmentation:

- `--segmentation-method auto` first tries explicit mask evidence, then moving-region evidence,
  then falls back to motion-residual clustering.
- `--segmentation-method mask_points` assigns mesh vertices to the nearest part's mask-filtered
  sparse points in the reference state.
- `--segmentation-method moving_evidence` uses already captured articulation states to find
  mesh vertices whose sparse-observation residual changes across states. It also writes
  `processed/segmentation/moving_evidence_points.ply` when state-to-state sparse moving points
  can be extracted. The moving region is treated as a seed, not as the final part: candidates are
  grown along mesh connectivity and then rescored by whether a separate rigid part transform
  improves the per-state residual. Geometry proposals are used only as auxiliary candidates when
  they overlap this moving evidence.
  Use `--segmentation-moving-evidence-source image_diff` to replace sparse 3D residual scores with
  per-camera image-difference votes on canonical mesh vertices, or `combined` to take the maximum
  of sparse and image scores. Image evidence is still a segmentation seed: it does not reconstruct
  hidden surfaces by itself.

Road S: silhouette-based moving-part evidence (for textureless parts):

- Sparse triangulation is starved on textureless parts (a flat flush handle yields ~0 SIFT points),
  so the per-state object cloud (~a couple hundred points) is enough for whole-object pose but not
  for part segmentation. Road S instead uses the sparse pose only as an anchor and extracts the
  dense moving-part signal from image silhouettes, which are texture-independent.
- `--enable-silhouette-evidence` (stage 1, validation) renders the registered mesh body silhouette
  into each view (filled-triangle coverage mask, no per-pixel depth needed) and subtracts it from
  the state-to-reference image change. The remainder (`change and not body`) is the moving/separated
  part in image space: the body is masked out by geometry (no background control needed), and this
  stage is intended for fixed-body captures where the object body stays in place and only the
  articulated/separable part changes. If the whole object is moved between captures, use
  `--registration-image-refine` to estimate each frame's body pose first; the later reconstruction
  path should accumulate pose-normalized evidence instead of raw reference-current image
  differences. Overlays are written to
  `processed/silhouette_evidence/<state>/silhouette_<serial>.png` as `[current | overlay]` where the
  overlay tints the body grey, the raw change red, and the beyond-body part green.
  `--silhouette-body-dilate-iters` grows the body mask margin; `--silhouette-max-cameras` limits how
  many views are exported. Per-view `beyond_fraction_of_change` is reported under
  `pipeline_manifest.json > silhouette_evidence`.
- Stage 2 (next) carves these per-view beyond-body masks across views into a 3D moving-part cloud
  (visual hull within the ROI, above the ground plane so cast shadow drops out), and stage 3 fits a
  revolute/rigid part transform mapping a mesh slab onto that carved cloud (segmentation + joint +
  hidden-surface reconstruction). Shadow on the floor is rejected geometrically at the carve, not in
  the 2D overlay, so some green shadow in stage-1 overlays is expected and harmless.
- Registration-orientation caveat: the stage-1 overlays double as a registration check. For a flat,
  weakly textured object the sparse cloud (~200 points, clustered on the textured face) under-
  constrains the in-plane rotation, so whole-object registration can pick a wrong orientation while
  the crop-to-mesh `object_surfdist` metric stays low (self-fulfilling). If the rendered body (grey)
  is skewed relative to the actual object, the pose is wrong and every downstream signal is poisoned.
  `--silhouette-orientation-sweep` rotates the mesh about its slab normal through a full turn and
  scores silhouette-outline vs image-edge agreement (Canny distance transform), writing per-angle
  `silhouette_orientation/<state>/sweep_<serial>.csv` plus `sweep_current_<serial>.png` (0 deg) and
  `sweep_best_<serial>.png`. A clear best angle that aligns the silhouette means the error was only
  in-plane (recoverable by edge-scored orientation refinement); no aligning angle means the slab
  plane itself is tilted and a fuller pose search is needed.

Pose-free moving-part carve (registration-free, for fixed-body captures):

- When whole-object registration is unreliable (a flat, weakly textured object under-constrains the
  6-DOF pose from ~200 clustered sparse points), the moving part can still be recovered in 3D without
  any registration or object masks, as long as the body and cameras are fixed and only the part moves.
- `--enable-moving-part-carve` fills the camera-derived working volume (`object_roi_crop` ROI sphere,
  above the fitted ground plane) with voxels and keeps the voxels that project into the state-to-
  reference image-change mask (`_image_motion_diff_mask`) in at least `--carve-min-view-fraction` of
  the viewing cameras (`--carve-min-opportunities` minimum). Shadows land on the floor and are dropped
  by the above-ground constraint; background and photographer fall outside the ROI. `--carve-grid-
  resolution` sets voxels per axis.
- Output `processed/moving_part_carve/carved_<state>.ply` (per state) and `carved_all.ply` (union) in
  world coordinates. Because these share the world frame with `multiview/<state>/object_points_roi.ply`,
  the carve can be validated by overlaying the two without any registration: the carved cloud should
  sit on the object where the moving part rests and where it is displaced. This becomes the dense,
  texture-free moving-part observation that later stages fit a joint to and reconstruct.
- Hybrid sessions: the carve runs independently per placement group (its own in-group reference
  state and ROI volume), because image change against a reference is only meaningful while the body
  stayed in place. Per-group stats are under `moving_part_carve > groups`; `carved_all.ply` unions
  every group.
- `--segmentation-method geometry_proposals` searches for generic part candidates from disconnected
  components, small connected PCA-tail protrusions, and connected high-residual regions. Candidates
  are scored by mesh size, narrow cut boundary, sparse observation coverage, and per-state residual
  evidence. This is mostly a diagnostic/fallback mode; standalone disconnected components can be
  false positives when an unused mesh fragment is present.
- `--segmentation-method motion_residual` clusters mesh vertices by their per-state nearest
  observation residual after each state point cloud is mapped back into the canonical object frame.
- generated part meshes are written to `processed/segmentation/parts/<part_id>.obj`.
- after segmentation, small disconnected label islands are reassigned to adjacent labels by default.
  This reduces stray body fragments in generated part meshes without forcing each part to be a
  single connected component. Use `--skip-segmentation-cleanup` to inspect raw labels.
- segmentation graph construction virtually welds exact/near-duplicate vertices by default with
  `--segmentation-adjacency-weld-tolerance 1e-6`. The original mesh is preserved; only adjacency
  used for connected components, proposal boundaries, and cleanup is augmented. Use `0` for exact
  duplicate coordinates only, or a negative value to disable this behavior.

The generated part meshes are canonical-frame meshes cut from the original input mesh. If no
`--part-mesh-paths` are provided, these generated meshes are also used as the per-part
registration inputs later in the same calculation run.

For small moving/protruding parts that exist in the original mesh but are easy to miss in sparse
reconstruction, start with moving evidence and stop after segmentation while tuning:

```powershell
python src/dataset_acquisition/articulated_object/calculate.py `
  --object-name OBJECT_NAME `
  --mesh-path ~/shared_data/mesh_blender/OBJECT_NAME/OBJECT_NAME.obj `
  --min-parts 2 `
  --max-parts 2 `
  --part-labels body moving_part `
  --segmentation-method moving_evidence `
  --segmentation-only
```

Object-only sparse cropping (default): before motion-residual and moving-evidence are computed,
each state's sparse cloud is inverse-registered into the canonical frame and cropped to points
within `--segmentation-object-crop-distance` (fraction of the mesh bbox diagonal, default 0.06)
of the registered mesh surface. Multiview triangulation reconstructs the whole scene, so this
removes table/background/outlier points that otherwise dominate the residual and flag the whole
scene as "moving". Because a moved part's displaced observations fall outside this shell, the crop
also sharpens the signal: a static neighbor keeps a nearby observation in every state (low motion),
while a moved part's rest-pose observation disappears once it moves (high motion / disappeared
evidence). The applied crop counts are recorded per state under `moving_points.states[*]` and the
motion-residual `states[*].object_crop`. Set `--segmentation-object-crop-distance 0` to disable and
use the full per-state cloud. This crop relies on tight per-state registration; verify the
`object_registration` quality in `registration/<state>/registration.json` first.

Useful moving-evidence knobs are `--segmentation-moving-min-score`,
`--segmentation-moving-min-evidence-fraction`,
`--segmentation-moving-min-observed-fraction`, and
`--segmentation-moving-geometry-overlap-fraction`. For thin handles where only an edge or strip is
detected, tune `--segmentation-moving-seed-grow-rings`,
`--segmentation-moving-seed-relax-steps`, and
`--segmentation-moving-seed-relaxed-score-fraction` before widening the generic proposal range.
`--segmentation-moving-raw-seed-penalty` discourages accepting only the thin high-motion seed as
the final part; set it to `0` when the true moving part is itself extremely small.
`--segmentation-moving-two-body-weight` controls how strongly candidates are preferred when they
explain the states better as a separate rigid part. Raise the evidence/overlap thresholds when a
static disconnected mesh fragment is selected instead of the moving part; lower them when the
moving part is very sparse.

When sparse multiview points are too few or biased, try image-space motion evidence:

```powershell
python src/dataset_acquisition/articulated_object/calculate.py `
  --object-name OBJECT_NAME `
  --mesh-path ~/shared_data/mesh_blender/OBJECT_NAME/OBJECT_NAME.obj `
  --min-parts 2 `
  --max-parts 2 `
  --part-labels body moving_part `
  --segmentation-method moving_evidence `
  --segmentation-moving-evidence-source image_diff `
  --segmentation-only
```

This compares each non-reference state image against the reference state for the same camera serial,
thresholds the image difference, projects canonical mesh vertices into those views using the current
whole-object registration, and accumulates a per-vertex image-motion score. Diagnostics are written
to `processed/segmentation/diagnostics/image_motion_vertex_scores.csv`,
`image_motion_camera_comparisons.csv`, and `image_motion_seed_vertices.ply`.

The current image-difference mode is intended for the near-term fixed-body capture case, where the
main body stays roughly fixed and the articulated part moves. The data path is deliberately
canonical-frame vertex voting, so later pose-compensated image residuals, silhouette evidence, and
dense reconstruction observations can be added without changing the downstream part proposal/scoring
interface. Hidden-surface completion still needs a separate dense/MVS/silhouette reconstruction
stage; image motion only helps decide which canonical part should receive that evidence.

The default `--segmentation-relation-prior articulated` penalizes disconnected moving candidates
unless they have direct moving-seed support. Use `--segmentation-relation-prior separable` for
detached multi-part objects where an independently moving disconnected component should be valid.

Useful tuning knobs are `--segmentation-proposal-min-vertex-fraction`,
`--segmentation-proposal-max-vertex-fraction`, `--segmentation-proposal-max-boundary-fraction`,
`--segmentation-proposal-min-observed-fraction`, and `--segmentation-proposal-min-score`.
These affect `geometry_proposals` directly and also the auxiliary geometry candidates used inside
`moving_evidence`.

`processed/segmentation/segmentation.json` records `diagnostics.label_cleanup` and
`diagnostics.component_stats`. These are the first files to inspect when a result splits the
main body instead of a small moving/support part. For unwelded OBJ files whose triangles duplicate
vertices, also check `diagnostics.proposal_generation.adjacency.weld.virtual_edge_count`.

Sparse multiview diagnostics are written alongside normal outputs:

- `processed/visual_debug_report.html` is the first visual inspection entry point. It links
  segmentation overview images, multiview heatmaps, registration overlays, silhouette overlays,
  carve outputs, and top proposal meshes from one static page.
- `processed/multiview/<state_id>/pairs.csv` lists keypoints, matches, RANSAC inliers, and
  triangulated points for every camera pair.
- `processed/multiview/<state_id>/pairs_ranked.csv` sorts camera pairs by triangulated point count
  so the useful and dead camera pairs are easy to spot.
- `processed/multiview/<state_id>/triangulation_heatmap.csv` is a camera-pair matrix of
  triangulated point counts.
- `processed/multiview/<state_id>/triangulation_heatmap.png` and `ransac_inlier_heatmap.png`
  show the same pair quality visually.
- `processed/multiview/multiview_diagnostics.csv` summarizes point and pair counts by state.
- `processed/registration/<state_id>/sparse_vs_aligned_mesh_overlay.ply` overlays sparse points
  and the registered mesh sample for visual inspection.
- `processed/segmentation/diagnostics/segmented_parts_overview_xy.png`,
  `segmented_parts_overview_xz.png`, and `segmented_parts_overview_yz.png` render the whole
  segmented mesh and each part side by side.
- `processed/segmentation/diagnostics/segmented_mesh_colored.ply` colors the original mesh by
  assigned part label, and `segmented_parts_side_by_side.ply` offsets the parts for 3D inspection.
- `processed/segmentation/diagnostics/segmentation_part_summary.csv` lists part sizes, component
  counts, mesh paths, and warnings.
- `processed/segmentation/diagnostics/moving_evidence_mesh_near_points.ply` and
  `moving_evidence_mesh_far_points.ply` split moving evidence by distance to the canonical mesh.
- `processed/segmentation/diagnostics/image_motion_vertex_scores.csv`,
  `image_motion_camera_comparisons.csv`, and `image_motion_seed_vertices.ply` summarize image-space
  motion votes when `--segmentation-moving-evidence-source image_diff` or `combined` is enabled.
- `processed/segmentation/proposals/rank_*.obj` exports top scored part proposals. Set
  `--segmentation-diagnostic-top-proposals 0` to disable this export.

Part output contract:

- `--max-parts N` records the maximum allowed part hypotheses for later automatic segmentation.
- `--min-parts N` records the minimum required part slots in the output model.
- `--part-labels base lid handle` creates named required part slots. If fewer labels than
  `--min-parts` are provided, generic slots fill the remainder.
- `--part-mesh-paths ...` can provide existing part meshes. The planned part count grows to
  at least this mesh count.
- If the number of provided meshes equals the planned part count, segmentation is skipped by
  default and those meshes are used directly for relation/joint estimation.
- If fewer meshes are provided than the planned part count, the remaining slots stay
  `pending_segmentation` until segmentation fills them from the input/composite mesh.

When `--part-mesh-paths ...` are provided but `--mesh-path` is missing, the calculation step
generates `processed/multi_mesh/composite_input_mesh.obj` by concatenating the provided part
meshes. This composite mesh is used only as the whole-object registration target frame. The
individual provided meshes remain the canonical part meshes.

If part slots are forced but part meshes are not available yet, the model writes those parts
with `status: pending_segmentation`. This keeps the downstream tracking/visualization schema
stable without pretending that mesh segmentation has already been solved.

When `--part-mesh-paths ...` is provided, the calculation step also writes candidate
per-part pose observations under `processed/part_registration/`. The initial pose for each
part is the whole-object `T_world_object` composed with that part's `canonical_T_part`.
Currently `canonical_T_part` defaults to identity, so the expected case is a part mesh whose
vertices are already expressed in the original object canonical frame.

Useful multi-mesh modes:

```powershell
# Two provided part meshes -> preserve both meshes and estimate their relation.
python src/dataset_acquisition/articulated_object/calculate.py `
  --object-name OBJECT_NAME `
  --part-mesh-paths base.obj lid.obj `
  --part-labels base lid

# Two provided part meshes -> request one extra slot from segmentation.
python src/dataset_acquisition/articulated_object/calculate.py `
  --object-name OBJECT_NAME `
  --part-mesh-paths base.obj moving_group.obj `
  --min-parts 3 `
  --part-labels base moving_group handle
```

Use `--force-segmentation-with-provided-parts` when provided meshes are complete but you still
want to re-segment the input/composite mesh for diagnostics.

Part registration target selection:

- default mask location: `<state>/masks/<part_id>/<serial>.png`
- label alias also works: `<state>/masks/<label>/<serial>.png`
- `--part-mask-root ROOT` also searches:
  - `ROOT/<state_id>/<part_id>/<serial>.png`
  - `ROOT/<state_id>/<label>/<serial>.png`
  - `ROOT/<part_id>/<serial>.png`
  - `ROOT/<label>/<serial>.png`

When masks are found, sparse 3D points are projected into calibrated camera views and kept
when enough masks contain the projection. Use `--part-mask-min-views`, `--part-mask-threshold`,
and `--part-mask-min-points` to tune this. Use `--require-part-masks` to fail per-part
registration instead of falling back to the whole-state sparse point cloud.

If masks are not available, sparse points are assigned to the nearest generated/provided
canonical part mesh and written as `target_segmented_sparse_points.ply`. Use
`--skip-segmentation-targets` to disable this, or tune it with `--part-target-sample-count`,
`--part-target-min-points`, and `--part-target-max-distance`.

Per-part results should still be treated as pose candidates until cross-state motion
consistency is checked.

Hidden-surface completion:

- completion runs after per-part registration
- per state, the part target point cloud is transformed back into that part's canonical frame
- points already explained by the current part mesh are rejected using
  `--completion-min-surface-distance`
- the remaining sparse residual points are voxel-merged and exported as
  `completion_points.ply`
- `completed_surfel_mesh.obj` contains the original part mesh plus small triangular surfels
  at the completion points

By default, completion uses only `part_mask_sparse_pointcloud` and
`segmented_mesh_sparse_pointcloud` targets. Whole-state fallback targets are ignored because
they mix points from all parts; use `--completion-allow-whole-targets` only for debugging.

This is sparse hidden-surface completion, not dense MVS/TSDF fusion or a watertight mesh
reconstruction. If a hidden surface becomes visible and feature-matched in at least one
captured state, it can appear in `completion_points.ply` and in the surfel OBJ candidate.
If it is textureless, never matched, or not assigned to the right part target, it will not
be reconstructed by this stage.

Motion observation export:

- `part_pose_tracks.json` stores each part's `T_world_part` trajectory by state.
- `motion_observations.json` stores pairwise relative transforms
  `T_parent_child(state) = inv(T_world_parent) @ T_world_child`.
- Each pair file also stores `delta_from_reference_T` against a reference state.

These files are the intended input for later revolute/prismatic/separable joint fitting.
Use `--motion-reference-state STATE_ID` to choose the reference state, and
`--motion-min-states N` to control when a part pair is marked usable.

Kinematic model export:

- `kinematic_model.json` contains generated part meshes and candidate edges.
- Candidate edge types are `fixed`, `revolute`, `prismatic`, with an optional
  `separable_candidate` flag when both large rotation and large translation are observed.
- Revolute candidates store `axis_parent`, `pivot_parent`, and angular limits in radians.
- Prismatic candidates store `axis_parent` and translation limits.

This is the final offline output for this repo. Runtime tracking should consume the exported
part meshes plus `kinematic_model.json`, then run either per-part 6D tracking with constraint
projection or a joint-space tracker parameterized by base pose plus joint state. Tracking code
is intentionally not implemented here.

End-to-end algorithm implemented here:

1. Capture multiple static articulation states with synchronized calibrated multiview images.
2. Undistort raw images and build per-state calibrated sparse point clouds.
3. Register the original whole mesh into each state.
4. Segment the original whole mesh into canonical part OBJ files.
5. Assign each state's sparse points to part targets using masks when available, otherwise
   generated/provided part meshes.
6. Register each part mesh per state and export part pose observations.
7. Accumulate sparse residual observations in each part's canonical frame as hidden-surface
   completion candidates.
8. Convert part pose observations into relative motion trajectories.
9. Fit a candidate kinematic model with part meshes and joint parameters.

Tracking methodology outside this repo:

- **Per-part tracking + projection**: run a model-based 6D tracker independently for each
  exported part mesh, then project the poses onto the exported joint constraints.
- **Joint-space tracking**: track a compact state consisting of base pose plus joint values,
  rendering all part meshes through `kinematic_model.json` and optimizing against multiview
  image/mask evidence.
- **Separable case**: when `separable_candidate` is true, allow the child part to temporarily
  leave the articulated graph and be tracked as an independent rigid body.

Example:

```powershell
python src/dataset_acquisition/articulated_object/calculate.py `
  --object-name OBJECT_NAME `
  --mesh-path ~/shared_data/mesh_blender/OBJECT_NAME/OBJECT_NAME.obj `
  --max-parts 3 `
  --min-parts 2 `
  --part-labels base lid handle
```
