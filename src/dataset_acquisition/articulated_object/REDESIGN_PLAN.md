# Articulated Object Pipeline Redesign Plan (SAM3 era)

Drafted 2026-07. Status: Round 0 implemented (`generate_masks_sam3.py` + calc
`--object-mask-source`); its validation run on `hybrid_01` is pending. Prompt
`"wooden picture frame"` was hand-tested on hybrid_01 images: SAM3 returns the whole
object including the handle as a single instance across top/front/back views, with
scattered small holes (absorbed by vote-ratio hull fusion + enclosed-hole filling).

## 1. Context and goal

The downstream task is 6D object tracking on real manipulation videos that are too
occluded to derive object structure from. This pipeline is the *clean-capture
preprocessing* that makes that tracking viable: photograph each object in a controlled
multiview rig in several states, and produce the object's structural model that the
tracker then consumes.

End deliverable per object (the "handoff contract", see section 7):

- object state class: rigid / articulated / separable
- part decomposition of the base mesh (per-part meshes, canonical frame)
- kinematic model (joint type, axis, limits observed across states)
- surfaces not present in the base mesh (e.g. interior faces revealed when opened)
- per-capture-state 6D poses of every part (doubles as evaluation ground truth)

The current `calc_states.py` (~10k lines) implements this end to end but has accreted a
large *mask-defense layer* (background plates, shadow suppression, trust gating,
mask-score polish rounds, hull erosion) whose only reason to exist is that classical
plate-differencing masks are unreliable off the white cloth. The lab already runs SAM3
in two places: on the processing PC (`object_6d` conda env, used by
`tracking/run/run_video.py --sam3-seed-tracking` with `--target_class <noun>` prompts)
and locally on the project PC (repo at `~/sam3`, conda env `sam3`, checkpoint in the
local HF cache) — so mask generation and calc run on one machine. SAM3 masks make most
of the defense layer unnecessary and are the natural seam for restructuring the code.

## 2. Design principles

1. **Masks are a provider interface, not a pipeline stage.** Anything that writes the
   existing layout `object_masks/<state_id>/<serial>.png` is a valid provider. SAM3
   becomes the primary provider; plate-diff is demoted to fallback and verifier. The
   consumers (hull, registration scoring) do not know or care who made the masks.
2. **calc stays Python 3.8 and torch-free.** SAM3 runs in its own env on whichever
   machine has it; interchange is plain PNG/CSV/JSON under `shared_data`. No paradex
   imports inside the SAM script either, so it runs standalone on the processing PC.
3. **Validate before restructuring.** SAM3 masks must first prove themselves through
   the *existing* pipeline (Round 0). Only after pose converges do we start deleting
   defense code and splitting modules.
4. **Every refactor round is behavior-preserving and regression-gated.** A golden-run
   harness (manifest + key-output comparison on `frame_oak_hybrid/hybrid_01`) gates
   each extraction step.
5. **Motion discovers parts; SAM only draws their boundaries.** Part discovery remains
   a cross-state comparison problem (within-placement diffs / mask XOR). SAM3 is
   instance-level by design (atomic noun phrases; unreliable for unnamed parts), so it
   is used to produce clean whole-object masks and to *clean up* motion-indicated part
   regions, never to name parts on its own.
6. **Public-repo hygiene.** No lab-internal absolute paths, hostnames, or usernames in
   code or defaults; machine-specific locations come in as arguments. The one allowed
   convenience default is `--sam3-repo-root ~/sam3`.

## 3. Target architecture

```
articulated_object/
  capture_states.py            # capture (unchanged)
  calculate.py                 # thin driver -> pipeline runner
  generate_masks_sam3.py       # NEW, standalone, runs in the SAM3 env (section 5)
  pipeline/
    config.py                  # argparse -> typed config, stage toggles
    runner.py                  # stage orchestration + pipeline_manifest.json
    io_bundle.py               # camera bundles, image IO, PLY/CSV/JSON helpers
    scene.py                   # triangulation, ROI sphere, ground plane
    grouping.py                # placement groups (auto/manual)
    masks.py                   # mask providers: external(SAM3) | plate | auto; verifier
    hull.py                    # per-state visual hull from masks
    registration.py            # pose providers: hull+PCA/ICP + mask refine | external import
    motion.py                  # within-placement image diff, mask XOR, silhouette evidence, carve
    parts.py                   # evidence fusion -> mesh vertex labels -> part meshes
    kinematics.py              # part pose tracks -> joint type/axis/limits, state class
    completion.py              # residual hull -> unseen-surface reconstruction
    report.py                  # visual debug report, exports, handoff bundle
```

Data flow: `masks -> hull -> registration` (per state), then
`grouping -> motion -> parts -> kinematics -> completion -> handoff` (across states).
`calc_states.py` remains as a compatibility shim during migration and is deleted at the
end of Round 2.

## 4. Mask provider contract

Layout (identical to today, so both providers are interchangeable):

- `object_masks/<state_id>/<serial>.png` — 8-bit, 255 = object, full image resolution
- `object_masks/mask_stats.csv` — per (state, serial): provider, mask_fraction,
  confidence (SAM score or plate-diff proxy), agreement_iou (verifier, when available)
- `object_masks/masks_manifest.json` — provider name, prompt used, model/checkpoint id,
  generation timestamp, per-state camera coverage
- `object_masks/<state_id>/overlay_<serial>.jpg` — [image | mask] inspection panels
  (first N cameras)

`--object-mask-source {auto,plate,external}` in calc: `external` consumes whatever is
in `object_masks/` and skips plate building; `plate` forces the legacy path; `auto`
prefers external when a manifest declares it, else builds plates. When the source is
external, plate-based defense flags (trust gating, shadow suppression) are inert; if
plates exist anyway, the verifier computes per-camera agreement IoU inside the trusted
(cloth) region and flags cameras below a threshold — today's trust map survives as QA,
not as evidence gating.

## 5. `generate_masks_sam3.py` spec (implemented, Round 0)

Standalone script executed in the SAM3 env on the project PC
(`conda run -n sam3 python generate_masks_sam3.py ...`); paradex-import-free. The
public image API is confirmed and small: `build_sam3_image_model()` +
`Sam3Processor(model)`, then per image `set_image(pil)` -> state,
`set_text_prompt(prompt, state)` -> `{"masks", "boxes", "scores"}`,
`add_geometric_prompt(box, label, state)` for visual box prompts (normalized
`[cx, cy, w, h]`). The main pass uses the object concept prompt; fallback retries
use a separate text prompt plus the projected-hull box on a fresh image state.
The checkpoint resolves from the local HF cache by default; `--checkpoint` overrides.
The SAM3 env requires Python 3.12+/torch 2.7+, so the separation from calc (3.8)
is mandatory, not stylistic.

- Inputs: `--session-dir` (the `<object>/<session>` capture dir), `--output-dir`
  (calc output dir, default `<session>/processed`), `--prompt` (noun phrase),
  `--prompt-json` (per-object prompt file, tolerant schema), `--states` subset,
  `--sam3-repo-root` (default `~/sam3`), `--checkpoint`, `--device`, `--min-score`,
  `--instance-mode`, `--fill-holes-max-frac`, `--overlay-cameras`.
- Prompt resolution order: explicit `--prompt` > `--prompt-json[object]` > cleaned
  object name (underscores to spaces). Note: `apple` is a valid noun phrase as-is;
  `frame_oak_hybrid` is not — the cleaned-name fallback is a convenience, not a
  guarantee. Going forward, capture object names should be chosen
  noun-phrase-friendly, or the prompt JSON maintained per object. The validated
  prompt for the current target is `"wooden picture frame"`.
- Per image: one PCS call -> instances. Keep the top-scoring instance (`top1`,
  default; SAM3 was observed to return the whole object, handle included, as a single
  instance) or OR all instances above `--min-score` (`union`).
- ROI-aware generation (added after the first hybrid_01 validation, corrected after
  the second and third): the object box per state/camera is the projected 3D bbox of
  the triangulated object cloud (`multiview/<state>/object_points_roi.ply`,
  mask-independent), median-anchored and size-bounded by the input-mesh diagonal
  because the cloud itself can be scene-contaminated (the calc ROI sphere crop is
  skipped for off-center placements, and a plain percentile bbox of such a cloud
  projected to the whole image). The generator (a) skips
  cameras that cannot see the box (`object_out_of_view` instead of a fake SAM
  failure), (b) gates instance selection on box overlap, and (c) fills concept-miss
  views from calibrated multi-view geometry. Direct geometry-only fallback is removed:
  even with an object-sized box it can segment floor/background inside the box and
  poison hull voting. The current fallback leaves PCS misses blank during the SAM
  pass, then carves a coarse visual hull from the successful SAM masks inside the
  object box. The projected hull is used only as a location guide for a second SAM3
  call that combines `--fallback-prompt` (default `"foreground object"`) with the
  hull-derived box; only a returned SAM3 candidate with enough guide overlap/coverage
  is saved (`prompt_mode=hull_guided_sam3` in
  `mask_stats.csv`). Findings that forced this:
  close-up top views showing only wood grain do not look like a "picture frame" to PCS,
  and those maskless top views let the visual hull extrude into a pillar whose PCA axes wreck the initial
  registration; prompting with the ROI *sphere* instead (camera-convergence centered,
  1.5x mesh diagonal) made SAM segment cloth/background chunks and kept object-absent
  views alive.
- Post-processing: fill *enclosed* background holes smaller than
  `--fill-holes-max-frac` of the mask area (SAM masks show scattered pinholes;
  real openings like the fold-out handle gap stay open because they are larger or
  touch the outside background). Cross-view hole disagreement is absorbed by the
  vote-ratio hull anyway; the filling just cleans the per-view evidence.
- Outputs: the mask contract of section 4, with SAM confidence per mask and the
  provider manifest that switches calc's `auto` source to external.
- Existing masks/overlays of a processed state are deleted before writing so plate
  and SAM outputs never mix in one state directory.

## 6. Round plan

**Round 0 — SAM mask import + validation (no restructure). [implemented; first
validation run: side/oblique masks good, but close-up top views failed PCS
(wood grain only, concept miss) -> pillar-shaped hull -> broken initial poses.
Second validation: retrying those views with the ROI-*sphere* box made SAM grab
cloth/background and kept object-absent views alive -> box source switched to the
projected 3D bbox of the triangulated object cloud. Third validation: that cloud
itself can be scene-wide (ROI crop skipped on off-center placements) and its plain
bbox covered the whole image -> box made median-anchored and size-bounded by the
input-mesh diagonal; direct projected-box masks clipped to the box were still too coarse.
Fourth diagnosis: direct geometry-only fallback can still make false positives on
floor/background. Fifth update: fallback now uses a separate prompt (`"foreground object"`
by default) plus the hull-derived box.
Sixth diagnosis: with good SAM masks the pose still failed — `[REG] target=hull`
but the ~800-point hull was still not a clean frame (floating blobs, front smear) and
only ~20 crop/tight inliers survived. Root cause is structural, not a mask bug: a visual
hull over-estimates exactly the thin (thickness) dimension whose axis the orientation
depends on, so reconstruct-then-fit is the wrong direction for a flat/thin/untextured
object. Added a direct-fit alternative `--registration-coarse-method mask_silhouette`:
fit the known metric mesh to the 2D masks (translation from a robust mask-centroid ray
intersection, orientation from a coarse SO(3) grid scored by a trimmed-mean multiview
silhouette-vs-mask cost, then local refine), never building a hull; trimming + robust
translation tolerate FP/FN masks and the moving articulated part; auto-falls back to
`hull_pca` when a state has too few masks.
Seventh update: with mask_silhouette, green sticks to the masks in most states, but a
placement whose articulated part moved still locked onto a 90-degree in-plane flip (the
near-square body is symmetric and the moved part misleads the per-view cost — an objective
problem, not a search problem, so more seeds do not help). Added
`--registration-group-share-pose` (default on): the body pose is shared across a
placement group's states, so after the per-state fits a post-pass scores each state's
orientation and its +/-90/180-degree in-plane variants against the group's pooled masks
and adopts the group-best for every state (`mask_silhouette_group_shared`). This is the
payoff of the 5-placement x 3-joint hybrid capture.
Eighth update (performance): the group-share fixed the gross flips but the full pipeline
ran ~20-30 min. The cost was per-candidate silhouette rasterization (fill-poly over
thousands of triangles) inside the coarse grid, the local refine, and — worst — the
group-share pass, which re-ran a full greedy translation refine for every 0/90/180/270
candidate. Replaced the scorer with a distance-transform / chamfer cost
(`_multiview_chamfer_cost`): each view precomputes a mask distance field once, and a
candidate is scored by projecting `--registration-silhouette-score-points` (2000) mesh
surface points -> precision (fraction of points OUTSIDE the mask, scale-free strong
anti-spill, + small smooth outside-distance term for a basin, ~1.5 px boundary tolerance
so point-vs-rasterized-mask discretization is not a noise floor) + recall (occupancy
splat coverage), no rasterization. Collapsed group-share to score-then-single-refine
(score each candidate with the state's existing translation, refine translation only for
the adopted winner). Smoother objective also lets the refine walk a finer step tail
(10/4/1.5 deg, 0.04/0.015/0.005 diag) to trim the residual. Translation placement kept
identical to the old scorer (mesh vertex-mean centroid), so only scoring changed. Legacy
raster scorer (`_multiview_mask_pose_cost`, `_maybe_decimate_faces`) retained but unused.
(First cut used a weak saturated-distance-only precision + surface-mean centroid, which on
real data drifted -- green past orange, a whole group off, orange mispositioned; the
fraction-outside precision and vertex-centroid revert fixed it.) Re-validation pending.]**
Ninth update (2026-07-09, capture rethink + decoupling): with SAM3 providing masks, the
move-the-object-around/median-plate rationale is dead, and off-center placements (some
groups masked in only 8-9 of 22 cameras) are the likely cause of the still-unfixable
"orange wrong from the start" states. New capture direction (thinking): keep the object
CENTERED in the calibrated volume every state (all 22 views) and reorient it in place
instead of translating; articulate at each orientation; grouping becomes an explicit
capture label (`p<k>_j<n>`), not centroid clustering. First concrete step taken: a new
standalone `preprocess.py` (cv2 + numpy, paradex-free) lifts the undistortion out of calc;
`generate_masks_sam3.py` auto-invokes it (`--auto-undistort`, default on) when a state has
raw captures but no undistorted images, so the mask step no longer needs a prior calc run
(new flow: capture -> generate_masks_sam3 [self-undistorts] -> registration). Leaning
toward a salvage-based re-architecture where the defense layer (plates, trust, shadow,
hull erosion, off-center ROI, centroid clustering) and possibly triangulation/hull for
pose are deleted, leaving pure multiview mask-silhouette registration as the core.
New-pipeline modules built so far (self-contained, paradex-free, numpy/cv2/trimesh):
`preprocess.py` (standalone undistort, auto-invoked by generate_masks_sam3),
`register_states.py` (multiview mask-silhouette registration: chamfer coarse+refine,
label/size grouping, group-share with front/back+normal candidates + group-joint
continuous refine, per-state fit_iou, registration_report.html), and `segment_parts.py`
(Stage A part decomposition: static/movable vertex labeling from cross-state multiview
mask consistency at the shared body pose, region-grow + hole-close + component merge,
body/part submesh + colored-mesh + segmentation_report.html; validated precision-strong,
recall partial near the joint). Stage B (per-state part pose from beyond-body residual +
residual/motion/geometry refinement) and Stage C (solid part completion) are now built.
Stage B starts from a conservative full-mesh residual (`object mask - full known rest
mesh`) rather than `object mask - Stage-A body`: Stage A is only a proposal, so a static
face it removes cannot be allowed to manufacture residual evidence for itself. The old
body-only residual remains a comparison/refinement channel after high-confidence motion
labels exist; both overlay sets are exported for diagnosis. The next evidence upgrade
adds within-placement RGB change as a positive-only channel: SAM3 absence and occlusion
are unknown, not negative votes. The default hybrid fit map keeps image motion directly
and admits silhouette-residual pixels only near it, so a broad mesh/SAM residual cannot
dominate candidate fitting. Once a broad Stage-A component proposes a part transform,
Stage B splits it into welded, locally smooth mesh patches and retains only patches whose
camera-facing projections consistently prefer the fitted moving pose to the body pose
across independent placement groups. This is the first point where the supplied mesh is
used as local candidate geometry and visibility structure rather than only as a global
silhouette.
The 2026-07-10 run exposed the limit of independent per-patch testing: the surviving
wrong patches were rear-frame surfaces that the handle occludes or reveals — their
disocclusion is real image change at their own footprint, and hinge-adjacent geometry
sweeps into the handle's change region under the shared transform, so "this patch
moved" and "its occluder moved" predict the same pixels. Patch certification is
therefore a JOINT greedy subset selection under the same depth-composited scene
(default `composite_joint`): patches are added by marginal explained-change gain minus
a spurious-change penalty, so the handle alone explains the revealed surfaces' change
and their gain collapses; the part pose is then refit on the selected subset and the
selection repeated once (the broad-candidate pose was biased by its static surfaces).
Per-patch OBJ/PLY exports and explained/spurious/unexplained overlays make each
selection auditable.
First joint-selection run + overlays exposed two corrections: (1) the true handle
patches were never selected because the RGB evidence contained false thin change
bands along the fixed frame's high-contrast edges — they dragged the part-pose fit
toward the frame and fed rim-shaped frame patches their "explained" pixels; RGB
motion within a band of the registered body silhouette boundary is now suppressed.
(2) Refit-round adoption now compares the joint score (evidence-fixed, comparable)
instead of preferring the latest round; subset fit IoU is size-biased, reported
separately, and can no longer downgrade the candidate's moving status (which had
silently disabled Stage C).
The next correction makes that explaining-away competition cross-candidate rather
than per-candidate. A small protruding hinge can provide the only reliable SE(3)
track, while the larger handle candidate provides the mesh patches that should move
with it. Every patch is therefore evaluated under every moving-candidate track, with
one-track-per-patch exclusivity and a placement-group-normalized objective so track
coverage cannot win by raw pixel count. Selected patches are grouped by the borrowed
track and passed to Stage C as that one moving component; this is still not a screw
constraint, so different tracks remain valid competing components for separable
objects. A successful `frame_oak` result must show the handle patches assigned to the
hinge track and the unrelated frame patch losing marginal gain. If those patches do
not align even under the hinge track, the remaining failure is pose quality and the
next step is screw-constrained pose sharing, not more segmentation thresholds.
Stage C fixes the "shell" symptom of A/B: A/B only score the camera-visible surface that
vacates its rest footprint, so the handle back/interior stay body and the exported part is
an open shell. Stage C scores EVERY vertex with an explaining-away test (the discrete form
of PARIS's compositional split): movable evidence = residual support at the fitted part
pose minus at the body pose (does this pose explain pixels the body render cannot?),
static evidence = object-support loss at the part pose (the part pose throws a true body
vertex into free space). Plain object-mask advantage is depth-blind for a flat part folded
against the body (both footprints stay inside the silhouette -> zero contrast; only the
protruding hinge scored). A state contributes motion evidence only where the vertex does
not exit in that same state (hits without a misses-check seed the whole mesh). Stage C
first rejects bad body/part registrations, near-full-object residuals, and near-full-mesh
motion states; then it requires evidence from independent placement groups rather than a
single state winning through max aggregation. The Stage-B motion-refined shell is only a
soft seed and cannot override definitely-static evidence, so a broad `handle + body` Stage-B
candidate can shrink again. Coincident unwelded vertices are welded for connectivity, but
distance-based bridges across disconnected mesh pieces are off by default: proximity is not
part identity and otherwise leaks from a frame into hinges or metal fragments. The union of
trusted definitely-static regions is the wall; the seed grows through the intrinsically
ambiguous near-axis band to close the solid part (segmentation/stage_c/, shell-vs-solid
boundary-edge metric, per-state trust diagnostics). Stage D (joint fitting, Sturm/screw axis
from the filtered per-state part poses already in stage_b/part_observations.json) follows
only after Stage C evidence is stable.
Hull path re-validation (if still used) recommended with `--hull-mask-erode 0` since SAM
masks lack the plate blur bloat that erode compensated.
New `generate_masks_sam3.py`; calc gains `--object-mask-source` and the external
consumption path (plate building skipped, plate trust gating force-disabled so stale
trust maps cannot gate learned masks); nothing else changes. Run on `hybrid_01` with a
fresh output dir for side-by-side comparison against the plate run. Success gate: masks
visibly correct off-cloth; `[REG] target=hull` poses converge (few degrees, correct
flip) with at most 1 refine round. This round decides everything downstream — if SAM
masks do not fix pose, stop and rethink before touching structure.

**Round 1 — defense layer demotion.**
Plate-diff becomes verifier only (agreement stats, warnings). Defaults simplified:
trust gating / hull mask erode / multi-round polish / shadow tuning become legacy flags
that default off with external masks. Registration candidate set shrinks (clean masks
need fewer rescue candidates). No deletion yet — flags remain for plate-only sessions.

**Round 2 — package extraction (behavior-preserving; three sub-rounds).**
- 2a: golden-run regression harness (hash/compare manifests + key PLY/CSV outputs on
  hybrid_01), then extract `config`/`io_bundle`/`scene`.
- 2b: extract `grouping`/`masks`/`hull`/`registration`.
- 2c: extract `motion`/`parts`/`kinematics`/`completion`/`report`; `calc_states.py`
  becomes a shim; delete dead defense code paths whose flags no session uses.
Each sub-round must reproduce the golden outputs bit-for-bit (or with documented
tolerance) before the next starts.

**Round 3 — part evidence upgrade.**
- Mask XOR within placement: registration-free 2D moving-part evidence from SAM masks
  (replaces/augments raw image diff — sharper boundaries, immune to shadows).
- Part hull: visual-hull the XOR regions per placement -> 3D swept volume of the
  moving part; merge with the existing carve.
- Optional part-prompt mode in `generate_masks_sam3.py`: sample positive points from
  XOR blobs, prompt SAM for a clean part-level mask per view (SAM as boundary
  cleaner for motion-discovered parts).

**Round 4 — kinematics + object state classification.**
Per-state part poses from part registration; classify rigid / revolute / prismatic /
separable from part pose tracks (relative-motion residuals); axis/limit fitting with
inlier gating. Emits the object state class — the first field of the handoff contract.

**Round 5 — completion + tracking handoff.**
Residual volume (state hull minus registered base mesh, fused across states) ->
unseen-surface points -> part mesh completion. Define and emit the handoff bundle
(section 7) and validate it against what the tracking pipeline actually loads.

**Round 6 (optional) — external pose provider.**
If `run_video.py --sam3-seed-tracking` seed refinement works on discrete state images
(open question 8.1), `registration.py` gains an import mode that consumes its 6D poses;
the internal hull+ICP path remains the fallback. This could shrink the registration
stage dramatically, but is strictly an optimization — the plan does not depend on it.

## 7. Handoff bundle (draft, to align with the tracking side)

```
handoff/
  object_model.json      # state class, parts, joints {type, axis, origin, limits}
  parts/<part_id>.obj    # canonical-frame part meshes (base mesh split + completed)
  state_poses.json       # per capture state: per-part world 6D pose
  prompts.json           # SAM3 prompt(s) that worked for this object
```

`prompts.json` matters: the tracking side uses the same SAM3, so the prompt validated
here transfers directly to seeding the tracker.

## 8. Open questions (non-blocking; design keeps them pluggable)

1. Does `run_video.py` seed refinement run on discrete states (no temporal
   continuity)? -> decides Round 6. Check on the processing PC.
2. ~~Exact SAM3 python API~~ answered: confirmed from the public repo and implemented
   in section 5. Remaining sliver: the tracking repo's `object_prompts_original.json`
   schema, if we ever want to share one prompt file with the tracker (the generator's
   `--prompt-json` accepts a tolerant schema meanwhile).
3. Handoff bundle format -> align with the tracking pipeline owners before Round 5.
4. Capture protocol: with SAM masks, plates are no longer needed, so multiple
   placements stop being mandatory for masks. Keep the hybrid protocol anyway (pose
   diversity, unseen-surface exposure, within-placement diffs) unless capture cost
   becomes the bottleneck.
5. SAM confidence as per-camera weight in hull voting — only if hull quality turns out
   uneven; not planned by default.

## 9. What gets deleted vs kept

Deleted or demoted to legacy after Rounds 1-2: plate building as primary mask source,
shadow suppression tuning surface, trust gating as evidence gate (survives as
verifier), `--hull-mask-erode`, multi-round mask-score polish, edge-channel scoring as
registration driver, sparse-cloud registration target as default.

Kept as the permanent core: camera bundle/calibration IO, triangulation + ROI + ground
plane, visual hull, PCA/ICP registration + single-round mask refine, placement
grouping, within-placement motion evidence + carve, mesh segmentation, part
registration, kinematics fitting, completion, debug report.

## 10. Risks

- SAM3 misses or merges instances on unusual objects -> hull-projection fallback +
  plate verifier; worst case the legacy plate path still exists per session.
- Refactor regressions in a 10k-line split -> golden harness gates every sub-round.
- ~~Two-PC workflow friction~~ resolved: SAM3 turned out to be available locally on
  the project PC (`~/sam3`, conda env `sam3`), so mask generation and calc share one
  machine and one filesystem. The plain-file contract still keeps a remote generator
  possible if a bigger GPU is ever needed.
