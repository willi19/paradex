# Grasp sequence validation

`src/validate/grasp/grasp_sequence.py` validates one object/episode directly from the
ECCV 2026 capture layout. It never writes a report file: dataset inputs are
read-only and the JSON result is printed to stdout.

## Usage

```bash
python src/validate/grasp/grasp_sequence.py apple 3 allegro --pretty
python src/validate/grasp/grasp_sequence.py apple 0 human --pretty
python src/validate/grasp/grasp_sequence.py apple 0 inspire --pretty
python src/validate/grasp/grasp_sequence.py apple 0 inspire_f1 --pretty
```

Exit status is `0` when every selected report is valid, `1` for a completed
but invalid sequence, and `2` for a missing or malformed input.

## Validate every object and episode

The batch command scans every numeric episode directory under `allegro_v5`,
`human`, `inspire_dftp`, and `inspire_f1`, then stores all reports in one JSON
file:

```bash
python src/validate/grasp/batch_grasp_sequence.py \
  --output /home/temp_id/paradex/src/validate/grasp/grasp_validation_results.json \
  --workers 1 \
  --pretty
```

Use `--workers 1` for the lowest memory usage. A larger value, such as
`--workers 4`, validates episodes in separate processes. The output is
checkpointed atomically into the same file every 10 completed episodes. If the
run is interrupted, that file remains readable with `"complete": false`.

Each episode has one of three statuses:

- `valid`: all enabled validity criteria passed;
- `invalid`: the episode was readable but failed at least one criterion;
- `error`: required data was missing or malformed.

Incomplete episode directories are therefore recorded rather than silently
skipped. The command refuses to put its output anywhere below either source
root. It also refuses to replace an existing result unless `--overwrite` is
given. Meshes and captures are only read; no file or directory is created,
deleted, or modified under `/home/temp_id/shared_data`.

For episodes that establish a grasp and later lose contact, the report also
contains `contact_loss_event`:

- `normal_motion`: contact remains absent for at least two sampled frames, but
  the full-resolution object motion shows either downward displacement or a
  velocity trend toward gravity. Contact loss alone does not invalidate the
  episode.
- `tracking_error`: after enough observation time, the object shows neither
  downward displacement nor a velocity trend toward gravity.
- `insufficient_observation`: the post-loss interval is too short to decide;
  this does not invalidate the episode.

## Input resolution

For object `apple` and episode `3`, the defaults resolve as follows.

| Input | Path |
| --- | --- |
| Object mesh | `/home/temp_id/shared_data/mesh_new/apple/apple.obj` |
| Allegro hand | `/home/temp_id/shared_data/capture/eccv2026/v0/allegro_v5/apple/3/raw/hand` |
| Robot arm | `/home/temp_id/shared_data/capture/eccv2026/v0/allegro_v5/apple/3/raw/arm` |
| Allegro object pose | `/home/temp_id/shared_data/capture/eccv2026/v0/allegro_v5/apple/3/object_6d_pose_v2.npz` |
| Inspire hand | `/home/temp_id/shared_data/capture/eccv2026/v0/inspire_dftp/apple/3/raw/hand` |
| Inspire F1 hand | `/home/temp_id/shared_data/capture/eccv2026/v0/inspire_f1/apple/3/raw/hand` |
| Human hand | `/home/temp_id/shared_data/capture/eccv2026/v0/human/apple/3/hand` |
| Human object pose | `/home/temp_id/shared_data/capture/eccv2026/v0/human/apple/3/object_6d_pose_v2.npz` |

The third positional argument selects exactly one capture source:

- `human` → `human`
- `allegro` (alias `allegro_v5`) → `allegro_v5`
- `inspire` (alias `inspire_dftp`) → `inspire_dftp`
- `inspire_f1` → `inspire_f1`

This explicit selector avoids ambiguity when the same object and episode number
exists under multiple hand directories.

## Coordinate and time alignment

Robot validation:

1. Load the 6-DoF arm state and the selected hand's state file.
2. Interpolate each stream by its `time.npy` onto
   the time span in `raw/timestamps/timestamp.npy`.
3. Convert raw Inspire encoders when needed, then concatenate arm and hand
   values in the selected URDF's joint order.
4. Convert each camera-frame object pose to robot coordinates with
   `inv(C2R) @ camera_from_object`.

The robot models are `xarm_allegro_v5.urdf`, `xarm_inspire_DFTP.urdf`, and
`xarm_inspire_f1_right.urdf`.

Human validation pairs:

- `hand/mano/NNNNN.obj`
- `hand/mano_params/NNNNN.json`
- `frame_N` from `object_6d_pose_v2.npz`

by the same zero-based frame index. MANO and object poses remain in their shared
camera coordinate system. `C2R` is not applied to either trajectory; only its
rotation is used to express the robot-frame gravity direction in that camera
frame.

## Validity criteria

The validator first rejects malformed data: missing files, discontinuous frame
names, non-finite values, trajectory dimension mismatches, or object matrices
that are not valid SE(3) transforms.

It then samples at most 120 frames uniformly and computes deterministic
mesh-surface distances:

- Robot contacts use the configured distal meshes for Allegro or Inspire
  fingers (four for Allegro, five for Inspire).
- Human contacts use the standard MANO fingertip vertices for thumb, index,
  middle, ring, and pinky.
- The longest interval with at least two simultaneous fingertip contacts is
  the inferred grasp phase. One sampled-frame contact gap is bridged to reduce
  sensitivity to pose-estimation noise.

A sequence is invalid when any enabled criterion fails:

- no stable multi-finger contact phase;
- the inferred grasp is shorter than 10 original frames;
- the object moves less than 1 cm during grasp;
- after persistent contact loss, the object shows no gravity-consistent motion
  despite at least 0.1 seconds of observation;
- robot joints violate URDF position/velocity limits.

Thresholds are CLI options. Run
`python src/validate/grasp/grasp_sequence.py --help` for the complete list.

Robot self-collision is assumed absent and is not evaluated. Hand-object mesh
distance uses deterministic vertex/triangle center samples and a KD-tree. This
is intentionally fast enough for episode screening, but it is approximate
rather than continuous collision detection.

Relative object translation and rotation with respect to the palm/wrist remain
available as diagnostic metrics (`max_relative_translation_m` and
`max_relative_rotation_deg`), but they do not affect validity.

## Motion after contact loss

After an established grasp, persistent multi-finger contact loss only selects
the interval to inspect; contact loss itself is not a validity failure. The
check uses every original object-pose frame from the last contact sample through
the complete no-contact interval.

The object translations are projected onto a unit vector pointing in the
gravity direction. Robot trajectories use `(0, 0, -1)` in the robot/world
frame. Human trajectories use `C2R[:3, :3] @ (0, 0, -1)` because their hand and
object poses remain in the camera frame.

This is deliberately a relaxed gravity check, not a strict acceleration fit:

- `gravity_displacement_m` is the net displacement in the gravity direction.
- Gravity-axis velocities are finite differences of the projected positions.
  The medians of the first and last thirds form
  `gravity_velocity_change_m_s`.
- Motion is accepted when the interval lasts at least 0.1 seconds and either
  downward displacement is at least 5 mm or downward velocity increases by at
  least 0.05 m/s.
- Motion in other directions and object rotation are unrestricted by this
  check. A horizontal or upward throw is therefore accepted when its
  gravity-axis velocity trends downward.
- A shorter interval is reported as `insufficient_observation` without adding
  an invalidating issue.

If neither gravity condition is met, the validator adds
`OBJECT_POST_CONTACT_GRAVITY_INCONSISTENT`. The defaults are configurable with
`--min-contact-loss-samples`, `--gravity-min-observation`,
`--gravity-min-displacement`, and `--gravity-min-velocity-change`.
The reported maximum linear/angular speeds and accelerations are diagnostics
only; they do not affect validity.

This validator has no environment/support mesh. An object that loses hand
contact and immediately rests on a table may therefore be flagged when it shows
no downward motion.
