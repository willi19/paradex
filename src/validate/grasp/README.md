# Grasp sequence validation

`src/validate/grasp/grasp_sequence.py` validates one object/episode directly from the
ECCV 2026 capture layout. Dataset inputs are read-only and the JSON result is
printed to stdout. Frame projection is currently excluded.

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

## Input resolution

For object `apple` and episode `3`, the defaults resolve as follows.

| Input | Path |
| --- | --- |
| Object mesh | `/home/temp_id/shared_data/mesh_new/apple/apple.obj` |
| Object pose | `<episode>/object_6d_pose.npz` |
| Human hand | `<episode>/hand/mano` and `<episode>/hand/mano_params` |
| Robot hand/arm | `<episode>/raw/hand` and `<episode>/raw/arm` |
| Camera-to-robot transform | `<episode>/C2R.npy` |

The third positional argument selects exactly one capture source:

- `human` → `human`
- `allegro` (alias `allegro_v5`) → `allegro_v5`
- `inspire` (alias `inspire_dftp`) → `inspire_dftp`
- `inspire_f1` → `inspire_f1`

This explicit selector avoids ambiguity when the same object and episode number
exists under multiple hand directories.

The validator never reads `object_6d_pose_v2.npz`. It prefers
`object_6d_pose.npz` and falls back only to `object_6d_pose_v1.npz`, because
some human episodes use that legacy non-v2 filename.

## Validity criteria

An episode is valid only when all three checks pass:

1. Before persistent grasp onset, the object remains close to its initial pose.
2. After a persistent grasp release, object motion is consistent with gravity
   or with support from the inferred floor.
3. The complete object trajectory has no abrupt translation jump.

Any single fingertip contact counts as grasp contact. The contact phase starts
when at least one finger touches the object and release begins when all finger
contacts are lost.
`NO_STABLE_CONTACT_PHASE`, `GRASP_TOO_SHORT`, `OBJECT_NOT_MOVED`, object slip,
self-collision, and robot-joint values do not directly affect validity.
Missing required inputs or malformed trajectories are dataset errors.

Video, camera calibration, median-edge projection, and mask-IoU projection are
not loaded or evaluated. `projection_alignment` is `null`, and no projection
overlay is generated.

## Motion before grasp

The validator evaluates object-pose frames from the sequence start up to 15
frames before persistent grasp onset. The 15-frame grace interval avoids
classifying normal object motion during the immediate grasp approach as
pre-grasp tracking failure. Translation is measured from the median position
of the first 15 frames, and orientation is measured from their mean rotation.

- Translation excursion greater than 0.05 m adds
  `OBJECT_PRE_GRASP_TRANSLATION_EXCESSIVE`.
- Rotation excursion greater than 120 degrees adds
  `OBJECT_PRE_GRASP_ROTATION_EXCESSIVE`.
- The total pre-grasp translation path length is reported as a diagnostic but
  is not itself a failure criterion.
- If no persistent grasp is found, the pre-grasp check is reported as not
  applicable because grasp onset cannot be defined.
- If fewer than 15 frames remain before the grace interval, it is reported as
  insufficient observation rather than failed.

The limits can be changed with `--max-pregrasp-translation` and
`--max-pregrasp-rotation`; the grace interval can be changed with
`--pregrasp-grace-frames`.

## Motion after grasp release

The longest sampled interval with at least one fingertip contact is the
inferred grasp. If persistent loss of all fingertip contacts follows it, the
validator evaluates every object-pose frame from the last contact through the
no-contact interval.

- Gravity is `(0, 0, -1)` in robot coordinates. Human sequences rotate that
  direction into their shared hand/object coordinate system using `C2R`.
- The interval is accepted when downward displacement is at least 10 mm or its
  gravity-axis velocity becomes at least 0.05 m/s more downward.
- These two gravity thresholds are minimum required effects, not maximum
  limits. A smaller value therefore does not provide enough evidence of
  gravity-consistent motion.
- Horizontal motion and rotation remain unrestricted.
- If the object is already supported by the inferred floor, it may stay still.
- If it falls to the inferred floor, it may stop moving afterward.
- An object remaining more than 0.2 m above the inferred floor beyond the
  relaxed fall deadline adds `OBJECT_DID_NOT_REACH_INFERRED_FLOOR`.
- Other gravity-inconsistent post-release motion adds
  `OBJECT_POST_CONTACT_GRAVITY_INCONSISTENT`.
- If no persistent release exists, this check is reported as not applicable
  rather than failed.

## Object position jumps

For every consecutive non-v2 object-pose pair, the validator measures the
translation displacement. Frame `i` adds `OBJECT_POSITION_JUMP` only when its
displacement exceeds both:

- the absolute 0.1 m limit; and
- eight times the median displacement of the five neighboring steps on each
  side, using a 0.002 m minimum local baseline.

The absolute condition ignores ordinary pose noise, while the local condition
avoids treating sustained fast motion as a discontinuous tracking jump. These
values can be changed with `--max-object-position-jump`,
`--object-jump-local-factor`, and `--object-jump-window`.
