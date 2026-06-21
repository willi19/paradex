# Graphics — Sharp Image Capture

Captures sharp (static) reference images while sweeping camera **exposure × gain**, either at a single xarm pose or at every waypoint of a trajectory. Produces the "sharp" counterpart to the motion-blur dataset. Both scripts run in camera free-run (no hardware sync).

## Scripts
| File | Purpose |
|------|---------|
| `image_capture.py` | Move xarm to ONE pose (`q_deg[pose_idx]`), then capture an exposure×gain **grid** of images (full cartesian product of `--exposures` × `--gains`). |
| `image_traj.py` | Move xarm to every selected trajectory waypoint; at each, capture images for `--exposures`/`--gains` **paired 1:1**. |

## Usage
Both run on the **main PC**; they call `remote_camera_controller` directly (capture PCs must already be running their daemons).

Single-pose grid (default 4 exposures × 5 gains = 20 cells at the first waypoint):
```bash
python src/dataset_acquisition/graphics/image_capture.py --name run1 \
    --pose_idx 0 --exposures 2500 8000 16000 30000 --gains 0 3 6 9 12 --settle 1.5
```

Per-waypoint sweep (paired exp/gain, every 20th waypoint):
```bash
python src/dataset_acquisition/graphics/image_traj.py --name run1 \
    --step 20 --exposures 2500 8000 16000 30000 --gains 12 6 0 0
```
- `image_traj.py` requires `len(--gains) == len(--exposures)`.
- No keyboard control; both scripts run to completion then exit (cameras + arm released in `finally`).

## Inputs & Outputs
- **Hardware:** multi-camera rig + XArm. No hand, no teleop, no sync generator.
- **Input traj:** `--traj` (default `~/mcc_minimal/traj/dynamic/xarm/seed42_fwd100.npz`, key `q_deg`, degrees).
- **Output** (`image_capture.py`): `~/shared_data/capture/graphics/sharp_grid/<name>/<timestamp>/`
  - `exp{e}_gain{g}/raw/images/<serial>.png` (raw capture)
  - `by_serial/<serial>/exp{e}_gain{g}.png` (reorganized via hardlink/copy)
  - `qpos.npy`, `ee_pose.npy`, `meta.json`
- **Output** (`image_traj.py`): `~/shared_data/capture/graphics/sharp_traj/<name>/<timestamp>/`
  - `q{i:04d}/exp{e}_gain{g}/images/<serial>.png`
  - `by_serial/<serial>/q{i:04d}_exp{e}_gain{g}.png`
  - per-waypoint `robot.npy`, `qpos.npy`, `eef.npy`; top-level `meta.json` + saved camparam.

## Related
- [`motion_blur/`](motion_blur) — the matching blurred-video capture using the same trajectory/exposure/gain conventions.
- [`paradex/io/camera_system/remote_camera_controller.py`](../../../paradex/io/camera_system/remote_camera_controller.py)
- [`paradex/io/robot_controller/`](../../../paradex/io/robot_controller) (`get_arm`, `XArmController`).
- [`paradex/calibration/utils.py`](../../../paradex/calibration/utils.py) — `save_current_camparam` (used by `image_traj.py`).
