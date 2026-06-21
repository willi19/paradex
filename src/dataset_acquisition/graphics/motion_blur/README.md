# Graphics — Motion Blur Capture

Records hardware-synced multi-camera **videos** of an xarm replaying a fixed joint trajectory, sweeping camera exposure/gain and joint speed. Produces a motion-blur study dataset (pair the "sharp" version in [`../`](..)).

## Scripts
| File | Purpose |
|------|---------|
| `capture.py` | Main entry. Loads trajectory, builds a `CaptureSession(camera, arm="xarm")`, and for each `(exposure,gain)` pair × `speed` records one synced video while replaying the trajectory. |
| `replay.py` | Helpers `reset_to()` (blocking move to start) and `replay_q_deg()` (servo-stream the trajectory at a target deg/s via a 100 Hz loop). Imported by `capture.py`. |
| `__init__.py` | Empty (package marker). |

## Usage
Run on the **main PC** (capture-PC camera daemons must be live; uses the UTGE900 sync generator):
```bash
python src/dataset_acquisition/graphics/motion_blur/capture.py --name run1 \
    --exposures 2500 8000 16000 30000 --gains 12 6 0 0 \
    --speeds 30 --fps 30
```
- `--exposures` and `--gains` are paired 1:1 (lengths must match, else `SystemExit`).
- Each trial: reset to `q_deg[0]`, `cs.start(... mode="video", stage=...)`, `replay_q_deg()`, `cs.stop()`.
- Runs all cells then returns the arm to start; no keyboard control.

## Inputs & Outputs
- **Hardware:** multi-camera rig + XArm + UTGE900 sync generator + TimestampMonitor (arm present). No hand/teleop.
- **Input traj:** `--traj` (default `~/mcc_minimal/traj/dynamic/xarm/seed42_fwd100.npz`, key `q_deg`, degrees).
- **Output:** `~/shared_data/capture/graphics/motion_blur/<name>/<timestamp>/`
  - `meta.json` (traj, exposures, gains, speeds, fps, start/end q).
  - `raw/<stage>/` per trial where `stage = exp{e}_gain{g}_spd{s}`, containing `videos/`, `arm/`, `timestamps/` (per `CaptureSession.start`), plus saved camparam + C2R on stop.

## Related
- [`paradex/dataset_acqusition/capture.py`](../../../../paradex/dataset_acqusition/capture.py) — `CaptureSession`.
- [`paradex/io/robot_controller/xarm_controller.py`](../../../../paradex/io/robot_controller/xarm_controller.py) — `move(action_rad, is_servo=True)` 100 Hz servo loop used by `replay_q_deg`.
- [`../`](..) — `image_capture.py` / `image_traj.py` sharp counterparts.
- Video upload: [`src/util/upload_video/`](../../../util/upload_video).
