# Dataset Acquisition Scripts

Per-dataset capture pipelines that orchestrate the multi-camera rig (plus arm / hand / teleop where needed) to record raw sessions under `~/shared_data/capture/...`. Each subdirectory is one dataset/experiment family, most built on `paradex.dataset_acqusition.capture.CaptureSession`.

## Subdirectories
| Directory | What it captures |
|-----------|------------------|
| [`graphics/`](graphics) | Static-pose and per-waypoint image grids sweeping camera exposure × gain (sharp reference data). |
| [`graphics/motion_blur/`](graphics/motion_blur) | Synced videos of an xarm sweeping a trajectory at varying speeds and exposures, for motion-blur study. |
| [`hri/`](hri) | Camera-only human-robot-interaction recordings, keyboard-triggered. |
| [`miyungpa/`](miyungpa) | Teleoperated (XSens) camera + arm + hand demonstrations. |
| [`object_turntable/`](object_turntable) | Camera-only turntable captures of an object for 6D / COLMAP reconstruction. |

## Common Model
- Most scripts instantiate `CaptureSession(camera=..., arm=..., hand=..., teleop=...)`, then loop `cs.start(rel_path) ... cs.stop()` and finally `cs.end()`.
- `cs.start()` writes camera video/images via the remote camera controller, fires the UTGE900 sync generator, and (when arm/hand present) starts a `TimestampMonitor` plus per-sensor recorders. It also saves the active camera params + camera-to-robot transform on `stop()`.
- The `graphics` image scripts bypass `CaptureSession` and drive `remote_camera_controller` directly in free-run (no sync).

## Outputs
All sessions land under `~/shared_data/capture/<dataset>/<name>/<timestamp>/` with a `raw/` (or `raw/<stage>/`) subtree containing `videos/` or `images/`, plus `arm/`, `hand/`, `teleop/`, `timestamps/`, `state/` as applicable.

## Related
- [`paradex/dataset_acqusition/capture.py`](../../paradex/dataset_acqusition/capture.py) — `CaptureSession`.
- [`paradex/io/camera_system/`](../../paradex/io/camera_system) — remote camera controller, UTGE900 sync generator, timestamp monitor.
- [`paradex/io/robot_controller/`](../../paradex/io/robot_controller) — `get_arm`/`get_hand`, `XArmController`.
- Post-processing: [`src/process/miyungpa/`](../process/miyungpa), [`src/process/object_turntable/`](../process/object_turntable), and video upload [`src/util/upload_video/`](../util/upload_video).
