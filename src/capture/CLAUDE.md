# CLAUDE.md — src/capture

## Purpose
Parent group for data acquisition: multi-camera capture (`camera/`) and robot
motion capture (`robot/`). Most full datasets combine both.

## Files
No scripts at this level — only subdirectories:
- `camera/` — image/video/stream capture. Local scripts (`image.py`, `video.py`,
  `stream.py`) vs main-PC orchestrators (`*_remote.py`) that command capture-PC
  daemons over ZMQ. See `camera/CLAUDE.md`.
- `robot/` — `teleop_real.py` (CaptureSession teleop recording) and
  `xarm_teaching.py` (manual-mode waypoint saving). See `robot/CLAUDE.md`.

## paradex modules used
- Camera: `paradex.io.camera_system.*`, `paradex.io.capture_pc.*`, `paradex.image.merge`.
- Robot: `paradex.dataset_acqusition.capture.CaptureSession`, `paradex.io.robot_controller`,
  `paradex.io.teleop`, `paradex.transforms.conversion`.

## Data flow & IO
- Distributed camera: main-PC `remote_camera_controller` → capture-PC
  `src/camera/server_daemon.py` → `CameraLoader` writes to that PC's
  `~/shared_data/<save_path>/<timestamp>/raw`. Cam params snapshotted on the main PC.
- Robot: `CaptureSession` / XArm SDK write arm/hand/state files locally.
- Output convention: `~/shared_data/<save_path>/<timestamp-or-index>/`.

## When working here
- Two-PC pattern for camera capture: daemon side lives in `src/camera/`, orchestrator
  side in `src/capture/camera/`. Always start daemons before remote scripts.
- `CaptureSession` (in `paradex/dataset_acqusition`) is the shared abstraction for
  multimodal capture (camera+arm+hand+teleop).

## Gotchas
- No `image_main.py`/`video_main.py`/`stream_main.py` files exist — those strings are
  controller labels passed to `remote_camera_controller`; work runs in `server_daemon.py`.
- Module dir is `dataset_acqusition` (missing 'i') — intentional typo, do not "fix".
- Flag inconsistency: `video.py` uses `--frame_rate`, `video_remote.py` uses `--fps`.
