# Capture

Top-level entry point for recording data with the Paradex rig: multi-camera
imagery and robot (arm/hand) motion. Splits into two subsystems that are
typically run together for a full dataset.

## Subdirectories
| Dir | What it captures | Where it runs |
|-----|------------------|---------------|
| [`camera/`](camera) | Images, video, and live preview streams from the multi-camera rig. | Local (`image.py`/`video.py`/`stream.py`) or distributed via `*_remote.py` on the **main PC** + `server_daemon.py` on each **capture PC**. |
| [`robot/`](robot) | Teleop demonstrations and XArm teaching waypoints. | The PC connected to the robot / teleop device. |

## How the two sides fit together
- **Camera capture is distributed.** Each capture PC runs
  `src/camera/server_daemon.py` (a ZMQ camera server). The main PC runs a
  `*_remote.py` orchestrator (`image_remote.py`, `video_remote.py`,
  `stream_remote.py`) that connects to all daemons and issues synchronized
  start/stop commands. The local `image.py`/`video.py`/`stream.py` variants are
  for single-PC use.
- **Robot capture is local** to the robot PC and uses `CaptureSession` (teleop) or
  the XArm SDK directly (teaching).
- For a full multimodal session, combine camera capture with robot capture; the
  `CaptureSession` abstraction (`paradex/dataset_acqusition/capture.py`) can drive
  camera + arm + hand + teleop together.

## Typical workflow
1. On every capture PC: `python src/camera/server_daemon.py`.
2. On the main PC: run the desired `*_remote.py` camera orchestrator with a
   `--save_path`.
3. For robot data, run a script in [`robot/`](robot) on the robot PC.

See each subdirectory's README for exact commands, keyboard controls, and outputs.

## Outputs
All captures land under `~/shared_data/<save_path>/`, organized into
timestamped (or indexed) session folders. Camera orchestrators also snapshot the
active camera calibration params per capture.

## Related
- Capture-PC daemons: [`../camera`](../camera).
- [`paradex/io/camera_system`](../../paradex/io/camera_system), [`paradex/io/capture_pc`](../../paradex/io/capture_pc), [`paradex/io/robot_controller`](../../paradex/io/robot_controller), [`paradex/dataset_acqusition`](../../paradex/dataset_acqusition).
