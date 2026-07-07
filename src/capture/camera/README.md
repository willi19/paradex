# Camera Capture Scripts

Main-PC and local scripts for image/video/stream capture. For the camera-system
architecture, daemon lifecycle, heartbeat semantics, and recovery guide, read
[`docs/camera_system.md`](../../../docs/camera_system.md).

## Choose The Right Entry Point

| Goal | Normal entry point | Runs on | Requires capture-PC daemons |
|------|--------------------|---------|-----------------------------|
| distributed still images | `image_remote.py` | main PC | yes |
| distributed video | `video_remote.py` | main PC | yes |
| distributed live preview | `stream_remote.py` | main PC | yes |
| single-PC still images | `image.py` | capture/local PC | no |
| single-PC video | `video.py` | capture/local PC | no |
| single-PC stream | `stream.py` | capture/local PC | no |

Distributed scripts use:

```text
*_remote.py -> remote_camera_controller -> capture-PC server_daemon.py
```

Local scripts instantiate `CameraLoader` directly on the PC where they run.

## Prerequisite For Distributed Capture

On every capture PC:

```bash
python src/camera/server_daemon.py
```

If a daemon is unreachable, `remote_camera_controller` raises a `ConnectionError`
naming the unreachable PCs.

## Common Commands

Distributed images:

```bash
python src/capture/camera/image_remote.py --save_path <dataset>
```

Distributed video:

```bash
python src/capture/camera/video_remote.py --save_path <dataset> [--sync_mode] [--fps 30]
```

Distributed preview:

```bash
python src/capture/camera/stream_remote.py
```

Single-PC/local:

```bash
python src/capture/camera/image.py --save_path <dataset>
python src/capture/camera/video.py --save_path <dataset> [--sync_mode] [--frame_rate 30]
python src/capture/camera/stream.py [--sync_mode] [--frame_rate 30]
```

Keyboard convention:

| Key | Meaning |
|-----|---------|
| `c` | capture/start |
| `s` | stop |
| `q` | quit |

`stream_remote.py` is preview-only and SSH-launches `stream_client.py`; do not run
`stream_client.py` by hand for the normal workflow.

## Output Shape

Remote scripts pass a relative capture target such as:

```text
shared_data/<save_path>/<timestamp>/raw
```

Capture PCs write image/video data under that target. The main PC also snapshots
camera params next to the capture root via `save_current_camparam`.

## Implementation Pointers

| Need | File |
|------|------|
| main-PC command fan-out | `paradex/io/camera_system/remote_camera_controller.py` |
| capture-PC daemon | `src/camera/server_daemon.py` |
| camera group backend | `paradex/io/camera_system/camera_loader.py` |
| shared-memory stream reader | `paradex/io/camera_system/camera_reader.py` |
| SSH launch | `paradex/io/capture_pc/ssh.py` |
| ZMQ data preview | `paradex/io/capture_pc/data_sender.py` |
| hardware trigger | `paradex/io/camera_system/signal_generator.py` |

After changing daemon-side camera code, restart daemons on every capture PC.
