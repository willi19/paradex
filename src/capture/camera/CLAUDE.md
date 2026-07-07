# CLAUDE.md - src/capture/camera

Main-PC and local capture script entry points. Keep this file as a routing note;
the canonical camera-system guide is `docs/camera_system.md`.

## Pick The Path

| Need | File | Runs on | Backend |
|------|------|---------|---------|
| distributed stills | `image_remote.py` | main PC | `remote_camera_controller` |
| distributed video | `video_remote.py` | main PC | `remote_camera_controller` |
| distributed preview | `stream_remote.py` | main PC | `remote_camera_controller` + `stream_client.py` |
| local stills | `image.py` | capture/local PC | `CameraLoader` |
| local video | `video.py` | capture/local PC | `CameraLoader` |
| local preview | `stream.py` | capture/local PC | `CameraLoader` |

Normal rig usage is distributed:

```text
*_remote.py -> remote_camera_controller -> capture-PC server_daemon.py
```

Local scripts instantiate camera hardware on the same PC and are mainly for
validation or single-PC use.

## Runtime Boundary

- The string passed to `remote_camera_controller(name)` is a controller label,
  not a real script filename.
- Remote scripts never instantiate `Camera` or `PyspinCamera`; actual hardware
  opens only in `server_daemon.py -> CameraLoader -> Camera -> PyspinCamera`.
- `server_daemon.py` must already be running on every capture PC.
- Keyboard convention: `c` capture/start, `s` stop, `q` quit.

## Local Details Worth Preserving

- `video_remote.py --sync_mode` drives the `UTGE900` signal generator.
- `video.py` uses `--frame_rate`; `video_remote.py` uses `--fps`.
- `stream_remote.py` SSH-launches `stream_client.py`; do not run
  `stream_client.py` by hand for the normal preview workflow.
- Remote output paths are passed as relative `shared_data/<save_path>/<ts>/raw`;
  camera params are snapshotted from the main PC next to the capture root.
