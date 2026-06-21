# CLAUDE.md — src/capture/camera

## Purpose
Capture stills / video / live stream from the multi-camera rig. Two execution
models coexist:
1. **Local**: `image.py` / `video.py` / `stream.py` drive a `CameraLoader` on the
   PC they run on.
2. **Distributed**: `*_remote.py` run on the **main PC** and command the capture-PC
   `server_daemon.py` instances via `remote_camera_controller` (ZMQ). `stream` is
   special — it SSH-launches `stream_client.py` and pulls frames back.

All scripts use the `listen_keyboard` event pattern: `c`=capture/start,
`s`=stop, `q`=quit/exit. The main loop polls `Event`s with short sleeps.

## Files
- `image.py` — capture PC / local. On `c`: `find_latest_index` → increment →
  `camera.start("image", False, save_path=.../<idx>/raw)` → `stop()`.
- `image_remote.py` — MAIN PC. `remote_camera_controller("image_main.py")`. On `c`:
  `save_current_camparam` then `rcc.start("image", False, shared_data/<save_path>/<ts>/raw)`.
- `video.py` — capture PC / local. `c`=start `video` mode, `s`=stop; saves cam params
  after each clip. `--sync_mode`, `--frame_rate`.
- `video_remote.py` — MAIN PC. `remote_camera_controller("video_main.py")`. `c`=start,
  `s`=stop. With `--sync_mode` also drives `UTGE900` signal generator
  (start/stop/end) using `network_info["signal_generator"]["param"]`. `--fps`.
- `stream.py` — capture PC / local. `c`=start `stream` mode, `s`=stop. Preview/no save.
- `stream_client.py` — CAPTURE PC. `MultiCameraReader` → resize /8 → JPEG q85 →
  `DataPublisher(port=1234)`. `CommandReceiver(port=6890)` for `exit`. SSH-launched
  by `stream_remote.py`; do not run manually.
- `stream_remote.py` — MAIN PC. `run_script("python src/capture/camera/stream_client.py")`
  to launch clients, `DataCollector` to gather frames, `remote_camera_controller("stream_main.py")`
  to start cameras, `merge_image` + `cv2.imshow` for preview. `q`=quit.

## paradex modules used
- `paradex.io.camera_system.camera_loader.CameraLoader` (local scripts)
- `paradex.io.camera_system.remote_camera_controller.remote_camera_controller` (remote scripts)
- `paradex.io.camera_system.camera_reader.MultiCameraReader` (stream_client)
- `paradex.io.camera_system.signal_generator.UTGE900` (video_remote sync)
- `paradex.io.capture_pc.ssh.run_script`, `.data_sender.{DataPublisher,DataCollector}`,
  `.command_sender.{CommandSender,CommandReceiver}`
- `paradex.image.merge.merge_image`
- `paradex.calibration.utils.save_current_camparam`
- `paradex.utils.{file_io.find_latest_index, path.shared_dir, keyboard_listener.listen_keyboard, system.network_info}`

## Data flow & IO
- Local: `CameraLoader` writes to `os.path.join(shared_dir, save_path)`.
- Remote: controller passes `shared_data/<save_path>/<timestamp>/raw` as the path;
  files land on the **capture PC** disks. Main PC writes the cam-param snapshot under
  `shared_dir/<save_path>/<timestamp>`.
- Stream: capture PC publishes 1/8-scale JPEGs (ZMQ 1234) → main PC decodes + merges;
  nothing persisted.

## When working here
- The string arg to `remote_camera_controller(...)` is a label only; there is no
  `image_main.py` / `video_main.py` / `stream_main.py` file. The capture work runs in
  `src/camera/server_daemon.py` on each PC.
- `CameraLoader.start(mode, syncMode, save_path=, fps=)` and
  `remote_camera_controller.start(mode, syncMode, save_path=, fps=, exposure_time=, gain=)`.

## Gotchas
- `image.py` uses positional `camera.start("image", False, save_path=...)`; sync is
  always False there.
- `video.py` uses `--frame_rate`, but `video_remote.py` uses `--fps` — different flag names.
- `image_remote.py` builds the target as `shared_data/...` (relative) while saving cam
  params under absolute `shared_dir`; keep this distinction.
- Hardware sync (`UTGE900`) is only wired in `video_remote.py --sync_mode`; the local
  `video.py`/`stream.py` accept `--sync_mode` but rely on the camera daemon for it.
- `server_daemon.py` must be running on every capture PC first, else remote scripts
  raise `ConnectionError`.
