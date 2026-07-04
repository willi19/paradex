# CLAUDE.md — src/camera

## Purpose
Capture-PC-side long-running daemons. These run on the 6 capture PCs and expose
cameras to the network. The main PC drives them via the `*_remote.py` scripts in
`src/capture/camera`.

## Files
- `server_daemon.py` — instantiates `camera_server_daemon()` (ZMQ server wrapping
  one `CameraLoader`) then sleeps forever. Capture-PC role. Receives `register /
  start / stop / heartbeat / reload / end` commands. Ports: ping 5480 (REP),
  monitor 5481 (PUB status), command 5482 (REP).
- `monitor_daemon.py` — one-liner: `CameraMonitor(web_port=1234)`. Capture-PC role.
  Web dashboard of camera status.

## paradex modules used
- `paradex.io.camera_system.camera_server_daemon.camera_server_daemon`
- `paradex.io.camera_system.monitor_daemon.CameraMonitor`

## Data flow & IO
Main PC's `remote_camera_controller` (in `src/capture/camera/*_remote.py`) sends
JSON commands over ZMQ. `server_daemon` translates `start(mode, syncMode, save_path,
fps, exposure_time, gain)` into `CameraLoader` calls; files land on the capture PC's
disk at the `save_path` carried in the command.

## When working here
- These are entry points only — logic lives in `paradex/io/camera_system/`.
- `server_daemon.py` must be running before any remote orchestrator connects, or
  `remote_camera_controller.initialize()` raises `ConnectionError`.

## Gotchas
- `server_daemon.py` blocks forever via `while True: time.sleep(1)`; the server runs
  in background threads.
- The string passed to `remote_camera_controller(name)` (e.g. `"image_main.py"`) is
  only a controller label — there is NO `image_main.py` file. The work is dispatched
  to `server_daemon.py` here.
- Server enforces a single-controller lock (register/force_takeover); a second
  controller must `force_takeover` to grab it.
