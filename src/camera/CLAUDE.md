# CLAUDE.md — src/camera

Capture-PC daemon entry points. Keep this file short. Editing camera internals?
The canonical agent guide is `agent_docs/camera_system/internals.md` (layer map,
lifecycle, config, failure modes); the human overview is `docs/camera_system.md`.

## Files

- `server_daemon.py` — capture-PC ZMQ command server. Instantiates
  `camera_server_daemon()`, which owns `CameraLoader -> Camera -> PyspinCamera`.
  Receives `register / start / stop / heartbeat / reload / end`.
- `monitor_daemon.py` — capture-PC web/status monitor.
- `reset_cameras.py` — main-PC recovery tool. SSHes capture PCs, kills/restarts
  daemons. Use when cameras hang and the next run cannot start them.

## Runtime Boundary

Main-PC apps and `src/capture/camera/*_remote.py` talk to
`remote_camera_controller`; they do not instantiate hardware cameras. Actual
hardware opens only inside the capture-PC daemon path:

```text
server_daemon.py -> CameraLoader -> Camera -> PyspinCamera
```

## Operational Notes

- `server_daemon.py` must be running before remote capture scripts connect.
- Ports: ping `5480`, monitor `5481`, command `5482`.
- Idle/dead-man timeout: `PARADEX_CAMERA_IDLE_TIMEOUT_S`, default `5.0`.
- The string passed to `remote_camera_controller(name)` is a controller label,
  not a script filename.
- Changing daemon-side camera code requires restarting daemons on every capture PC.
