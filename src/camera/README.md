# Camera Daemons

Capture-PC entry points for the camera system. For the full mental model and
failure-mode guide, read [`docs/camera_system.md`](../../docs/camera_system.md).

## What Runs Here

| File | Runs on | Purpose |
|------|---------|---------|
| `server_daemon.py` | every capture PC | ZMQ command server; owns `CameraLoader -> Camera -> PyspinCamera` |
| `monitor_daemon.py` | capture PC | web/status monitor for local cameras |
| `reset_cameras.py` | main PC | SSH reset/relaunch of capture-PC daemons |

The main PC does not open cameras directly. Main-PC apps use
`remote_camera_controller`; this daemon receives the resulting commands.

## Start The Daemon

On each capture PC:

```bash
python src/camera/server_daemon.py
```

Optional monitor:

```bash
python src/camera/monitor_daemon.py
```

`server_daemon.py` listens on:

| Port | Purpose |
|------|---------|
| `5480` | ping |
| `5481` | monitor/status pub |
| `5482` | command |

## Lifecycle Contract

`server_daemon.py` handles:

```text
register -> start -> heartbeat... -> stop -> end
```

- `start` returns `ok` only if every local camera arms successfully.
- `heartbeat` reports `running`, expected/detected camera counts, per-camera
  states, frame ids, and errors.
- If no command/heartbeat arrives within `PARADEX_CAMERA_IDLE_TIMEOUT_S`
  (default `5.0`), the daemon stops cameras and releases the controller lock.
- `end` stops running cameras if needed, then releases the lock.

## Recovery

If the main-PC app dies or a camera wedges hard enough that the next run cannot
start, run from the main PC:

```bash
python src/camera/reset_cameras.py
python src/camera/reset_cameras.py --pc_list capture1 capture2
python src/camera/reset_cameras.py --no_restart
```

After changing daemon-side code (`camera_server_daemon.py`, `camera_loader.py`,
`camera.py`, `pyspin.py`), update and restart the daemon on every capture PC.

## Related

- [`paradex/io/camera_system/camera_server_daemon.py`](../../paradex/io/camera_system/camera_server_daemon.py)
- [`paradex/io/camera_system/camera_loader.py`](../../paradex/io/camera_system/camera_loader.py)
- [`paradex/io/camera_system/camera.py`](../../paradex/io/camera_system/camera.py)
- [`paradex/io/camera_system/pyspin.py`](../../paradex/io/camera_system/pyspin.py)
- Main-PC capture scripts: [`../capture/camera`](../capture/camera)
