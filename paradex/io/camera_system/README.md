# Camera System — How to Use (for humans & agents)

Read this before writing code that captures images/video. It exists to stop
callers from flailing. Full reference: the online docs
(`willi19.github.io/paradex/camera_system.html`) and
[`camera_system_api`](https://github.com/willi19/paradex/blob/main/docs/camera_system_api.md).

## TL;DR — which class do I use?

- **On the MAIN PC (almost always you):** use `remote_camera_controller` (`rcc`).
  It drives the camera daemons on every capture PC over the network. **Do not**
  instantiate `Camera` / `CameraLoader` / `PyspinCamera` on the main PC.
- **On a capture PC:** the daemon (`src/camera/server_daemon.py`) already runs
  `CameraLoader`. You normally don't write capture-PC code.

## Prerequisite

`server_daemon.py` must already be running on every capture PC, or `rcc`
construction raises `ConnectionError` naming the unreachable PC(s). Start/monitor
them with the dashboard (`monitor_daemon.py`) or `src/camera/reset_cameras.py`.

## Running the daemon (start / stop / restart)

The daemon (`src/camera/server_daemon.py`) is the capture-PC process `rcc` talks to.

```bash
# start (on each capture PC)
python src/camera/server_daemon.py
PARADEX_CAMERA_IDLE_TIMEOUT_S=3 python src/camera/server_daemon.py   # shorter dead-man

# start them all from the main PC (SSH)
python src/camera/monitor_daemon.py     # dashboard; auto-starts missing daemons
python src/camera/reset_cameras.py      # pkill -9 + relaunch on every capture PC
```

| Stop with | Result |
|-----------|--------|
| `Ctrl-C` / `kill <pid>` (SIGTERM) | clean — the signal handler runs `shutdown()` → DeInit + free SHM |
| `pkill -9 -f server_daemon.py` (SIGKILL) | forced — no cleanup (SHM self-heals on next start; camera may need re-force) |

**Restart / easy cleanup:** `python src/camera/reset_cameras.py` (main PC), or the
`↻ restart` buttons in the monitor dashboard. **After a `git pull`, restart the daemons**
so code changes take effect — `reset_cameras.py` does that in one shot.

## The 90% recipe

```python
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

rcc = remote_camera_controller("my_app")          # background thread; connects + locks the daemons
rcc.start("image", False, save_path="dataset/001/raw")   # begin capture (blocks until sent)
# ... do work (move robot, wait, etc.) ...
rcc.stop()                                          # stop capture
rcc.end()                                           # RELEASE THE LOCK — always call this
```

### `start(mode, syncMode, save_path=None, fps=30, exposure_time=None, gain=None)`

| param | meaning |
|-------|---------|
| `mode` | `image` (stills) / `video` (.avi) / `stream` (SHM only) / `full` (video **and** SHM) |
| `syncMode` | `True` = wait for the hardware trigger (UTGE900 must be running); `False` = free-run at `fps` |
| `save_path` | session-relative dir for `image`/`video`/`full`; omit for `stream` |
| `fps` | frame rate (free-run) |
| `exposure_time`, `gain` | `None` → use the per-camera `camera.json` baseline (recommended) |

## Detecting failures (do this in long captures)

`rcc` now surfaces problems — poll it instead of assuming success:

```python
st = rcc.get_status()
# {'error': bool, 'stalled': [serial, ...], 'pc': {pc: {'status','msg'}}}
if st['error'] or st['stalled']:
    print("camera trouble:", st)
    rcc.reload_cameras()      # or abort the trial
```

- `error` — a camera raised (disconnect, arm failure). `pc[..]['msg']` has the message.
- `stalled` — frames stopped arriving (dead trigger / LAN drop) though no exception.
- `rcc.is_error()` is the same flag as a bool. Both are **live** (reflect the current tick).

## Recovery / cleanup

- **Cameras wedged / won't restart:** `python src/camera/reset_cameras.py` (main PC) —
  force-kills + relaunches every daemon. Or the "restart" buttons in the monitor.
- **Another controller holds the lock:** `rcc.force_takeover()`.
- **Your app crashed:** the daemon auto-releases the lock and stops cameras after the
  idle timeout (~5 s, `PARADEX_CAMERA_IDLE_TIMEOUT_S`). Still, prefer calling `end()`.

## Gotchas (the things people get wrong)

- `rcc.start()`/`stop()` only **set events**; the background `run()` loop does the send.
- The string arg to `remote_camera_controller(name)` is just a **label**; there is no
  `<name>.py` file.
- `stream` needs no `save_path`; `image`/`video`/`full` require one.
- `syncMode=True` produces no frames unless the UTGE900 trigger is running.
- **Always `end()`** to release the lock (or the daemon frees it after the idle timeout).
- Per-camera `gain`/`exposure`/`packet_size`/`pixel_format`/... live in
  `system/current/camera.json`; pass `None` to `start()` to use them.

## Where the code lives

| Layer | File |
|-------|------|
| main-PC driver (`rcc`) | `remote_camera_controller.py` |
| capture-PC daemon | `camera_server_daemon.py` (entry: `src/camera/server_daemon.py`) |
| group of cameras | `camera_loader.py` |
| one camera + capture thread | `camera.py` |
| PySpin driver | `pyspin.py` |
| dashboard | `monitor_daemon.py` |
| recovery | `src/camera/reset_cameras.py` |

## Improving this subsystem later

Design/roadmap (acquisition–sink decoupling, known limitations, pending fixes):
[`design/camera-recording-redesign.md`](https://github.com/willi19/paradex/blob/main/design/camera-recording-redesign.md).
