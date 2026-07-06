# Camera System — How to Use (for humans & agents)

Read this before writing code that captures images/video. It exists to stop
callers from flailing. Full reference: the online docs
(`willi19.github.io/paradex/camera_system.html`) and
[`camera_system_api`](https://github.com/willi19/paradex/blob/main/docs/camera_system_api.md).

> **Editing the camera code itself** (daemon / acquisition / pyspin / lifecycle),
> not just calling it? Read [`internals.md`](internals.md) instead.

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

**Stop only (no relaunch):** `python src/camera/reset_cameras.py --no_restart` (main PC),
or the `■ stop` / `■ stop all` buttons in the monitor dashboard.

## The model: two modes + orthogonal sinks

There are exactly **two acquisition modes**, and on top of continuous acquisition
the outputs are independent **sinks** you toggle any time:

- **`image`** — one PySpin single-frame still. `rcc.start("image", syncMode, save_path)`.
- **`acquire`** — continuous acquisition (`rcc.arm(...)`). Nothing is written until you
  turn on a sink:
  - **video** sink → `.avi` on disk: `rcc.set_record(save_path, on=True/False)`
  - **stream** sink → shared memory (read by `MultiCameraReader`): `rcc.set_stream(on=True/False)`
  - **snapshot** sink → next N frames as images: `rcc.snapshot(save_path, count=1)`

`video` / `stream` / `full` are **not modes** anymore — `full` was just record+stream.

## The 90% recipe

```python
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

rcc = remote_camera_controller("my_app")     # background thread; connects + locks the daemons

# --- continuous capture, sinks toggled live ---
rcc.arm(syncMode=False, fps=30)              # cameras acquire; no output yet
rcc.set_stream(True)                         # live SHM preview on
rcc.set_record("dataset/001/raw", on=True)   # start recording .avi (any time)
# ... do work (move robot, wait) ...
rcc.set_record(on=False)                     # stop recording; acquisition keeps running
rcc.stop()                                   # disarm
rcc.end()                                    # RELEASE THE LOCK — always call this

# --- one still per camera ---
rcc.start("image", False, "dataset/001/raw"); rcc.stop()
# or, while armed: rcc.snapshot("dataset/001/raw", count=1)
```

| call | what it does |
|------|--------------|
| `arm(syncMode=False, fps=30, exposure_time=None, gain=None)` | begin continuous acquisition, no sink |
| `set_record(save_path, on=True)` | video `.avi` sink on/off (live) |
| `set_stream(on=True)` | SHM stream sink on/off (live) |
| `snapshot(save_path, count=1)` | write the next `count` frames as images |
| `start("image", syncMode, save_path)` | single-frame still (its own mode) |
| `stop()` / `end()` | disarm / disarm + release the lock |

- `syncMode=True` → wait for the UTGE900 hardware trigger; `False` → free-run at `fps`.
- `exposure_time` / `gain` `None` → per-camera `camera.json` baseline (recommended).

## Detecting failures (do this in long captures)

`rcc` detects both **live** problems and a **sticky** capture interruption — poll it:

```python
st = rcc.get_status()
# {'error': bool, 'stalled': [serial,...], 'capture_interrupted': bool,
#  'interrupt_msg': str|None, 'pc': {pc: {'status','msg','states','frame_ids',...}}}
if rcc.capture_interrupted():          # a daemon died/restarted mid-recording
    abort_or_restart(st['interrupt_msg'])   # NOT auto-resumed; sticky until next arm()/start()
elif st['error'] or st['stalled']:     # live: down PC / camera error / frozen frames
    handle(st)
```

- `error` (live) — a PC is down (PUB silent >2 s), a camera raised, or detected<expected cameras.
- `stalled` (live) — frame ids stopped advancing while capturing (dead trigger / LAN drop).
- `capture_interrupted` (**sticky**) — a daemon went down/restarted while you were
  capturing. It stays set through the daemon's recovery + rcc's auto re-register, so an
  interrupted recording never silently looks healthy. Cleared only by the next `arm()`/`start()`.
- `rcc.is_error()` = `error` **or** `capture_interrupted`, as a bool.

### What each side detects if the other dies

- **daemon dies → rcc knows** within ~2 s (PUB silence) and commands to it time out
  (never hang); if you were recording, `capture_interrupted` latches.
- **rcc/your app dies → the daemon knows** within ~5 s (dead-man): it stops the cameras
  and releases the lock on its own (`PARADEX_CAMERA_IDLE_TIMEOUT_S`).
- **daemon restarted mid-session** → rcc auto re-registers to reclaim the lock (so later
  commands work), but does **not** auto-resume capture — you decide via `capture_interrupted`.

## Recovery / cleanup

- **Cameras wedged / won't restart:** `python src/camera/reset_cameras.py` (main PC) —
  force-kills + relaunches every daemon. Or the "restart" buttons in the monitor.
- **Another controller holds the lock:** register is now refused if a *different* live
  controller holds it — use `rcc.force_takeover()` to override on purpose.
- **A daemon was restarted:** rcc re-registers automatically (reclaims the lock); if you
  were recording, check `rcc.capture_interrupted()` and re-`arm()` if you want it back.
- **Your app crashed:** the daemon auto-releases the lock and stops cameras after the
  idle timeout (~5 s, `PARADEX_CAMERA_IDLE_TIMEOUT_S`). Still, prefer calling `end()`.

## Gotchas (the things people get wrong)

- `arm()` alone records **nothing** — you must turn on a sink (`set_record`/`set_stream`)
  or `snapshot()`. Turning a sink off does **not** stop acquisition; `stop()` does.
- `video` / `stream` / `full` are gone as modes; `Camera.start` rejects them. Use
  `arm()` + `set_*`. Only `image` is still a mode.
- The string arg to `remote_camera_controller(name)` is just a **label**; there is no
  `<name>.py` file.
- `syncMode=True` produces no frames unless the UTGE900 trigger is running.
- **Always `end()`** to release the lock (or the daemon frees it after the idle timeout).
- Per-camera `gain`/`exposure`/`packet_size`/`pixel_format`/... live in
  `system/current/camera.json`; pass `None` to use the baseline.
- Optional command auth: set the same `PARADEX_CAMERA_TOKEN` on every daemon **and** the
  controller to require it (off by default = accept any peer on the LAN).

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
