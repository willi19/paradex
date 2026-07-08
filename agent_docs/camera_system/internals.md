# Camera System — Internals (for agents editing this module)

**You are here because you are changing camera *internals* (daemon / acquisition /
pyspin / lifecycle), not just calling the API.** If you only want to *capture* from
another program, stop and read [`usage.md`](usage.md) instead — do not touch these files.

This document is the map that keeps you from flailing: what each layer owns, how a
command flows through them, which function to edit for a given change, and the traps
that look like bugs but aren't. Function-level detail lives in the NumPy-style
docstrings in each `.py`; this file is the glue between them.

---

## 1. The five layers (main PC → hardware)

```
remote_camera_controller   (MAIN PC)      network client + health, single-controller lock
        │  ZMQ 5482 REQ/REP (commands), 5481 SUB (status), 5480 REQ (ping)
        ▼
camera_server_daemon       (CAPTURE PC)   ZMQ server; owns ONE CameraLoader; lock + idle dead-man
        ▼
CameraLoader               (CAPTURE PC)   group of cameras; config resolution; fan-out start/stop
        ▼
Camera                     (CAPTURE PC)   one camera + its capture THREAD; event handshake; SHM/video sink
        ▼
PyspinCamera               (CAPTURE PC)   thin PySpin/Spinnaker driver: Init, node config, GetNextImage
```

**Golden rule:** `remote_camera_controller` runs on the main PC; everything below it
runs on a capture PC inside `server_daemon.py`. The two halves never share memory —
they only exchange JSON over ZMQ. If you're tempted to call `CameraLoader` from the
main PC, you're in the wrong layer (that's what the daemon is for).

| Layer | File | Owns | Key methods |
|-------|------|------|-------------|
| Controller | [`remote_camera_controller.py`](../../paradex/io/camera_system/remote_camera_controller.py) | network, lock, health polling | `start/stop/end`, `run` (bg loop), `get_status`, `_update_health` |
| Daemon | [`camera_server_daemon.py`](../../paradex/io/camera_system/camera_server_daemon.py) | ZMQ server, lock, idle timeout | `execute_command`, `command_thread`, `monitor_thread`, `pingpong_thread`, `shutdown` |
| Group | [`camera_loader.py`](../../paradex/io/camera_system/camera_loader.py) | per-camera config, fan-out | `load_pyspin_camera`, `start/stop/end`, `get_summary`, `get_all_errors` |
| One camera | [`camera.py`](../../paradex/io/camera_system/camera.py) | capture thread, events, sink | `start/stop/end`, `run`, `continuous_acquire`, `single_acquire`, `get_status` |
| Driver | [`pyspin.py`](../../paradex/io/camera_system/pyspin.py) | Spinnaker calls | `load_camera`, `PyspinCamera.start/get_image/stop`, `autoforce_ip` |

---

## 2. How one command flows (the thing to internalize)

`start()` is representative — trace it once and the rest follow the same shape.

1. **Main PC.** `rcc.start(mode, syncMode, save_path, ...)` does **not** send anything
   itself. It stores the args and sets an internal event. The background `run()` loop
   picks it up and does the actual ZMQ send. *(Gotcha #1: `start`/`stop` only set flags.)*
2. **Wire.** `run()` sends a JSON command on 5482 (REQ) to every capture PC's daemon.
3. **Capture PC.** `camera_server_daemon.command_thread` receives it and calls
   `execute_command`, which for `"start"` calls `CameraLoader.start(...)`.
4. **Group.** `CameraLoader.start` resolves each camera's config (see §4) and calls
   `Camera.start(...)` for every camera.
5. **One camera.** `Camera.start` sets the `start` event and **blocks up to `timeout`**
   waiting for the capture thread's `acquisition` event to confirm the camera actually
   armed. On timeout it clears `start` and **sets the `error` event** (deterministic
   failure — no silent half-start).
6. **Thread.** `Camera.run` (started once at construction) is parked on the `start`
   event. When set, it calls `PyspinCamera.start` (BeginAcquisition), signals
   `acquisition`, then loops in `continuous_acquire`/`single_acquire` grabbing frames
   into the SHM and/or video sink until the `stop` event.
7. **Back up.** Errors surface without polling the wire: `Camera.get_status` reports
   `state="ERROR"` + an `error` string → `CameraLoader.get_all_errors` /
   `get_summary` → daemon `monitor_thread` PUBs on 5481 and the heartbeat reply carries
   frame_ids+states → `rcc._update_health` sets `error`/`stalled` → `rcc.get_status()`.

**Stop/end/reload follow the same relay.** `end` additionally releases the
single-controller lock; forgetting it is why the next run can't register (or waits for
the idle dead-man).

---

## 3. The event handshake inside `Camera` (where lifecycle bugs live)

`Camera` runs a persistent thread (`run`) and communicates with it purely through
`threading.Event`s. Getting these wrong = hangs or half-started cameras, so this is the
most fragile spot in the module.

| Event | Set by | Meaning |
|-------|--------|---------|
| `start` | `start()` | thread should arm + begin acquiring |
| `acquisition` | thread | camera armed successfully (unblocks `start()`) |
| `stop` | `stop()` / thread on finish | thread should end acquisition |
| `exit` | `end()` | thread should terminate for good |
| `error` | thread **or** `start()` timeout | something failed; `get_error()` has the message |

Invariants you must not break when editing `run`/`start`/`*_acquire`:

- `start()` **always** resolves: either `acquisition` fires, or it times out and sets
  `error`. Never leave `start` set with no confirmation.
- `single_acquire` is wrapped in try/except/finally that **always sets `stop`** — a
  raise mid-grab must not leave the thread spinning.
- `continuous_acquire` guards `camera.stop()` so a driver already-stopped doesn't raise
  on the way out.
- A caught exception sets the `error` event *and* stashes the message
  (`get_error()`), so it propagates up to `rcc` (§2 step 7). Swallowing it = the
  "images just stopped and nobody noticed" class of bug.

---

## 4. Config resolution (who decides gain/exposure/packet_size/…)

Precedence, highest first — implemented in `CameraLoader` + `PyspinCamera`:

1. **Explicit arg** to `rcc.start(... exposure_time=, gain=)` — wins if not `None`.
2. **`system/current/camera.json[serial]`** — the per-camera baseline. Loaded on the
   capture PC by `paradex.utils.system`. Keys: `gain`, `exposure`, `packet_size`,
   `buffer_count`, `buffer_mode`, `pixel_format`.
3. **Driver default** — `PyspinCameraConfig` field defaults.

`CameraLoader.load_pyspin_camera` reads `camera.json` and passes each camera its own
`cfg` dict into `load_camera(serial, cfg=...)` → `PyspinCamera(..., cfg=cfg)`. So "make
camera X brighter" = edit `camera.json[X].gain`, not code. `system/current/` is
per-machine and **not** in git.

---

## 5. "I want to change X → edit Y"

| Change | Edit |
|--------|------|
| A capture-thread grab/timeout/arm bug (P4 hang) | `Camera.continuous_acquire` / `single_acquire`, `PyspinCamera.get_image` (timeout re-raise), `PyspinCamera.start` (fps-relative `grab_timeout_ms`) |
| Start/stop determinism, error propagation | `Camera.start` (timeout→error), `get_status`; then it flows up for free |
| A new per-camera hardware knob | `PyspinCameraConfig` field + read it in `PyspinCamera.start`; document the key in `camera.json` and in `usage.md` |
| Daemon command/lock/idle-timeout behavior | `camera_server_daemon.execute_command` / `command_thread`; idle via `PARADEX_CAMERA_IDLE_TIMEOUT_S` |
| What health the main PC sees | `Camera.get_status` → `CameraLoader.get_summary/get_all_errors` → daemon `monitor_thread` → `rcc._update_health/get_status`. Add a field at the bottom and thread it up. |
| IP recovery after power-cycle | `pyspin.autoforce_ip` (now called *inside* `CameraLoader`'s retry loop, not once before) |
| Dashboard readout / restart / stop buttons | `monitor_daemon.py` + `templates/monitor.html` |

Match the change to the layer. A fix in the wrong layer (e.g. retry logic in `rcc`
for a driver-level grab timeout) is the usual way this module accretes cruft.

---

## 6. Traps that look like bugs but aren't

1. `rcc.start()`/`stop()` set events; the send happens in `run()`. Don't "fix" the
   apparent no-op.
2. The string to `remote_camera_controller(name)` is a **label**, not a filename —
   there is no `<name>.py`.
3. `syncMode=True` produces **zero frames** unless the UTGE900 hardware trigger runs.
   A "stall" in sync mode is often just a dead trigger, not a code bug.
4. `SIGKILL` (`pkill -9`) leaves no cleanup; SHM self-heals on next start but a camera
   may need re-`autoforce_ip`. Prefer SIGTERM (`shutdown()` runs) when you can.
5. Ping (5480) = daemon liveness; heartbeat (on 5482) = controller keep-alive + health.
   Different channels — don't conflate them when debugging "why is it dead".
6. `camera_server_daemon.py` has thin docstrings *because* the logic is in
   `CameraLoader`/`Camera`; read those, not the daemon, for behavior.

---

## 7. Before you commit a change here

- No hardware in this environment: syntax-check (`python -m py_compile`) and reason
  through the event invariants (§3). Most changes here are **not** hardware-tested until
  the user runs the rig — say so in the commit/PR.
- If you add/rename a health field, thread it all the way up (§2 step 7) or the main PC
  never sees it.
