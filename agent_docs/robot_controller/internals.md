# Robot Controllers — Internals (for agents editing this module)

**You are here because you are changing a controller's *internals* (control loop /
fault handling / logging / lifecycle), not just calling it.** If you only want to
*move* a device from another program, read [`usage.md`](usage.md) instead.

Function-level detail lives in each `.py`'s docstrings; this file is the glue: the
shared control-loop shape, the fault-handling contract you must not break, and the
traps.

---

## 1. The shared shape (every controller looks like this)

```
__init__            connect device, spawn ONE daemon loop thread (~100 Hz)
  loop thread:      while not exit_event: write latest target → device; if recording, read state → buffers
  move(target)      swap shared target under self.lock  (cheap, non-blocking)
  start/stop        arm/flush the per-tick recording buffers → *.npy
  end               set exit_event, join(timeout), disconnect
```

| Controller | File | Loop method | Transport |
|-----------|------|-------------|-----------|
| `XArmController` | [`xarm_controller.py`](../../paradex/io/robot_controller/xarm_controller.py) | `control_loop` → `_control_step` | XArm SDK / TCP |
| `AllegroController` | [`allegro_controller.py`](../../paradex/io/robot_controller/allegro_controller.py) | `_spin_loop` + `control_loop` | ROS2 `rclpy` |
| `InspireControllerIP` | [`inspire_controller_ip.py`](../../paradex/io/robot_controller/inspire_controller_ip.py) | `move_hand` → `_hand_step` | Modbus TCP (`pymodbus`) |
| `InspireController` | [`inspire_controller.py`](../../paradex/io/robot_controller/inspire_controller.py) | `control_loop` | USB serial |
| Factory | [`__init__.py`](../../paradex/io/robot_controller/__init__.py) | `get_arm` / `get_hand` | resolve name → class via `network_info` |

---

## 2. The fault-handling contract (do NOT regress this)

A controller drives real hardware from a background thread. The bug this design
kills: **a device errors → the loop thread dies or blocks → the whole program
hangs, unresponsive even to `Ctrl-C`.** Four guarantees prevent it. If you edit a
loop, preserve all four.

| # | Guarantee | How it's implemented | If you break it |
|---|-----------|----------------------|-----------------|
| 1 | An IO error never kills the loop thread | the per-tick body is a separate method (`_control_step`/`_hand_step`) **called inside `try/except`**; on exception → `logger.exception`, set `error_event`, wake waiters, `continue` | one socket blip silently kills the thread; callers hang forever |
| 2 | A device fault halts output, not the program | check `arm.has_err_warn` each tick → log `error_code`/`warn_code`/`state` once, set `error_event`, wake `position_control_event`, `continue` (no auto-clear) | loop hammers a dead arm; or fault goes unnoticed |
| 3 | `Ctrl-C` is always deliverable | every main-thread `wait()`/`join()` has a **timeout** and re-checks `exit_event`/`error_event` (e.g. `move`'s `while not position_control_event.wait(0.2)`) | a no-timeout wait blocks in C and defers `KeyboardInterrupt` → unkillable |
| 4 | A stuck thread can't block exit | loop threads are `daemon=True` | a wedged thread keeps the process alive after main exits |

### Why both #1 and #2 (not just `try/except`)

The XArm SDK reports **most servo failures as return codes / `has_err_warn`, not
Python exceptions**. So `try/except` alone misses coded faults, and `has_err_warn`
alone misses real exceptions (socket drop, `report_type="devlop"` thread, modbus).
**Both are required.** This was the crux of a real debugging session: the visible
symptom (unkillable process) came from an *unhandled* IO error killing the thread —
the `try/except` is the fix, the `has_err_warn` check is the completion.

### The refactor pattern

To wrap a loop body in `try/except` without re-indenting a 60-line block, the body
was **extracted into a method** (`_control_step`, `_hand_step`) and the loop became:

```python
while not self.exit_event.is_set():
    if self.arm.has_err_warn:            # guarantee #2 (xarm only)
        ... log once, set events, continue
    try:
        self._control_step()             # guarantee #1
    except Exception as e:
        logger.exception("... IO error: %s", e)
        self.error_event.set(); self.position_control_event.set()
        time.sleep(0.05); continue
    ... pace to fps ...
```

Keep the extracted step method pure per-tick work; keep the fault/except handling
in the loop.

---

## 3. Logging

`paradex.utils.log.get_logger(name)` → console **+** a per-PC/per-day NAS file:
`~/shared_data/log/<pc_name>/<YYYYMMDD>/<name>_<HHMMSS>.log`. It is idempotent per
process (one file per run), and degrades to console-only if the NAS is down (never
raises). Module-level `logger = get_logger("xarm" | "inspire")`. **Do not** add
`print()` — route through the logger so faults land on the NAS. The fault log line
(`error_code`/`warn_code`/`state`) is the primary diagnosis tool; keep it.

---

## 4. XArm control loop specifics

- Two modes dispatched per `move` inside `_control_step`, on `action.shape`:
  `(6,)` joint → `set_servo_angle_j` (servo) / `set_servo_angle` (position);
  `(4,4)` pose → `set_servo_cartesian_aa` (servo) / `set_position` (position).
- **No interpolation.** The loop writes whatever `self.action` is. Smoothness is
  the caller's job; a large single-step target can fault the arm (overspeed).
- **Position mode** (`is_servo=False`) toggles `set_mode(0)/set_state(0)`, does one
  blocking `wait=True` move, restores servo, sets `position_control_event`. `move`
  blocks on that event (with timeout — guarantee #3).
- Recovery: `clear_error()` (clean warn/err + `motion_enable` + servo mode, no
  reconnect) vs. `reset()` (new `XArmAPI`, full re-init). `is_error()` = `error_event`.
- **`np.clip(action, -2π, 2π)`** in `move` is a *cosmetic* clamp — real XArm6 limits
  are tighter (J2/J3/J5, see [`docs/robot.md`](../../docs/robot.md) §12). It does not
  prevent out-of-range faults.

---

## 5. Inspire (IP) control loop specifics

- `move_hand` sets up, then loops `_hand_step`: read target under lock →
  `write6('angleSet', ...)` → `read6('angleAct'/'forceAct')` (+ tactile if enabled)
  → append to buffers if recording. Wrapped per guarantee #1.
- No `error_event`/`is_error()` — a raised modbus error is logged and the loop
  retries. Extra helpers: `home_robot`, `set_homepose`, `calibrate_force`
  (writes the force-cal register, polls until firmware clears it).

---

## 6. Traps that look like bugs but aren't

- **`dataset_acqusition` typo** is intentional repo-wide — don't "fix" imports.
- **`_hand_step` is defined after `calibrate_force`**, not right after `move_hand`
  — Python doesn't care about method order; leave it.
- **`camera.py` needs no `try/except` wrapper** — its `acquire`/`single_acquire`/
  `connect_camera` already self-guard (`except → event["error"]`). Don't add one.
- **A 0-byte `action_qpos.npy`** is the fingerprint of a process killed mid-`stop()`
  (`np.save` truncated) — i.e. a crash/SIGKILL during shutdown, not a save bug.
- **Allegro `get_data` reorders** raw joints through `JS_TO_CMD`; `qpos` is in
  command order, not `/joint_states` order.

---

## 7. Stale import paths / config keys (real breakage)

These are genuine bugs found while tracing consumers — fix at the call site, don't
"work around" them here:

- **`gui_controller_prev` is gone.** Several `src/inference/*` scripts still do
  `from paradex.io.robot_controller.gui_controller_prev import RobotGUIController`,
  but `gui_controller_prev.py` now lives under `deprecated/` → those imports raise
  `ImportError`. The live GUI is `gui_controller.py` (`RobotGUIController(robot, hand)`).
- **`get_hand("inspire_left")` needs a `network_info["inspire_left"]` entry.** The
  factory reads `network_info[hand_name]` directly for the IP path, but the shipped
  `system/current/network.json` only defines `inspire` — so `inspire_left` (used by
  `src/validate/robot/inspire_left_overlay.py`) `KeyError`s without adding that config key.
- **Config schema is inconsistent between factory branches.** Inspire-IP reads
  `network_info["inspire"]` **directly** (`ip`,`port`), while xArm / USB-Inspire /
  Allegro read `network_info[name]["param"]`. Match whichever branch you copy when
  adding a device.
