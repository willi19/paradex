# Robot Controllers — How to Use (for humans & agents)

Read this before writing code that moves an arm or hand. Full mental model with
diagrams: [`docs/robot.md`](../../docs/robot.md). Per-method API:
[`docs/robot_api.md`](../../docs/robot_api.md).

> **Editing a controller's internals** (control loop / fault handling / logging),
> not just calling it? Read [`internals.md`](internals.md) instead.

## TL;DR

```python
from paradex.io.robot_controller import get_arm, get_hand

arm  = get_arm("xarm")                 # connects + spawns a ~100 Hz loop thread
hand = get_hand("inspire_left")        # or "allegro", "inspire"

arm.move(qpos_rad)                     # (6,) joint radians — servo stream (non-blocking)
arm.move(pose_4x4, is_servo=False)     # one-shot, BLOCKS until the arm arrives
hand.move(np.zeros(6)+800)             # inspire: 0–1000 motor units (per device)

arm.start("dataset/001/arm")           # begin recording state → *.npy each tick
# ... motion ...
arm.stop()                             # flush buffers to time/position/action/... .npy
arm.end()                              # join loop thread, disconnect (calls stop() if open)
```

You never import a concrete controller class — `get_arm`/`get_hand` resolve the
name through `system/current/network.json` (`network_info`).

## The one thing to understand: the loop owns the device

Construction spawns a background thread that writes the **latest target** to the
device at ~100 Hz. `move()` does **not** talk to hardware — it just swaps that
shared target under a lock (cheap, thread-safe). The loop does the talking.

- **Servo** (`is_servo=True`, default): streams the target continuously. Returns
  immediately. Feed it a **smooth stream of small steps** — it does *not*
  interpolate, so a single large jump between `move()` calls asks the arm to
  cross it in one 10 ms tick and can trip an overspeed/kinematic fault.
- **Position** (`is_servo=False`): switches to position mode, issues one blocking
  move, waits for arrival, returns to servo. `move()` **blocks** until done.

## Devices

| Name | Class | Action | Units |
|------|-------|--------|-------|
| `xarm` | `XArmController` | `(6,)` joint **or** `(4,4)` pose | radians / homogeneous |
| `allegro` | `AllegroController` | `(16,)` | radians (clip `MAX_ANGLE=2.1`) |
| `inspire` / `inspire_left` | `InspireControllerIP` | `(6,)` | 0–1000 motor units (0 closed, 1000 open) |

`get_hand("inspire", ip=True)` → Modbus TCP (`network_info["inspire"]`).
`ip=False` → USB `InspireController` (`network_info["inspire_usb"]`).
`tactile=True` also reads the 17-pad tactile grid each tick.

## Recording

`start(save_path)` arms per-tick logging; `stop()` dumps `time.npy`,
`position.npy`, `velocity.npy`, `torque.npy`, `action.npy`, `action_qpos.npy`
(arm) into `save_path`. Buffers only fill **between** `start` and `stop`. `end()`
auto-`stop()`s if a session is still open.

## When a device faults (the important part)

A controller **cannot** be made to hang the whole program anymore. If the device
errors:

- The loop **keeps running** (the fault is caught, not fatal) and **stops sending
  commands** to the dead device.
- The error is **logged** (console + NAS, see below) with the device's real error
  code — this is your diagnosis.
- Any blocked `move()` is released; `Ctrl-C` still works.

It does **not** auto-recover (auto-clearing could re-drive into the same fault).
You recover explicitly:

```python
if arm.is_error():          # True after a fault
    arm.clear_error()       # clear warn/err + re-enable servo, no reconnect
    # arm.reset()           # heavier: full disconnect + reconnect
```

Inspire's loop is likewise fault-tolerant (logs + stays alive); it has no
`is_error()`/`clear_error()` — a raised IO error is logged and the loop retries.

## Logs

Everything logs through `paradex.utils.log.get_logger` → console **and**:

```
~/shared_data/log/<pc_name>/<YYYYMMDD>/<name>_<HHMMSS>.log
```

Logger names: `xarm`, `inspire`. **After a fault, read this file** — the arm's
`error_code`/`warn_code`/`state` are there, which tells you *why* (joint-range vs.
overspeed vs. singularity vs. collision) instead of guessing.

## Gotchas

- **Feed servo small steps.** No interpolation; large single jumps fault the arm.
- **`±2π` joint clamp is cosmetic.** Real XArm6 limits are tighter (J2/J3/J5); a
  target inside `±2π` but outside the real range is rejected by the arm itself.
- **Franka is stubbed** — only `xarm` returns an arm today.
- **Units differ per device** — arm radians/pose vs. Allegro radians vs. Inspire
  0–1000. Don't cross them.
- **Validate scripts** live in `src/validate/robot/` — run those to smoke-test a
  device without writing new code.
