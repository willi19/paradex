# agent_docs/robot_controller — agent orientation

Docs for **AI agents working on `paradex/io/robot_controller/`** — the layer that drives the
physical **robot arm** (xArm) and **robot hand** (Allegro / Inspire). Every controller is a thin
wrapper around a vendor SDK or bus protocol, spun up with a **background daemon thread** that
streams a target buffer to the device at ~100 Hz and (optionally) logs state to `.npy`. You never
instantiate the classes by name in application code — go through the two factory functions in
[`__init__.py`](../../paradex/io/robot_controller/__init__.py). Read this one file for the whole
module; the per-class sections below are the reference. (A `usage.md` / `internals.md` split also
exists in this folder — this README is the self-contained overview.)

Mental model: **`get_arm`/`get_hand` → a controller with a daemon thread + a target buffer.**
`move()` just writes the buffer (non-blocking); the thread pushes it to hardware every tick and
mirrors device readback into `get_data()`. `start(path)`→record ticks→`stop()`→dump `.npy`;
`end()` joins the thread and disconnects. Units and DOF differ per device — see the table.

## File map
| File | What it is |
|------|-----------|
| `__init__.py` | The public API: `get_arm(name)` and `get_hand(name, tactile, ip)` factories. Reads device params from `network_info` (i.e. `system/current/network.json`). |
| `xarm_controller.py` | `XArmController` — UFactory xArm arm over the `xarm` SDK (`XArmAPI`). Joint **or** cartesian, servo-streaming **or** blocking position moves. |
| `allegro_controller.py` | `AllegroController` — Allegro 16-DOF hand as a **ROS2 node** (`rclpy`). Subscribes `/joint_states`, publishes `/allegro_hand_position_controller/commands`. |
| `inspire_controller_ip.py` | `InspireControllerIP` — Inspire 6-DOF hand over **Modbus TCP** (`pymodbus`). The default hand path (`ip=True`). Optional tactile sensors. |
| `inspire_controller.py` | `InspireController` — same Inspire hand over **USB serial** (`pyserial`), custom byte framing. Legacy path (`ip=False`). |
| `gui_controller.py`, `gui_controller_simple.py` | Tkinter `RobotGUIController` — a **consumer** of the controllers above (waypoint teleop UI), not part of the arm/hand API. |
| `deprecated/`, `under_test/` | Old GUI variants and WIP tactile/CLI experiments. Not imported by the factories. See gotchas. |

All paths relative to [`paradex/io/robot_controller/`](../../paradex/io/robot_controller/).
Franka (mentioned in the top-level `CLAUDE.md`) is **not implemented** — its branch is commented
out in `get_arm`.

---

## Who calls this
| Caller | Uses |
|--------|------|
| [`paradex/dataset_acqusition/capture.py`](../../paradex/dataset_acqusition/capture.py) (`CaptureSession`) | `get_arm(arm)` / `get_hand(hand, ip=hand_ip)`; drives the record lifecycle `start(path)` → `stop()` → `end()`. The main integration point. |
| [`src/dataset_acquisition/graphics/image_capture.py`](../../src/dataset_acquisition/graphics/image_capture.py) | `get_arm("xarm")`; also imports `XArmController` directly in `image_traj.py`. |
| [`src/calibration/handeye/capture.py`](../../src/calibration/handeye/capture.py) | `XArmController` — replays saved `_qpos` waypoints for hand-eye calibration. |
| [`src/util/robot/visualize.py`](../../src/util/robot/visualize.py) | `get_arm(args.arm)` — read live `get_data()` for Viser replay. |
| [`src/validate/robot/allegro.py`](../../src/validate/robot/allegro.py) + sibling `inspire*.py` / `xarm_base_wiggle.py` | Per-device smoke tests via `get_hand` / `get_arm` / direct class imports. |
| [`src/validate/robot_controller/xarm_test.py`](../../src/validate/robot_controller/xarm_test.py) | `XArmController(**network_info["xarm"]["param"])` — direct cartesian wiggle test. |
| `src/inference/**` (`grasp_eval`, `grasp_w_gui`, `bodex`, `pringles_test`) | `get_arm("xarm")` + `get_hand("allegro")` fed into `RobotGUIController` for grasp execution. |

So `CaptureSession` and the `src/inference/*` grasp scripts are the primary consumers; everything
else is a validation/utility script.

---

## The factories (`__init__.py`)

```python
from paradex.io.robot_controller import get_arm, get_hand
arm  = get_arm("xarm")                       # -> XArmController
hand = get_hand("allegro")                   # -> AllegroController
hand = get_hand("inspire")                   # -> InspireControllerIP  (ip=True default)
hand = get_hand("inspire", ip=False)         # -> InspireController     (USB serial)
hand = get_hand("inspire", tactile=True)     # -> InspireControllerIP with tactile reads
```

- `get_arm(arm_name)` — only `"xarm"` is wired; constructs `XArmController(**network_info["xarm"]["param"])` (i.e. `ip=...`).
- `get_hand(hand_name, tactile=False, ip=True)`:
  - `"inspire"` / `"inspire_left"` + `ip=True` → `InspireControllerIP(**network_info[hand_name], tactile=tactile)`. **Note the config schema differs**: the ip/port come from `network_info["inspire"]` directly (keys `ip`, `port`), *not* from a nested `["param"]`.
  - `"inspire"` + `ip=False` → `InspireController(**network_info["inspire_usb"]["param"])` (a serial `addr`).
  - `"allegro"` → `AllegroController(**network_info["allegro"]["param"])`; the constructor takes **no device args** (all addressing is ROS2 topics), so that `param` must be `{}`.

Device addresses are **not** hardcoded in these classes — they come from
`system/current/network.json` via `paradex.utils.system.network_info` (e.g. `xarm` `192.168.1.196`,
`inspire` `192.168.11.210:6000`).

## Common shape of every controller
All four device classes share the same lifecycle contract (used by `CaptureSession`):

| Method | Behavior |
|--------|----------|
| `__init__(...)` | Connects to the device and starts a **daemon thread** running a ~100 Hz control loop. Some block until a `connection_event`/first readback. |
| `move(action)` | **Non-blocking**: copies `action` into the shared target buffer under a lock; the loop applies it next tick. (xArm `is_servo=False` is the one exception — it blocks.) |
| `get_data()` | dict `{"qpos"/"joint_value", "action", "time", ...}` — latest device readback + current target. |
| `start(save_path)` | Begin logging each loop tick into an in-memory buffer. |
| `stop()` | Stop logging and dump each field to `save_path/<field>.npy`. |
| `end()` | Signal exit, join the thread, disconnect; auto-`stop()`s if still recording. |

---

## `XArmController` (`xarm_controller.py`)
UFactory xArm via `xarm.wrapper.XArmAPI`. `control_loop` runs at `fps=100` on a daemon thread.

- **`move(action, is_servo=True, speed=None)`** — `action` is either:
  - **joint** `np.ndarray` shape `(6,)` in **radians** (clipped to `±2π` before sending), or
  - **cartesian** homogeneous matrix shape `(4,4)` in **meters** (translation is `×1000`→mm internally; rotation → euler `xyz` / axis-angle radians for the SDK).
  - `is_servo=True` (default) → `set_servo_angle_j` / `set_servo_cartesian_aa` **streaming, non-blocking**.
  - `is_servo=False` → SDK mode 0 position move with `wait=True`; **blocks** the caller until the loop signals `position_control_event` (polled with a 0.2 s timeout so Ctrl-C / faults release it). `speed` only applies to the joint blocking path.
- **`get_data()`** → `{"qpos": (6,) rad, "position": (4,4) homogeneous in meters, "time"}`.
- **`start(save_path)` / `stop()`** — logs `time, position(6), velocity(6), torque(6), action(cart 6-vec), action_qpos(6)` to `.npy`. On save it computes FK (joint moves) or IK (cartesian moves; logs a warning + `-1` vector on IK failure).
- **Error handling**: if `arm.has_err_warn` (e.g. singularity/overspeed), the loop **stops sending servo commands**, sets `error_event`, and wakes any blocked `move`. Recovery is **manual** — call `clear_error()` (clean warn/error + re-enable servo) or `reset()` (full reconnect); it will not auto-clear (that could re-drive into the same fault). `is_error()` exposes the flag.
- **`end(set_break=False)`** — joins the loop, optionally `motion_enable(False)`, disconnects.

## `AllegroController` (`allegro_controller.py`)
Allegro 16-DOF hand as a **ROS2 node**. `action_dof=16`.

- Publishes `Float64MultiArray` on `/allegro_hand_position_controller/commands`; subscribes `JointState` on `/joint_states`. The `ForwardCommandController` expects **absolute joint positions in radians, in the controller's joint order**.
- **`move(action)`** — `action` shape `(16,)`; non-blocking (sets target). The 100 Hz `control_loop` publishes it every tick, **clipped to `MAX_ANGLE=2.1` rad**.
- **`get_data()`** → `{"qpos": joint_value reordered by JS_TO_CMD, "action": target, "time"}`. `JS_TO_CMD` remaps the `/joint_states` order into command order — the raw subscribed `joint_value` is in ROS joint order.
- The first `/joint_states` message seeds the target to the current pose and fires `connection_event`; until then `move`/publish warn "No joint data received yet".
- `start`/`stop` log `action(16), position(16), time`. `end()` shuts down the executor and joins both the spin thread and control thread.

## `InspireControllerIP` (`inspire_controller_ip.py`) — default Inspire path
Inspire 6-DOF hand over **Modbus TCP** (`ModbusTcpClient`, `pymodbus==2.5.3` per the source note).
`action_dof=6`. Background `move_hand` thread at `fps=100`.

- **Angle units are integers `0–1000`, NOT radians/meters** (mapping is device-defined). `home_pose` defaults to `800`. `move(action)` sets the 6-vector `target_action`; the loop writes `angleSet` and reads back `angleAct`/`forceAct`.
- On construction it presets `speedSet=1000`, `forceSet=400`, `angleSet=1000` for all 6 joints.
- **Readers**: `get_qpos()` (`angleAct`), `get_data()` (`{qpos, action, time}`), `get_force()` (`forceAct`, converted uint16→signed int16).
- **Homing**: `set_homepose(pose)` / `home_robot(pose=None)` (moves target to home).
- **Tactile** (only if constructed with `tactile=True`): `read_tactile(name)`, `read_all_tactile()`, `read_all_tactile_raw()`, `get_tactile()` over the `TACTILE_LAYOUT` register map. `get_tactile()` raises if tactile mode is disabled.
- **`calibrate_force(timeout, poll_interval)`** — writes `forceClb`; hand must be open/untouched; polls until firmware flips the register. Raises on timeout.
- **`start` / `stop` / `save` / `end`** — logs `time, position, action, force` (+ `tactile` if enabled). `stop()` calls `save()` to flush `.npy`.
- The loop wraps each step in try/except and logs (never dies) so a dead device doesn't hang the program.

## `InspireController` (`inspire_controller.py`) — USB serial path
Same Inspire hand over **USB serial** (`serial.Serial(addr, 115200)`) with hand-rolled byte framing
(`command` opcode table, `write6`/`read6`, checksum). `action_dof=6`, angles `0–1000`, `home_pose=800`.

- `__init__` **blocks** on `connection_event.wait()` until the loop has opened the port and preset speed/power/angle.
- `move(action)`, `get_data()` (`{action, joint_value, time}`), `get_force()`, `start`/`stop`/`end`.
- `is_error()` always returns `False`. Chosen via `get_hand("inspire", ip=False)`.

---

## Gotchas for editors
- **`allegro_controller.py` calls `rclpy.init()` at module import time** (top level). Merely importing the module initializes ROS2 globally — importing it twice, or after another `rclpy.init()`, will raise. The factory imports it lazily inside `get_hand` to contain this, so keep it lazy.
- **Inspire angles are integers `0–1000`, not SI units.** xArm joints are radians, Allegro joints are radians (clipped to 2.1), Inspire is a `0–1000` count. Don't assume a shared unit across hands.
- **Config schema is inconsistent between factory branches.** Inspire-IP reads `network_info["inspire"]` **directly** (`ip`,`port`), while xArm / USB-Inspire / Allegro read `network_info[name]["param"]`. Adding a device? Match whichever branch you copy from.
- **`get_hand("inspire_left")` needs its own `network_info["inspire_left"]` entry** — the shipped `system/current/network.json` only has `inspire`; `inspire_left` (used by `src/validate/robot/inspire_left_overlay.py`) will `KeyError` without that config key.
- **`move()` is fire-and-forget for hands and for servo arm moves** — it only updates a buffer; there's no "reached target" signal except xArm's `is_servo=False` blocking path. Callers do their own convergence checking (see `RobotGUIController`'s velocity-limited stepping).
- **xArm faults latch and halt output.** After `has_err_warn`, servo commands stop and blocked `move`s are released, but nothing recovers automatically — you must call `clear_error()` or `reset()`. `is_error()` is the flag to poll.
- **Franka is not implemented** despite the top-level `CLAUDE.md` mention — `get_arm("franka")` returns `None` (commented-out branch).
- **`RobotGUIController` is imported from a moved path.** Several `src/inference/*` scripts do `from paradex.io.robot_controller.gui_controller_prev import RobotGUIController`, but `gui_controller_prev.py` now lives under `deprecated/` — those imports are broken (`ImportError`). The live GUI is `gui_controller.py` (`RobotGUIController(robot, hand)`).
- **`under_test/` and `deprecated/` are not wired into the factories** — treat them as scratch (WIP tactile/CLI demos, old GUIs). Don't import from them in library code.
- **Every controller owns a daemon thread.** Always pair construction with `end()` (join + disconnect); `CaptureSession.end()` does this. Skipping it leaves a live socket/serial/ROS node and a spinning thread.
