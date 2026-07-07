# agent_docs/robot_controller — agent orientation

Docs for **AI agents working on `paradex/io/robot_controller/`** — the layer that drives the
physical **robot arm** (xArm) and **robot hand** (Allegro / Inspire). Every controller is a thin
wrapper around a vendor SDK or bus protocol, spun up with a **background daemon thread** that streams
a target buffer to the device at ~100 Hz and (optionally) logs state to `.npy`. You never
instantiate the classes by name — go through the two factory functions in
[`__init__.py`](../../paradex/io/robot_controller/__init__.py). This is the *hardware* half of the
robot stack; the *computation* half (FK/IK/planning) is [`agent_docs/robot/`](../robot/README.md) —
the two meet only at `qpos`. Read the **one** file for your task.

Mental model: **`get_arm`/`get_hand` → a controller with a daemon thread + a target buffer.**
`move()` just writes the buffer (non-blocking); the thread pushes it to hardware every tick and
mirrors device readback into `get_data()`. Lifecycle: `start(path)`→record ticks→`stop()`→dump
`.npy`; `end()` joins the thread and disconnects. Units and DOF differ per device (xArm radians/pose,
Allegro radians, Inspire 0–1000 counts) — never cross them.

| Your task | Read |
|-----------|------|
| **Use** a device: `get_arm`/`get_hand`, `move`, record, fault recovery | [`usage.md`](usage.md) |
| **Edit** a control loop / fault handling / logging / lifecycle | [`internals.md`](internals.md) |
| Compute `qpos`/poses to feed `move()` (FK/IK/planning) | [`agent_docs/robot/`](../robot/README.md) |

Rule of thumb: **calling** these controllers → `usage.md`; **editing** them → `internals.md`.
`internals.md` also owns the **four-guarantee fault-handling contract** — read it before touching any loop.

## File map
| File | What it is |
|------|-----------|
| `__init__.py` | The public API: `get_arm(name)` and `get_hand(name, tactile, ip)` factories. Resolves device params from `network_info` (`system/current/network.json`). |
| `xarm_controller.py` | `XArmController` — UFactory xArm arm over the `xarm` SDK (`XArmAPI`). Joint **or** cartesian, servo-streaming **or** blocking position moves. Latching fault handling. |
| `allegro_controller.py` | `AllegroController` — Allegro 16-DOF hand as a **ROS2 node** (`rclpy`). Subscribes `/joint_states`, publishes `/allegro_hand_position_controller/commands`. |
| `inspire_controller_ip.py` | `InspireControllerIP` — Inspire 6-DOF hand over **Modbus TCP** (`pymodbus`). The default hand path (`ip=True`). Optional tactile. |
| `inspire_controller.py` | `InspireController` — same Inspire hand over **USB serial** (`pyserial`), custom byte framing. Legacy path (`ip=False`). |
| `gui_controller.py`, `gui_controller_simple.py` | Tkinter `RobotGUIController` — a **consumer** of the controllers above (waypoint teleop UI), not part of the arm/hand API. |
| `deprecated/`, `under_test/` | Old GUI variants and WIP tactile/CLI experiments. **Not** imported by the factories — treat as scratch. |

All paths relative to [`paradex/io/robot_controller/`](../../paradex/io/robot_controller/).
Franka (mentioned in the top-level `CLAUDE.md`) is **not implemented** — its branch is commented out
in `get_arm`. Related: `CaptureSession` ([`agent_docs/dataset_acquisition/`](../dataset_acquisition/README.md))
is the primary consumer; the camera half is [`agent_docs/camera_system/`](../camera_system/README.md).
