# Franka FR3 Control

How Paradex controls the Franka Research 3 (FR3) arm. Unlike the XArm (controlled
directly through a Python SDK), the Franka is driven by a **C++ daemon** that owns the
real-time control loop and talks to the robot over libfranka/FCI. Python communicates
with that daemon over the network. This document covers the Python side (which lives in
this repo) and the daemon/Docker side (which does **not** — see the placeholders below).

## Architecture

```
┌─────────────────────────┐        ZMQ + msgpack         ┌──────────────────────────┐        libfranka / FCI        ┌───────────┐
│  Python (this repo)      │  ───────────────────────▶   │  franka_daemon (C++)     │  ─────────────────────────▶   │  FR3 arm  │
│  FrankaController        │    REQ→REP  :5555 (command)  │  runs inside Docker      │      1 kHz real-time loop      │  + gripper│
│                          │    SUB←PUB  :5556 (state)    │  on the robot-control PC │                                └───────────┘
└─────────────────────────┘   ◀───────────────────────   └──────────────────────────┘
```

- **Python client** ([`paradex/io/robot_controller/franka_controller.py`](../paradex/io/robot_controller/franka_controller.py)):
  a thin, stateless-ish client. It sends commands and receives a streamed state feed. It
  does **not** run any real-time loop itself — the daemon does. The class API mirrors
  `XArmController` (`move`, `get_data`, `start`/`stop`/`end`, `reset`, `is_error`) so it
  can be dropped into the same call sites, plus Franka-only extras (gripper, impedance,
  guiding mode, etc.).
- **C++ daemon** (`franka_daemon`): the only process allowed to hold the real-time FCI
  connection to the robot. Must already be running before any Python script connects.
  Runs inside Docker (see below).

Why the split: libfranka requires a 1 kHz real-time control loop with a `PREEMPT_RT`
kernel. Isolating that in a dedicated C++/Docker process keeps the Python research code
free of real-time constraints and lets it connect/disconnect freely.

## Network / ports

| What | Value | Source |
|------|-------|--------|
| Daemon host (from Python) | `network_info["franka"]` = `172.16.0.2` | `system/current/network.json` |
| Command socket (REQ/REP) | `5555` | `FrankaController(command_port=...)` |
| State socket (PUB/SUB)   | `5556` | `FrankaController(state_port=...)` |
| Robot FCI IP (daemon → robot) | e.g. `172.16.1.11` | passed to `franka_daemon` on launch |

When the daemon runs on the **same PC** as the Python script, connect to `localhost`
(the default in `franka_teaching.py` / `validate_franka.py`). `network_info["franka"]`
is the daemon host; the FCI IP is what the *daemon* uses to reach the robot — they are
different addresses.

## Wire protocol (ZMQ + msgpack)

Every message is a msgpack-encoded dict with a `"type"` key.

### Commands (REQ → REP on :5555)
Reply is `{"type": "success", ...}` or `{"type": "error", "message": ...}`.

| `type` | Payload | Notes |
|--------|---------|-------|
| `ping` | — | health check |
| `get_state` | — | one-shot state (state also streams on :5556) |
| `move_to_qpos` | `qpos[7]`, `speed_scale` | joint-space move; **blocks** until done |
| `move_to_cartesian` | `position[3]`, `orientation[9]` (row-major R), `move_speed` | cartesian move; blocks |
| `set_joint_velocity` | `dq[7]`, `duration_ms` | `duration_ms=0` → **streaming** mode |
| `set_cartesian_velocity` | `twist[6]` (vx,vy,vz,wx,wy,wz, base frame), `duration_ms` | streaming when `0` |
| `set_torques` | `torques[7]`, `duration_ms` | direct torque; streaming when `0` |
| `set_joint_impedance` | `K_theta[7]` | joint stiffness Nm/rad |
| `set_cartesian_impedance` | `K_x[6]` | Kx,Ky,Kz,Kroll,Kpitch,Kyaw |
| `set_collision_behavior` | `torque_lower[7]`, `torque_upper[7]`, `force_lower[6]`, `force_upper[6]` | |
| `set_load` | `mass`, `F_x_Cload[3]`, `load_inertia[9]` | payload registration |
| `set_ee` | `NE_T_EE[16]` (**column-major**) | flange→EE transform; `eye(4)` = none |
| `set_guiding_mode` | `guiding_axes[6]` (bools), `nullspace` | hand-guiding |
| `error_recovery` | — | clears reflex/error state |
| `open_gripper` | `width`, `speed` | |
| `grasp` | `force`, `speed`, `eps_inner`, `eps_outer` | |
| `stop_streaming` | — | end a velocity/torque streaming loop (ramps to zero) |
| `stop` | — | emergency stop |

### State feed (PUB → SUB on :5556)
Broadcast continuously as `{"type": "state_update", ...}`:

`timestamp`, `qpos[7]`, `qvel[7]`, `tau_ext[7]`, `wrench[6]`, `O_T_EE[16]`
(**column-major** 4×4), `gripper_width`, `gripper_grasping`.

**Matrix conventions:** `O_T_EE` and `NE_T_EE` are flattened **column-major** (libfranka
convention) — the client reshapes with `order="F"`. The `orientation` in
`move_to_cartesian` is a **row-major** 9-element rotation matrix. Don't mix them up.

## Python usage

The controller is intentionally not wired into `get_arm()` yet — the stub in
[`__init__.py`](../paradex/io/robot_controller/__init__.py) is commented out (and, as
written, calls `FrankaController()` with no args, which would fail since `ip` is
required). Instantiate it directly for now:

```python
from paradex.io.robot_controller.franka_controller import FrankaController

fc = FrankaController("localhost")          # or network_info["franka"]
assert fc.ping()

data = fc.get_data()                         # {qpos, qvel, position(4x4), wrench, tau_ext, gripper_width, ...}
fc.move(target_qpos, speed_scale=0.15)       # (7,) joint move  ... or (4,4) cartesian move
fc.open_gripper(); fc.grasp(force=20.0)

fc.start("out/session_001")                  # begin recording state to .npy
# ... move around ...
fc.stop()                                    # writes time/position/velocity/torque/wrench/O_T_EE/gripper_width .npy
fc.end()                                     # close sockets
```

To wire it into the factory later, uncomment the stub and pass the IP like the other arms:
```python
if arm_name == "franka":
    from .franka_controller import FrankaController
    return FrankaController(network_info["franka"])
```

### Scripts
- **`src/capture/robot/franka_teaching.py`** — hand-guide the robot and save keyposes.
  Enables guiding mode; press `c` to save current qpos + EE pose, `q` to quit.
  `python franka_teaching.py --save_path <dir> [--host localhost]`
- **`src/validate/robot_controller/validate_franka.py`** — per-mode oscillation tests
  (joint/cartesian position & velocity, torque, impedance, collision, load, gripper).
  Records `.npy` per mode. `python validate_franka.py --host localhost [--mode joint_position]`.
  `--mock` runs serialization tests with no daemon.

### GUI
[`gui_controller.py`](../paradex/io/robot_controller/gui_controller.py)
auto-detects arm DOF from `get_data()` and, when the controller exposes
`open_gripper`/`grasp`/`set_guiding_mode` (i.e. a Franka), swaps the 16-DOF hand panel
for Gripper (Open/Grasp) + Guide-Mode buttons. `hand_controller` is now optional.

## Daemon & Docker (source NOT in this repo)

> ⚠️ **TODO — fill in.** The `franka_daemon` C++ source, its `Dockerfile`, and
> `docker-compose.yaml` live outside this repository (on the robot-control PC). The
> details below are reconstructed from script docstrings and are **placeholders** —
> replace with the authoritative locations/steps.

- **Where the daemon source lives:** `TODO` — repo/path of the C++ daemon and Docker files.
  Docstrings reference a compose file at `docker_robot/docker-compose.yaml`, a service
  named `franka_seoja`, and the built binary at
  `/workspace/src_main/cpp_sources/daemon/build/franka_daemon` inside the container.
- **How libfranka / the daemon is built & installed:** `TODO` — libfranka version, FCI
  firmware version, build steps (`cmake`/`make`), and any `PREEMPT_RT` kernel requirement.
- **How to launch the daemon (as documented in `validate_franka.py`):**
  ```bash
  # 1. Start the container
  docker compose -f docker_robot/docker-compose.yaml up -d
  docker compose -f docker_robot/docker-compose.yaml exec franka_seoja bash

  # 2. Inside the container, run the daemon against the robot's FCI IP
  /workspace/src_main/cpp_sources/daemon/build/franka_daemon 172.16.1.11 \
      --command_port 5555 --state_port 5556
  ```
- **Host requirements:** `TODO` — realtime kernel, network route to the robot's FCI IP,
  robot unlocked / FCI enabled in Desk, Docker host networking (so :5555/:5556 are
  reachable from Python).
