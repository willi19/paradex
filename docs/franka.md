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

## Daemon & Docker (lives outside this repo, on the robot-control PC)

The `franka_daemon` C++ source is **not** in this repo. As of 2026-07-16 it lives on the
robot-control PC at:

```
/home/exp_main/robothome/Sookwan/exp5-bimanual-paradex2/src_main/cpp_sources/daemon/
    franka_daemon.cpp      # ~60KB, the daemon
    servo_protocol.hpp     # msgpack schema + ZMQ endpoint helpers
    CMakeLists.txt
    build_daemon.sh        # run INSIDE the container
    build/franka_daemon    # built binary
```

> ⚠️ Two similarly-named trees exist. `/home/exp_main/sookwan/exp5-bimanual-paradex2`
> (lowercase, the `/workspace` mount) has only a **stale `build/CMakeFiles`** — no source,
> no binary. The authoritative copy is the `robothome/Sookwan` (capital S) path above.
> Neither tree is a git repo, so **this source is unversioned and has no backup.**

### Docker

| | |
|---|---|
| Image | `franka3/real:bimanual_control_libfranka0.18.0_v2` (~25.8 GB) |
| Container | `franka_seoja` (also `franka_jangja` for the second arm) |
| Network | **`host`** — ports are on the host stack directly, no port mapping |
| Mounts | `/home/exp_main/sookwan/exp5-bimanual-paradex2` → `/workspace`<br>`/home/exp_main/robothome` → `/mnt/robothome` |
| libfranka | **0.18.0**, installed at `/opt/libfranka` inside the image |

The `docker_robot/docker-compose.yaml` referenced by the container labels
(`.../exp5-bimanual-paradex2/docker_robot`) **no longer exists on disk** — the compose
file is gone; only the built image and stopped containers remain. Start the existing
container directly (`docker start -ai franka_seoja`) rather than via compose.

### Build

`build_daemon.sh` must run **inside the container** (it needs libfranka at `/opt/libfranka`):

```bash
docker start -ai franka_seoja        # or docker exec -it franka_seoja bash
cd /mnt/robothome/Sookwan/exp5-bimanual-paradex2/src_main/cpp_sources/daemon
./build_daemon.sh                    # cmake -DCMAKE_PREFIX_PATH=/opt/libfranka && make -j
```

### Run

```
Usage: ./franka_daemon <robot-ip> [--command_port PORT] [--state_port PORT]
```

Note the binary path in the script docstrings
(`/workspace/src_main/cpp_sources/daemon/build/franka_daemon`) points at the **stale**
lowercase tree and will not exist. Use the `/mnt/robothome` path:

```bash
/mnt/robothome/Sookwan/exp5-bimanual-paradex2/src_main/cpp_sources/daemon/build/franka_daemon \
    <robot-fci-ip> --command_port 5555 --state_port 5556
```

## Using the daemon from other PCs

**This already works — no configuration needed.** The daemon binds with
`tcp_endpoint()` in `servo_protocol.hpp`:

```cpp
inline std::string tcp_endpoint(int port) { return "tcp://*:" + std::to_string(port); }
```

`tcp://*` = `0.0.0.0` (all interfaces), and the container uses **host networking**, so
:5555/:5556 are reachable from any PC that can route to the daemon host. That is exactly
why `network_info["franka"]` is `172.16.0.2` and not `localhost`. From another PC:

```bash
python validate_franka.py --host 172.16.0.2
python franka_teaching.py --host 172.16.0.2 --save_path <dir>
```

### The real constraint: no client arbitration

- **State (PUB/SUB :5556)** — safe for any number of subscribers. ZMQ PUB broadcasts to
  all SUBs, so N PCs can monitor state concurrently with no side effects.
- **Commands (REQ/REP :5555)** — ⚠️ **the daemon has no ownership/lease/token concept.**
  (`g_robot_mutex` only guards the robot object for internal thread-safety; it does not
  arbitrate between clients.) ZMQ REP fair-queues multiple REQ clients, so two PCs *can*
  both send commands — and their `move` commands will interleave on a real robot arm.
  Also, `move_to_qpos` blocks, so a second client's request queues behind the first and
  may hit the client's 30 s `RCVTIMEO`.

So: multi-PC **monitoring** is free today. Multi-PC **commanding** needs either human
coordination ("one driver at a time") or a lease/token added to the daemon (C++ change).
For a point-to-point alternative that avoids exposing the ports, an SSH tunnel works
without touching the daemon: `ssh -L 5555:localhost:5555 -L 5556:localhost:5556 <host>`.

### Host requirements
Robot unlocked with FCI enabled in Desk, and a network route from the daemon host to the
robot's FCI IP. (`PREEMPT_RT` kernel requirement for libfranka 0.18.0 not verified here.)
