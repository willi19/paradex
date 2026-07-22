# Franka FR3 Control

How Paradex controls the Franka Research 3 (FR3) arm. Unlike the XArm (controlled
directly through a Python SDK), the Franka is driven by a **C++ daemon** that owns the
real-time control loop and talks to the robot over libfranka/FCI. Python communicates
with that daemon over the network. This document covers both the Python side and the
daemon/Docker side (the daemon C++ source is now vendored here — with a caveat, see
[Daemon & Docker](#daemon--docker)).

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
- **`src/validate/robot_controller/validate_franka.py`** — ⚠️ **this moves the robot.**
  Not a passive state check: after `ping` and an `Enter` prompt it oscillates joint 1 by
  ±0.3 rad (~±17°) through every control mode (joint/cartesian position & velocity, torque,
  impedance, collision, load, gripper). Clear the workspace and supervise.
  Records `.npy` per mode. `python validate_franka.py --host 127.0.0.1 [--mode joint_position]`.
  For a no-motion check use `--mock` (serialization only, no daemon needed), or connect and
  hit Ctrl+C at the prompt — `ping` has already run by then.

### GUI
[`gui_controller.py`](../paradex/io/robot_controller/gui_controller.py)
auto-detects arm DOF from `get_data()` and, when the controller exposes
`open_gripper`/`grasp`/`set_guiding_mode` (i.e. a Franka), swaps the 16-DOF hand panel
for Gripper (Open/Grasp) + Guide-Mode buttons. `hand_controller` is now optional.

## Daemon & Docker

The `franka_daemon` C++ source lives in this repo at [`cpp/franka_daemon/`](../cpp/franka_daemon/):

```
cpp/franka_daemon/
    franka_daemon.cpp      # ~60KB, the daemon
    servo_protocol.hpp     # msgpack schema + ZMQ endpoint helpers
    CMakeLists.txt
    build_daemon.sh        # run INSIDE the container
```

The built binary is **not** committed (gitignored) — it must come from inside the
container, which is the only place libfranka exists.

> 🚨 **The vendored source is incomplete — it does not build.** `franka_daemon.cpp:40-41`
> includes `../modules/franka_servo.hpp` and `../modules/robot_params.hpp`, but `cpp/modules/`
> was never vendored (it is absent from this repo **and** from the `franka_daemon.tar.gz`
> transfer archive). A source build stops at:
> ```
> franka_daemon.cpp:40:10: fatal error: ../modules/franka_servo.hpp: No such file or directory
> ```
> **Workaround in use:** the prebuilt binary from `franka_port/franka_daemon.tar.gz`
> (`build/franka_daemon`) is copied into `cpp/franka_daemon/build/`. Its sources are
> byte-identical to this repo's (verified by `diff`), and its `NEEDED` libs
> (`libfranka.so.0.18`, `libzmq.so.5`) match the image, so it runs as-is.
> **To fix properly:** copy `cpp_sources/modules/` from the original robot-control PC into
> `cpp/modules/`. Until then, any daemon source change cannot be compiled.

### Provenance

Vendored on 2026-07-16 from the robot-control PC at
`.../robothome/Sookwan/exp5-bimanual-paradex2/src_main/cpp_sources/daemon/`, which was
**not** a git repo — until then the daemon was unversioned with no backup, while its
Python client lived here. That path also holds a prebuilt `build/franka_daemon`.

> ⚠️ Two similarly-named trees exist on that PC. The lowercase `sookwan/...` tree (the
> `/workspace` mount) has only a **stale `build/CMakeFiles`** — no source, no binary.
> The authoritative copy was the `robothome/Sookwan` (capital S) path above. The script
> docstrings' `/workspace/src_main/.../franka_daemon` path points at the stale tree and
> does not exist.

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

`build_daemon.sh` must run **inside the container** (it needs libfranka at `/opt/libfranka`).
⚠️ On this PC it currently **fails** — see the missing-`cpp/modules/` warning above. Use the
prebuilt binary until those headers are vendored.

### Run — use the launcher

[`cpp/franka_daemon/run_daemon.sh`](../cpp/franka_daemon/run_daemon.sh) wraps the whole
container invocation (added 2026-07-16):

```bash
cd ~/paradex
./cpp/franka_daemon/run_daemon.sh                      # defaults: 172.17.1.11, 5555/5556
./cpp/franka_daemon/run_daemon.sh <FCI_IP> <cmd> <state>
```

It runs the image with `--network host` (so :5555/:5556 land on the host stack),
`--cap-add=SYS_NICE --ulimit rtprio=99 --ulimit memlock=-1` (needed for the RT control
loop), mounts the repo at `/workspace/paradex` (the image's bootstrap expects that path
and `pip install -e`'s it), and runs `build/franka_daemon`, building it first if absent.

Underlying binary usage:
```
Usage: ./franka_daemon <robot-ip> [--command_port PORT] [--state_port PORT]
```

> The image ships libfranka at `/opt/libfranka` but **no** `franka_daemon` executable —
> verified by `which franka_daemon` / `find / -name franka_daemon` inside the container.
> The binary must always come from the repo's `build/`.

> ⚠️ The container writes to the bind mount as **root**, so `cpp/franka_daemon/build/`
> ends up root-owned. A failed build leaves a root-owned dir the host user cannot clean —
> use `sudo rm -rf cpp/franka_daemon/build` before retrying.

## Using the daemon from other PCs

**This already works — no configuration needed.** The daemon binds with
`tcp_endpoint()` in `servo_protocol.hpp`:

```cpp
inline std::string tcp_endpoint(int port) { return "tcp://*:" + std::to_string(port); }
```

`tcp://*` = `0.0.0.0` (all interfaces), and the container uses **host networking**, so
:5555/:5556 are reachable from any PC that can route to the daemon host. That is exactly
why `network_info["franka"]` can be a routable IP rather than `localhost`. From another PC:

```bash
python validate_franka.py --host <daemon-host-ip>
python franka_teaching.py --host <daemon-host-ip> --save_path <dir>
```

> On the `robot` PC the daemon runs locally, so `network_info["franka"]` is **`127.0.0.1`**
> (`system/current/network.json`). It previously held `172.16.0.2`, a stale address from the
> original PC that does not route from here. Set it to this host's LAN IP (`192.168.0.2`)
> only if another PC needs to command the arm.

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
robot's FCI IP (`sudo ufw allow from <FCI_IP>`), plus a `PREEMPT_RT` kernel — see below.

## RT kernel + NVIDIA GPU on the same boot

libfranka's 1 kHz loop wants a `PREEMPT_RT` kernel, but the NVIDIA driver **refuses to
build against one**. Getting both on one boot needs a documented workaround. Setup on the
`robot` PC (paradex2), done 2026-07-16.

### Installing the RT kernel

```bash
sudo pro attach <ubuntu-pro-token>     # free personal token: https://ubuntu.com/pro/dashboard
sudo pro enable realtime-kernel        # installs 5.15.0-*-realtime; disables Livepatch (expected)
```
The old kernel is **not** removed — both stay installed and selectable in GRUB.

### The NVIDIA conflict (and the fix)

`sudo dkms autoinstall -k 5.15.0-1032-realtime` fails with:
```
The NVIDIA driver does not support real-time kernels.
*** Failed PREEMPT_RT sanity check. Bailing out! ***
```
This is a **deliberate gate in NVIDIA's Kbuild**, not a compile error — bypassing it works.
Booting RT without doing so leaves the GPU unbound (`nvidia-smi` fails, desktop falls back
to software rendering and looks visibly broken).

Add `IGNORE_PREEMPT_RT_PRESENCE=1` to the DKMS `MAKE[0]` line, then build:

```bash
sudo cp /usr/src/nvidia-580.159.03/dkms.conf /usr/src/nvidia-580.159.03/dkms.conf.bak
sudo sed -i 's/IGNORE_XEN_PRESENCE=1 IGNORE_CC_MISMATCH=1/& IGNORE_PREEMPT_RT_PRESENCE=1/' \
    /usr/src/nvidia-580.159.03/dkms.conf
grep -n IGNORE_PREEMPT_RT_PRESENCE /usr/src/nvidia-580.159.03/dkms.conf   # must print the MAKE line

sudo dkms build   -m nvidia -v 580.159.03 -k 5.15.0-1032-realtime --force
sudo dkms install -m nvidia -v 580.159.03 -k 5.15.0-1032-realtime --force
sudo modprobe nvidia && nvidia-smi
```

Verified working on this PC: RTX 3090, driver 580.159.03, `torch 2.8.0+cu128`
`cuda.is_available() == True`, with `/sys/kernel/realtime == 1`. `dkms status` shows the
module built for **both** the realtime and generic kernels.

> ⚠️ **A driver package update silently reverts this.** Reinstalling `nvidia-dkms-580`
> replaces `dkms.conf`, dropping the flag — the next RT boot then comes up with no GPU.
> Either re-apply the `sed` (a `.bak` is kept alongside) or `sudo apt-mark hold nvidia-dkms-580`.

> The original robot-control PC ran driver **550**, which had no such gate — hence
> `franka_port/README.md`'s claim that "nvidia DKMS rebuilds automatically" and "GPU works
> fine on the RT kernel". That is **true for 550 only**; 580 needs the flag above.

### Switching kernels

`GRUB_DEFAULT=saved` with a hidden menu, so select by name rather than at the boot screen:

```bash
# boot RT once — reverts to the saved default on the next reboot (safest for a first test)
sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 5.15.0-1032-realtime"
sudo reboot

# make a choice permanent
sudo grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 5.15.0-1032-realtime"
sudo grub-set-default "Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-124-generic"
```
Confirm with `uname -r`, `cat /sys/kernel/realtime` (→ `1`), `nvidia-smi`.

Without an RT kernel, libfranka can still run via `RealtimeConfig::kIgnore` at reduced
speed — acceptable for smoke tests, not for real control.
