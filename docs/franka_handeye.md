# Franka FR3 Hand-Eye Calibration

Teaching → replay → capture → solve, for the Franka FR3 arm (no gripper attached).
Everything below was verified on this setup; the gotchas section is the hard-won part.

---

## 0. Setup (one-time)

| Item | Value |
|---|---|
| Robot IP | `172.16.1.11` (Desk: <https://172.16.1.11/desk/>) — also in `system/current/network.json` |
| conda env | `cuda12.1_torch2.3` (same for every step) |
| ROS 2 | Humble + `~/franka_ros2_ws` |
| URDF | `rsc/robot/franka.urdf` (7-DoF arm, flange `fr3_link8`) |

**Shell aliases** (add to `~/.bashrc`):

```bash
alias ros-setting='source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash'
alias calc-env='PYTHONPATH= python'      # for calculate.py (must NOT see ROS)
```

**Required host config** (once):

```bash
# 1) firewall must let the robot's FCI UDP through, else "UDP receive: Timeout"
sudo ufw allow from 172.16.1.11

# 2) realtime priority for the 1 kHz control loop
sudo groupadd realtime && sudo usermod -aG realtime $USER
sudo cp ~/franka_ros2_ws/src/limits.conf /etc/security/limits.d/99-realtime.conf
# log out / reboot, then: ulimit -r   -> 99

# 3) CPU governor
sudo cpupower frequency-set -g performance
```

`fr3_ros_controllers.yaml` (in `franka_fr3_moveit_config`) also needs
`thread_priority: 98` under `controller_manager.ros__parameters` — already added.

**Before every session, in Desk:**
- Activate FCI: sidebar menu (≡) → **Activate FCI** (it is a menu item, not a popup). Keep Desk open.
- Unlock joints.
- **End-effector load must match reality.** With no gripper, set mass = 0. A stale
  `0.833 kg` (Franka Hand) makes gravity compensation shove the arm upward.

---

## 1. Launch the robot stack

ROS terminal (`ros-setting` first):

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
    robot_ip:=172.16.1.11 use_fake_hardware:=false load_gripper:=false
```

- `use_fake_hardware:=false` → real robot (`true` is a simulated demo).
- `load_gripper:=false` → no gripper attached.

Sanity check:

```bash
ros2 control list_controllers          # fr3_arm_controller  ... active
ros2 topic echo --once /joint_states   # real joint values
```

---

## 2. Teaching — collect calibration poses

ROS terminal:

```bash
python src/capture/robot/franka_teaching.py
```

What it does automatically:
1. Loads + activates `gravity_compensation_example_controller` (arm goes compliant —
   **3-second countdown, hold the arm**), deactivating `fr3_arm_controller`.
2. `c` = capture the current pose, `q` = finish.
3. On exit (also on Ctrl+C / error) it **always restores `fr3_arm_controller`**.
4. Detects a reflex (joint values frozen) and auto-recovers.
5. **Versioning**: any existing poses are moved to
   `system/current/hecalib/franka_versions/<timestamp>/` before a new session starts,
   so a previous set is never overwritten.

Output → `system/current/hecalib/franka/`:
- `{idx}_qpos.npy` — (7,) joints, ordered `fr3_joint1..7`
- `{idx}_aa.npy` — (4,4) `fr3_link0` → `fr3_link8` pose

Options: `--save_path`, `--manual-freedrive` (skip the controller swap; you enable
free-drive yourself).

> Guide the arm **gently** and stay in the **mid joint range**. Hitting a hard joint
> limit triggers a reflex; recovering from that needs the Desk joint-recovery screen
> (which requires the **X4 external enabling device**).

---

## 3. Replay check — motion only, no cameras

Verify the arm moves through the collected poses before wiring up cameras.

```bash
python src/capture/robot/franka_replay_check.py --step --home
```

- `--step` — wait for Enter before each move (**use this the first time**)
- `--home` — go to the FR3 home pose first
- `--step_size 0.15 --step_time 0.5` — speed (rad per waypoint / s per waypoint)
- `--min_steps 3 --max_steps 40` — clamp on the waypoint count

Move to home only:

```bash
python src/capture/robot/franka_home.py
```

FR3 home = `[0, -π/4, 0, -3π/4, 0, π/2, π/4]` (`FRANKA_HOME_QPOS`).

---

## 4. Capture — replay + multi-camera images

ROS terminal, with the capture PCs up:

```bash
python src/calibration/handeye/capture.py --arm franka
```

For each pose: move → capture images on all cameras → save `eef.npy`, `qpos.npy`.
Output: `<shared_dir>/handeye_calibration/<timestamp>/<idx>/`.

`get_data()` stores the **actual** robot pose, so even if a move stops short the
`(image, pose)` pair stays consistent and the calibration remains valid.

---

## 5. Solve — **without ROS**

```bash
calc-env src/calibration/handeye/calculate.py --arm franka
#  == PYTHONPATH= python src/calibration/handeye/calculate.py --arm franka
```

ROS's `pinocchio` is built against NumPy 1.x and segfaults under this env's NumPy 2.
Clearing `PYTHONPATH` makes conda's `pinocchio 3.8.0` load instead.

Pipeline: charuco detect → FK (`fr3_link8`) → `AX=XB` (Tsai-Lenz) → nonlinear refine →
`C2R.npy`, plus a debug overlay.

---

## Files

| Path | Role |
|---|---|
| `paradex/io/robot_controller/franka_controller.py` | `FrankaController` (ROS 2), `FRANKA_HOME_QPOS` |
| `src/capture/robot/franka_teaching.py` | teaching + free-drive + versioning |
| `src/capture/robot/franka_replay_check.py` | motion dry-run (no cameras) |
| `src/capture/robot/franka_home.py` | move to home |
| `src/calibration/handeye/capture.py` | `--arm franka` branch |
| `src/calibration/handeye/calculate.py` | `EEF_LINK = {"franka": "fr3_link8"}` |
| `rsc/robot/franka.urdf` | 7-DoF arm (NOT `fr3_inspire`, which is 19-DoF arm+hand) |

### How `FrankaController` works

It never talks to libfranka directly. It is an rclpy node that:
- subscribes `/joint_states` (by joint **name**) → `get_qpos()`
- reads TF `fr3_link0`→`fr3_link8` → `get_eef()`
- sends goals to `/fr3_arm_controller/follow_joint_trajectory` → `move()`

`move()` splits the motion into waypoints (count ∝ distance, clamped) so joint speed
stays bounded, verifies the arm **actually reached** the target (JTC reports "success"
on elapsed time even when a reflex killed the motion), and auto-recovers + retries.

On construction it runs `_ensure_arm_ready()`: if `fr3_joint*/effort` is not
`[available] [claimed]`, it re-activates the hardware component and cycles
`fr3_arm_controller`. No manual `ros2 control` commands needed.

### What "effort" means

`fr3_arm_controller` uses `command_interfaces: [effort]` — it commands joint **torque**.
The trajectory controller computes `tau = Kp·(q_target − q) + Kd·(−q̇)`; libfranka adds
gravity compensation on top. Teaching's gravity-comp controller is also effort-based,
it just commands `tau = 0` (so the arm floats and you can push it).

The effort interface must be **`[available]`** (hardware ready to take torque) and
**`[claimed]`** (a controller owns it) or nothing moves.

---

## Gotchas (all hit during bring-up)

| Symptom | Cause / Fix |
|---|---|
| `libfranka: UDP receive: Timeout` | **ufw blocked the robot's UDP.** `sudo ufw allow from 172.16.1.11`. (Not the RT kernel — tcpdump showed packets arriving at the NIC but dropped before the socket.) |
| `Connection to FCI refused` | FCI not activated: Desk → sidebar menu → **Activate FCI**. |
| `communication_constraints_violation` | 1 kHz control-loop jitter. `thread_priority: 98`, `rtprio 99`, `governor performance` all help; a **PREEMPT_RT kernel** is the real fix (this host runs a generic kernel). |
| Controller says `Goal reached, success` but the arm never moved | `fr3_joint*/effort` is `[unavailable]` after a reflex. `FrankaController` auto-recovers now; manually: `ros2 control set_hardware_component_state FrankaHardwareInterface active`, then deactivate+activate `fr3_arm_controller`. |
| Arm jerks **upward** when gravity comp turns on | Desk end-effector mass (`0.833 kg`) doesn't match reality (no gripper). Set it to 0 in Desk. |
| Arm locks at a joint limit and won't recover | Reflex. In gravity-comp (torque) mode joint limits are **hard** reflexes, not soft walls — approaching gently does not help. Desk shows a joint-recovery screen that needs the **X4 external enabling device**. Escape a singular/limit pose with a **joint-space** move (`franka_home.py`), which ignores singularities. |
| `calculate.py` → `_ARRAY_API not found` / segfault | ROS's pinocchio (NumPy 1.x) is on `PYTHONPATH`. Run with `calc-env` (`PYTHONPATH=`). |
| `franka_fr3_moveit_config not found` | Forgot `ros-setting` (workspace not sourced). |

### Environment split (important)

| Step | conda env | ROS sourced? |
|---|---|---|
| teaching / replay / capture | `cuda12.1_torch2.3` | **yes** (`ros-setting`) — needs rclpy |
| calculate | `cuda12.1_torch2.3` | **no** (`calc-env`) — needs conda pinocchio |
