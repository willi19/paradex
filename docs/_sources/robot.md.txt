# Robot Control

Overview of Paradex's robot subsystem — the arms, the hands, and the kinematics
layer that ties them to the vision rig. Read this to build the mental model; for
method signatures, parameters, and return values see the
{doc}`API reference <robot_api>`.

- Generated per-symbol API: {doc}`API Reference <autoapi/index>`
- Sibling subsystem: {doc}`Camera System <camera_system>`

---

## 1. Architecture

Robot control splits into two independent halves that meet only through the URDF
kinematic model:

- **Hardware controllers** (`paradex/io/robot_controller/`) — one background
  thread per device streaming commands to real hardware over its native
  transport (XArm SDK, ROS2, Modbus, serial).
- **Kinematics** (`paradex/robot/`) — pure computation on a URDF: forward
  kinematics, inverse kinematics, and GPU motion planning. No hardware I/O.

Every device is reached through the `get_arm` / `get_hand` factory, so callers
never import a concrete controller class. Each controller owns its own control
loop thread and exposes the same lifecycle verbs — `move` / `start` / `stop` /
`end` — regardless of the transport underneath.

```{mermaid}
flowchart TB
    subgraph Factory["paradex.io.robot_controller"]
      GA["get_arm(name)"]
      GH["get_hand(name, ip, tactile)"]
    end
    subgraph Ctrl["Hardware controllers (1 thread each)"]
      XA["XArmController<br/>XArm SDK / TCP"]
      AL["AllegroController<br/>ROS2 rclpy Node"]
      IP["InspireControllerIP<br/>Modbus TCP"]
      IU["InspireController<br/>USB serial"]
    end
    subgraph Kin["paradex.robot (no hardware I/O)"]
      RW["RobotWrapper<br/>Pinocchio FK / IK"]
      CU["CuroboPlanner<br/>GPU motion planning"]
      UR["urdf / utils<br/>URDF build + paths"]
    end
    GA --> XA
    GH --> AL
    GH --> IP
    GH --> IU
    XA -. "shares URDF model" .-> RW
    RW --> CU
    UR --> RW
```

Method-level details for each component are in the {doc}`API reference <robot_api>`.

---

## 2. Core Concepts

| Term | Meaning |
|------|---------|
| **Arm** | 6-DoF manipulator. **XArm** (UFactory SDK over TCP) is live; **Franka** is stubbed in the factory. |
| **Hand** | Multi-finger end-effector. **Allegro** (16 DoF, ROS2) or **Inspire** (6 DoF, Modbus TCP or USB). |
| **Controller** | A class wrapping one device: a background loop thread + thread-safe `move`/`get_data`. |
| **Action** | The command written to a device. Arm: `(6,)` joint radians *or* `(4,4)` homogeneous pose. Allegro: `(16,)` radians. Inspire: `(6,)` motor units 0–1000. |
| **`qpos`** | Measured joint state read back from the device (units match that device's action). |
| **Servo vs. position** | XArm streaming servo (`is_servo=True`, non-blocking) vs. one-shot position move (`is_servo=False`, blocks until reached). |
| **RobotWrapper** | Pinocchio model built from a URDF — the FK/IK backend, decoupled from hardware. |
| **CuroboPlanner** | CuRobo GPU planner: collision-aware trajectories to a pose goal-set or joint target. |

---

## 3. Controller Lifecycle

Every controller follows the same shape: construction connects to the device and
spawns a ~100 Hz loop that continuously writes the latest `action` to hardware.
`move()` only updates that shared target (cheap, thread-safe); the loop does the
talking. `start()` / `stop()` bracket an episode of logging; `end()` tears down.

```{mermaid}
sequenceDiagram
    participant C as Caller
    participant L as Controller loop (~100 Hz)
    participant HW as Device
    C->>L: get_arm / get_hand (connect + spawn loop)
    L->>HW: continuous write(latest action)
    C->>L: move(action)
    Note over L: swap shared target under lock
    C->>L: start(save_path)
    loop every tick while recording
      L->>HW: write action
      HW-->>L: read state → append to log
    end
    C->>L: stop()
    Note over L: flush log to *.npy
    C->>L: end()
    Note over L: join thread, disconnect
```

The loop reads the target under a `Lock` each tick, so `move()` from the caller
thread and the loop's writes never race. When `start()` has armed the save event,
every tick also reads back device state and appends it to per-key buffers that
`stop()` dumps as `time.npy`, `position.npy`, `action.npy`, etc.

---

## 4. Arms — XArm

`XArmController` wraps the UFactory `XArmAPI`. Its loop runs at 100 Hz and has two
modes selected per-`move`:

- **Servo** (`is_servo=True`): streams `set_servo_angle_j` (joint action) or
  `set_servo_cartesian_aa` (pose action) — non-blocking, for smooth trajectories.
- **Position** (`is_servo=False`): switches to position mode, issues one blocking
  `set_servo_angle` / `set_position`, waits for arrival, then returns to servo
  mode. `move()` blocks on `position_control_event` until this completes.

An action is a `(6,)` joint vector (radians, clipped to ±2π) **or** a `(4,4)`
homogeneous pose; the loop dispatches on the array shape. `get_data()` returns
both representations: measured `qpos` and the current end-effector pose.

```{mermaid}
flowchart TB
    M["move(action, is_servo, speed)"] --> S{"action.shape"}
    S -->|"(6,)"| J["joint target<br/>clip ±2π"]
    S -->|"(4,4)"| P["pose target<br/>homo → aa / cart"]
    J --> D{"is_servo?"}
    P --> D
    D -->|"True"| SV["servo stream<br/>(non-blocking)"]
    D -->|"False"| PC["position move<br/>wait arrival"]
```

Error state is surfaced via `is_error()` (set when the arm reports err/warn);
`clear_error()` clears and re-enables servo mode without a reconnect.

---

## 5. Hands — Allegro (ROS2) and Inspire (Modbus)

The two hand families share the controller contract but differ entirely in
transport:

| | Allegro | Inspire (IP) | Inspire (USB) |
|--|---------|--------------|---------------|
| Class | `AllegroController` | `InspireControllerIP` | `InspireController` |
| DoF (action) | 16 | 6 | 6 |
| Transport | ROS2 topics (`rclpy`) | Modbus TCP (`pymodbus`) | serial (`pyserial`) |
| Command | publish `Float64MultiArray` on `/allegro_hand_position_controller/commands` | write `angleSet` registers | write `setangle` frames |
| State | subscribe `/joint_states` | read `angleAct` registers | read `getactangle` |
| Units | radians (clipped to `MAX_ANGLE=2.1`) | 0–1000 (0 = closed, 1000 = open) | 0–1000 |
| Tactile / force | — | `get_force`, `get_tactile` (17-pad layout) | `get_force` |

`AllegroController` **is** an `rclpy.node.Node`: a spin thread services ROS
callbacks while a separate control thread republishes the target at 100 Hz. It
blocks at startup until the first `/joint_states` message arrives, then mirrors
that pose as the initial action. Note `get_data()` reorders the raw joint vector
through `JS_TO_CMD` so `qpos` matches command order.

`InspireControllerIP` opens a Modbus TCP client, pre-sets speed/force/angle
registers, and runs a `move_hand` loop that writes `angleSet` and reads back
`angleAct` + `forceAct` (+ optional tactile grid when `tactile=True`). Extra
Inspire-only helpers: `home_robot`, `set_homepose`, and `calibrate_force`
(writes the force-calibration register and polls until the firmware clears it).

```{mermaid}
flowchart TB
    subgraph A["AllegroController (ROS2)"]
      AC["control loop 100 Hz"] -->|"publish cmd"| RT["/allegro_hand_position_controller/commands"]
      JS["/joint_states"] -->|"subscribe"| AC
    end
    subgraph I["InspireControllerIP (Modbus TCP)"]
      IC["move_hand loop 100 Hz"] -->|"write angleSet"| MB["Modbus registers"]
      MB -->|"read angleAct / forceAct"| IC
    end
```

---

## 6. Kinematics — RobotWrapper (Pinocchio)

`RobotWrapper` builds a Pinocchio model straight from a URDF and exposes FK, IK,
Jacobians, and model introspection (joint/link names, DoF, joint limits). It is
hardware-agnostic — the same class serves the arm, a hand, or a merged arm+hand
URDF (`get_robot_urdf_path` resolves the path).

- **Forward kinematics**: `compute_forward_kinematics(qpos, link_list)` returns a
  `{link_name: 4x4 homogeneous}` dict for the requested links.
- **Inverse kinematics**: `solve_ik(target_pose, end_effector_name, ...)` is a
  damped-least-squares Gauss–Newton loop with random restarts (`try_num`), each
  step clipped to joint limits; returns `(qpos, success)`.

```{mermaid}
flowchart TB
    U["URDF path"] --> RW["RobotWrapper(urdf_path)"]
    Q["qpos"] --> FK["compute_forward_kinematics<br/>(qpos, link_list)"]
    RW --> FK
    FK --> POSE["{link: 4x4 pose}"]
    T["target 4x4 pose"] --> IK["solve_ik<br/>(target, eef_name)"]
    RW --> IK
    IK --> RES["(qpos, success)"]
```

Typical grasp pipeline usage (from `src/inference/`): `solve_ik(wrist_6d,
"palm_link")` for the wrist, then `compute_forward_kinematics(q, ["link6"])` to
read the resulting arm flange pose.

---

## 7. Motion Planning — CuRobo

`CuroboPlanner` (`paradex/robot/curobo.py`) wraps CuRobo's `MotionGen` +
`IKSolver` for collision-aware, GPU-accelerated planning against a world of
cuboids and object meshes. It is a heavyweight object: construction loads the
robot config, builds the world, and **warms up** the planner (compiles CUDA
graphs), so build it once and reuse it.

| Entry point | Goal | Returns |
|-------------|------|---------|
| `plan_goalset(init_state, goal_pose)` | reach any of N candidate poses | `(goalset_index, trajectory)` |
| `plan_to_joint_target(init_state, goal_joint_state)` | reach a joint configuration | `(success, trajectory)` |
| `plan_full_step(...)` | batch-plan over many grasp candidates, pick best per place | `(success, tensor_ids)` |
| `update_world(obj_dict)` | refresh collision meshes between plans | — |

```{mermaid}
flowchart TB
    W["obstacle_dict + obj_dict<br/>(cuboids + meshes)"] --> P["CuroboPlanner(...)"]
    RC["robot_cfg + tensor_args"] --> P
    P --> WU["MotionGen.warmup<br/>(compile CUDA graphs)"]
    G["goal pose(s) / joint target"] --> PL["plan_goalset /<br/>plan_to_joint_target"]
    WU --> PL
    PL --> TR["interpolated joint trajectory"]
```

Pose goals are converted from `(4,4)` matrices to CuRobo `Pose` (position +
**wxyz** quaternion — note the `xyzw → wxyz` reorder, matching the repo-wide
viser convention).

---

## 8. Combining Arm + Hand

A full manipulator URDF is assembled offline from separate arm and hand URDFs.
`merge_urdf.py` loads each into a `RobotWrapper`, computes the fixed
arm→wrist transform from `DEVICE2WRIST`, and calls `generate_urdf` (a `xacro`
subprocess) to emit `rsc/robot/<arm>_<hand>.urdf`. That merged URDF is what
`RobotWrapper`, `CuroboPlanner`, and the visualizers consume.

```{mermaid}
flowchart LR
    AU["arm.urdf"] --> MG["merge_urdf<br/>DEVICE2WRIST offset"]
    HU["hand.urdf"] --> MG
    XA["robot_combined.urdf.xacro"] --> MG
    MG --> OUT["rsc/robot/&lt;arm&gt;_&lt;hand&gt;.urdf"]
```

---

## 9. Component Reference

| Layer | Component | File | Responsibility |
|-------|-----------|------|----------------|
| Factory | `get_arm` / `get_hand` | `paradex/io/robot_controller/__init__.py` | Resolve name → controller via `network_info` |
| Arm | `XArmController` | `paradex/io/robot_controller/xarm_controller.py` | 100 Hz servo/position loop, logging |
| Hand | `AllegroController` | `paradex/io/robot_controller/allegro_controller.py` | ROS2 node, 16-DoF publish/subscribe |
| Hand | `InspireControllerIP` | `paradex/io/robot_controller/inspire_controller_ip.py` | Modbus TCP, 6-DoF + force + tactile |
| Hand | `InspireController` | `paradex/io/robot_controller/inspire_controller.py` | USB serial, 6-DoF + force |
| Kinematics | `RobotWrapper` | `paradex/robot/robot_wrapper.py` | Pinocchio FK / IK / Jacobian |
| Planning | `CuroboPlanner` | `paradex/robot/curobo.py` | CuRobo GPU motion planning |
| URDF | `generate_urdf` | `paradex/robot/urdf.py` | xacro → URDF |
| URDF | `get_robot_urdf_path` | `paradex/robot/utils.py` | Resolve arm/hand URDF paths |
| Retarget | `parse_inspire` | `paradex/robot/mimic_joint.py`, `inspire.py` | Motor units → URDF joint radians (mimic joints) |

---

## 10. Validation & Usage

| Script | Exercises | Hardware |
|--------|-----------|----------|
| `src/validate/robot/xarm_base_wiggle.py` | `get_arm("xarm")`, servo `move`, home on exit | XArm |
| `src/validate/robot/allegro.py` | `get_hand("allegro")`, per-joint sweep via `move` | Allegro |
| `src/validate/robot/inspire_left.py` | `InspireControllerIP`, pose list + sinusoid, `get_qpos` | Inspire (TCP) |
| `src/validate/robot/inspire_left_gui.py` | force + tactile + `calibrate_force` GUI | Inspire (TCP) |
| `src/validate/robot/inspire.py` | `InspireController` USB, tactile heatmap | Inspire (USB) |
| `src/capture/robot/teleop_real.py` | `CaptureSession` teleop recording (arm+hand) | full rig |
| `src/util/robot/merge_urdf.py` | build combined arm+hand URDF | none |
| `src/util/robot/replay.py` | live `get_data()["qpos"]` → Viser | XArm |

Method-by-method API (parameters / returns): {doc}`Robot Control — API <robot_api>`.

---

## 11. Notes & Gotchas

- **Factory coupling to `network_info`.** `get_arm`/`get_hand` spread connection
  params straight from `system/current/network.json`; a missing key raises at
  construction. `get_hand("inspire", ip=True)` uses `network_info["inspire"]`
  (`ip`+`port`); `ip=False` uses `network_info["inspire_usb"]["param"]`.
- **Franka is stubbed.** The `franka` branch in `get_arm` is commented out — only
  `xarm` currently returns a controller.
- **`action` units differ per device.** Arm radians / pose vs. Allegro radians vs.
  Inspire 0–1000 motor units. `parse_inspire` converts motor units to URDF joint
  radians (and fills mimic joints) when driving a `RobotWrapper` from hardware
  logs.
- **CuroboPlanner has latent gaps.** `plan_full_step` references `self.rough_steps`
  and `self.tensor_args`, neither set in `__init__` — that path is incomplete;
  `plan_goalset` / `plan_to_joint_target` are the supported entry points.
- **`start()` before `stop()`.** Logging buffers only fill between `start(save_path)`
  and `stop()`; `end()` calls `stop()` for you if a session is still open.
