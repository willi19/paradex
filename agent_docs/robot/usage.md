# paradex/robot — How to Use

The kinematics / planning / URDF layer. Pure computation on a URDF; the currency is a
joint-position vector `qpos` (np.ndarray). Nothing here talks to hardware — to move a real
arm/hand, hand your result to `paradex/io/robot_controller/` (`get_arm`/`get_hand`).

> Editing these modules (the IK loop, the CuRobo world, the joint mapping)? Read
> [`internals.md`](internals.md). Full architecture + per-symbol API + the hardware-controller
> half → `docs/robot.md` / `docs/robot_api.md`.

Import surface: only `RobotWrapper` is re-exported from the package
(`from paradex.robot import RobotWrapper`). Everything else is by submodule path.

---

## RobotWrapper — FK / IK / Jacobian {#RobotWrapper}

```python
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.robot.utils import get_robot_urdf_path

robot = RobotWrapper(get_robot_urdf_path(arm_name="xarm6"))   # or hand_name=, or both
```

| call | returns | notes |
|------|---------|-------|
| `compute_forward_kinematics(qpos, link_list=[...])` | `{link_name: 4x4}` | FK; pass the link names you want back. `link_list=[]` → `{}` (still updates internal state). |
| `solve_ik(target_pose, end_effector_name, q_init=None, try_num=30, ...)` | `(qpos, success)` | damped least-squares + **random restarts**; always check `success`. |
| `get_link_pose(link_id)` / `get_joint_pose(joint_id)` / `get_frame_pose(frame_id)` | `4x4` | needs a prior `compute_forward_kinematics(qpos)`. |
| `compute_single_link_local_jacobian(qpos, link_id)` | `6xN` | Pinocchio frame Jacobian. |
| `get_all_link_poses()` | `{name: 4x4}` | every frame; needs a prior FK call. |
| `joint_names` / `dof_joint_names` / `dof` / `link_names` / `joint_limits` | — | model introspection; `joint_limits` is `(N,2)` `[lower, upper]`. |
| `get_link_index(name)` / `get_joint_index(name)` | int | name → id; IDs feed the pose getters. |
| `get_end_links()` / `get_root_link()` | names | leaf links / the single root (raises if ambiguous). |

```python
# FK for hand-eye (this is exactly what src/calibration/handeye/calculate.py does):
eef = robot.compute_forward_kinematics(qpos, link_list=["link6"])["link6"]   # 4x4

# IK to a wrist target:
q, ok = robot.solve_ik(target_pose_4x4, end_effector_name="link6", q_init=q_now)
if not ok:
    ...   # no solution within joint limits after try_num restarts (q is the last iterate)
```

---

## CuroboPlanner — collision-free planning (GPU) {#CuRobo}

Heavyweight: `__init__` loads the robot config, builds the world, and **warms up**
(compiles CUDA graphs) — construct once, reuse. Needs a CUDA GPU and CuRobo installed.

```python
from curobo.types.base import TensorDeviceType
from paradex.robot.curobo import CuroboPlanner

planner = CuroboPlanner(obstacle_dict, obj_dict, robot_cfg, tensor_args=TensorDeviceType())
```

- `obstacle_dict` = `{"cuboid": {name: {"pose": [x,y,z,qw,qx,qy,qz], "dims": [..], "color": [..]}}}`
- `obj_dict` = `{name: {"pose": 4x4, "file_path": mesh}}` — poses are `(4,4)` matrices,
  converted to CuRobo `Pose` (position + **wxyz** quaternion) internally.

| entry point | goal | returns |
|-------------|------|---------|
| `plan_goalset(init_state, goal_pose)` | reach **any** of N candidate `(N,4,4)` poses | `(goalset_index, trajectory)` — `trajectory` is `(T, dof)` np |
| `plan_to_joint_target(init_state, goal_joint_state)` | reach a joint config | `(success: bool, trajectory | None)` |
| `update_world(obj_dict)` | swap collision meshes between plans | — |
| `get_robot_mesh(joint_state)` | visual meshes at a config | list of meshes |

```python
ok, traj = planner.plan_to_joint_target(q_start, q_goal)
if ok:
    for q in traj:                      # (T, dof)
        arm.move(q, is_servo=True)      # execute via paradex/io/robot_controller
```

> `plan_full_step(...)` (batch grasp planning) is **incomplete** — it reads
> `self.rough_steps`/`self.tensor_args` which `__init__` never sets. Use `plan_goalset` /
> `plan_to_joint_target`. See internals.

---

## URDF paths, generation, merge {#URDF}

```python
from paradex.robot.utils import get_robot_urdf_path
get_robot_urdf_path(arm_name="xarm6")               # rsc/robot/xarm6.urdf
get_robot_urdf_path(hand_name="allegro")            # rsc/robot/allegro_float.urdf
get_robot_urdf_path(arm_name="xarm6", hand_name="allegro")  # rsc/robot/xarm6_allegro.urdf
```
The file must already exist in `rsc/robot/`. To *build* a combined arm+hand URDF, use
`src/util/robot/merge_urdf.py` (computes the arm→wrist offset from `DEVICE2WRIST` and calls
`generate_urdf`).

```python
from paradex.robot.urdf import generate_urdf
generate_urdf(xacro_path, output_path, {"arg": "value"})   # shells out to `xacro`
```

---

## Inspire hand joint mapping {#Inspire}

Convert an Inspire trajectory (6 motor values per frame, **0–1000**, 0 = closed / 1000 = open)
into URDF joint radians, filling the mimic joints:

```python
from paradex.robot.mimic_joint import parse_inspire   # robot joint order (0=pinky … 5=thumb)
joints = parse_inspire(inspire_traj)                   # (T, 6) -> (T, 12)
```
`paradex/robot/inspire.py` has the **human-order** variant (`parse_inspire`) and
`parse_inspire_mimic`. Pick by the joint order your caller expects — the two files differ
only in `joint_order` defaults and whether mimic joints are filled (see internals).

---

## xArm kinematic calibration {#xarm-calib}

Make a URDF's FK match a specific physical xArm (factory per-unit calibration, units built
after Aug 2023):

```python
from paradex.robot.xarm_kinematic_calib import (
    read_xarm_kinematic_params, apply_kinematics_to_urdf, save_kinematic_yaml)

params = read_xarm_kinematic_params(robot_ip)           # TCP/Modbus, port 502
diff   = apply_kinematics_to_urdf(urdf_path, params["kinematics"])   # patches joint origins
```
`apply_kinematics_to_urdf` backs up the original to `<urdf>.original` on first run and always
re-patches **from that backup** → re-running is idempotent. Returns a per-joint
`{d_xyz_mm, d_rpy_deg}` diff. Driver script: `src/calibration/xarm_kinematic_calibration.py`.

---

## Gotchas
- `RobotWrapper` requires `nq == nv` — raises `NotImplementedError` on continuous/quaternion
  joints. Merged URDFs must use revolute/prismatic joints only.
- `solve_ik` is **randomized** and returns `(q, success)`; a failed solve still returns the
  last `q`. Never use the pose without checking the flag.
- The pose getters (`get_link_pose`, `get_all_link_poses`, …) read cached state — call
  `compute_forward_kinematics(qpos)` first, or you'll read the previous config.
- `get_joint_indices` and `get_root_link` **print** to stdout (leftover debug) — noisy in loops.
- CuRobo uses **wxyz** quaternions (repo-wide viser convention); Pinocchio/`RobotWrapper` uses
  SE3 matrices. `curobo.py` does the `xyzw → wxyz` reorder for you.
- Don't import `robot_wrapper_updating.py` — it's a WIP fork with an incompatible
  `compute_forward_kinematics` signature (no `link_list`).
