# agent_docs/robot — agent orientation

Docs for **AI agents working on `paradex/robot/`** — the **kinematics + motion-planning** layer.
Given a URDF and a joint vector `qpos`, this module answers "where is each link?" (FK), "what
`qpos` reaches this pose?" (IK), and "what collision-free trajectory gets there?" (CuRobo). It is
pure computation: it does **not** touch hardware (that's `paradex/io/robot_controller/`, see
[`agent_docs/robot_controller/`](../robot_controller/README.md)); the two layers meet only at `qpos`.
Read the **one** file for your task; don't scan everything.

Mental model: **URDF path → `RobotWrapper` (Pinocchio model + data)**. Every FK/IK/Jacobian query
runs against that in-memory model. `curobo.py` is the heavy GPU planner (separate CuRobo stack, not
Pinocchio); the rest are small helpers. All poses are **4×4 homogeneous matrices**; URDF lengths are
in **meters**, angles in **radians**. Only `RobotWrapper` is re-exported from `paradex.robot`;
everything else is imported by submodule path.

| Your task | Read |
|-----------|------|
| **Use** FK / IK / Jacobian (`RobotWrapper`), plan a trajectory (`CuroboPlanner`), resolve/build a URDF, decode an Inspire trajectory | [`usage.md`](usage.md) |
| Run xArm factory kinematic calibration into a URDF | [`usage.md`](usage.md) §xarm-calib |
| **Edit** the IK loop, the CuRobo world/warmup, the joint-mapping tables, the URDF plumbing | [`internals.md`](internals.md) |
| How the *hardware* side consumes `qpos` (move a real arm/hand) | [`agent_docs/robot_controller/`](../robot_controller/README.md) |

Rule of thumb: **calling** these modules → `usage.md`; **editing** them → `internals.md`.

## File map
| File | What it is |
|------|-----------|
| `robot_wrapper.py` | `RobotWrapper` — the Pinocchio FK/IK/Jacobian wrapper. **The module's entry point** (the only symbol re-exported from `paradex.robot`). |
| `curobo.py` | `CuroboPlanner` — CuRobo `MotionGen`+`IKSolver` GPU planner (collision-aware trajectories) + `to_quat`, `load_world_config` helpers. Heavy deps; **no in-repo callers** (only `to_quat` is imported). |
| `utils.py` | `get_robot_urdf_path(arm_name, hand_name)` — maps arm/hand names to URDF files under `rsc/robot/`. |
| `urdf.py` | `generate_urdf(xacro_path, output_path, args_dict)` — shells out to `xacro` to expand a `.xacro` into a `.urdf`. |
| `inspire.py` | `parse_inspire` / `parse_inspire_mimic` — convert Inspire-hand raw sensor trajectories (0–1000) into URDF joint angles (human joint-order default). |
| `mimic_joint.py` | `mimic_joint` / `limit` tables + a mimic-aware `parse_inspire` (robot joint-order default). Coupling + per-finger limits for the Inspire hand. |
| `xarm_kinematic_calib.py` | Read factory per-unit kinematic calibration from an xArm controller (TCP/Modbus port 502) and patch it into a URDF. |
| `robot_wrapper_updating.py` | **Stale/unused fork** of `RobotWrapper` (no in-repo importers, different FK signature). Do not build on it. |
| `__init__.py` | Re-exports `RobotWrapper` only. |

All paths relative to [`paradex/robot/`](../../paradex/robot/). Related: hardware controllers
([`agent_docs/robot_controller/`](../robot_controller/README.md)), URDF/mesh assets (`rsc/robot/`),
coordinate frames (`paradex/transforms/`, used by `src/util/robot/merge_urdf.py`).
