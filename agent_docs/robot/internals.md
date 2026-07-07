# paradex/robot — Internals (for editors)

You're changing `paradex/robot/` itself. Eight files; the weight is in `robot_wrapper.py`
(Pinocchio FK/IK) and `curobo.py` (CuRobo planning). This layer never touches hardware and
never *solves* calibration — it consumes a URDF and returns joint/pose data.

> Just calling these? Read [`usage.md`](usage.md). Architecture + diagrams → `docs/robot.md`.

---

## robot_wrapper.py — `RobotWrapper` {#robot_wrapper}

Thin Pinocchio wrapper (adapted from dex-retargeting). State: `self.model` +
`self.data` (built from the URDF), `self.q0 = pin.neutral(model)`. `__init__` **rejects**
`nq != nv` with `NotImplementedError` — i.e. no continuous/floating/quaternion joints; every
DoF must be a plain revolute/prismatic. Merged arm+hand URDFs are authored to satisfy this.

**FK.** `compute_forward_kinematics(qpos, link_list=[])` runs `pin.forwardKinematics` then, for
each requested link, `updateFramePlacement` → `.homogeneous`. Returns `{name: 4x4}`. The bare
pose getters (`get_link_pose`, `get_joint_pose`, `get_frame_pose`, `get_all_link_poses`) read
`self.data` (`oMi`/`oMf`) and therefore **depend on a prior FK call** — they don't recompute.

**IK.** `solve_ik(target_pose, end_effector_name, q_init, max_iter, tol, alpha, try_num)` is a
damped least-squares Gauss–Newton loop:
- Outer loop = `try_num` random restarts (`q_init` only used on the first try when provided;
  every other restart samples `np.random.uniform(q_min, q_max)`).
- Inner: 6D error `pin.log(target⁻¹ · current).vector`; step
  `Δq = -(JᵀJ + 1e-6·I)⁻¹ Jᵀ e`; `q = integrate(q, Δq, alpha)`; clip to limits.
- On `‖error‖ < tol` it does a **second acceptance check** (`‖ΔR‖<0.01` and `‖Δp‖<0.01`); if
  that fails it `break`s to the next restart. Returns `(q, True)` on accept, `(q, False)` after
  all restarts exhaust (q = last iterate).

**Topology helpers.** `get_end_links()` = BODY frames never used as a joint parent;
`get_root_link()` = the single link that's never a child (raises `RuntimeError` if 0 or >1).

### Landmines (don't copy; fix deliberately)
- `get_joint_indices` (L61) and `get_root_link` (L196/L202) contain leftover `print(...)` — remove
  when you touch them; they spam stdout inside loops.
- IK is **stochastic** — no seed. Two calls can return different `q` for the same target. If you
  need determinism, seed numpy or thread a seed through.
- The pose getters silently use stale `self.data` if FK wasn't called for the current `qpos`.
  Order-of-operations bug magnet.

---

## curobo.py — `CuroboPlanner` {#curobo}

Wraps CuRobo `MotionGen` + `IKSolver`. Free functions: `to_quat(4x4) -> [x,y,z, qw,qx,qy,qz]`
(scipy `xyzw` → CuRobo `wxyz` reorder), `load_world_config(obstacle_dict, obj_dict)` → CuRobo
`WorldConfig` dict (`cuboid` + `mesh`).

`__init__(obstacle_dict, obj_dict, robot_cfg, tensor_args)`:
- builds `self.world_cfg`, a `MotionGenConfig` (mesh collision checker, 64 IK / 64 trajopt / 32
  graph seeds, `trajopt_tsteps=80`, `interpolation_dt=0.01`), then `MotionGen(...)` and
  **`self.motion_gen.warmup(enable_graph=True, ...)`** — seconds of CUDA-graph compile.
- also builds a standalone `IKSolver` (`self.ik_solver`).
- config paths are **hardcoded** to `home_path/curobo/...` (`my_particle_ik.yml`) — machine-specific.

Entry points (see usage.md for signatures): `plan_goalset` (single start, N goal poses →
`plan_goalset`), `plan_to_joint_target` (`plan_single_js`), `update_world`, `get_robot_mesh`.
All pose inputs go `(4,4)` → position tensor + `wxyz` quaternion tensor.

### Known-broken / incomplete
- **`plan_full_step` is not runnable as-is**: it reads `self.tensor_args` and `self.rough_steps`,
  **neither of which `__init__` sets** (`tensor_args` is only a ctor arg; `rough_steps` never
  exists). It also sets `self.batched_plans`/`self.batch_idx`/`self.mpc_idx` that nothing else
  consumes. Treat it as dead/WIP — wire in the missing state before relying on it.
- The commented-out `plan_single` fallback + `reset_cuda_graph` lines are stale experiments.
- MPC imports (`MpcSolver`) are pulled in but unused.

If you extend the planner, the two supported paths (`plan_goalset`, `plan_to_joint_target`) are
what callers use (`src/util/robot/get_bounding_sphere.py`, grasp pipelines in `src/inference/`).

---

## inspire.py / mimic_joint.py — Inspire joint mapping {#inspire}

Two near-duplicate files converting Inspire motor units → URDF joint radians. Shared tables:
`limit` (per-finger max radians) and `mimic_joint` (dependent joints: `*_2_joint` mirrors
`*_1_joint`; thumb 3/4 driven off `thumb_2` with ratios 1.0/1.2).

Formula (both files): `ret[:, j] = limit[name] * (1 - inspire_traj[:, i] / 1000)` over the six
actuated joints `right_{little,ring,middle,index}_1`, `right_thumb_{2,1}`. So **1000 = open
(0 rad), 0 = closed (max rad)**.

The files differ only in defaults and mimic handling:
| | `mimic_joint.py :: parse_inspire` | `inspire.py :: parse_inspire` | `inspire.py :: parse_inspire_mimic` |
|--|--|--|--|
| default `joint_order` | 12-joint **robot** order | 6-joint **human** order | 12-joint robot order |
| fills mimic joints | yes (guards `if joint_name not in joint_order`) | no | yes (no guard) |

The duplication is intentional (different call sites want different orders). If you edit the
`limit`/`mimic_joint` tables, edit **both files** or they'll silently disagree. Callers:
`src/validate/robot/inspire_*`, `src/process/miyungpa/*`.

---

## urdf.py / utils.py — URDF plumbing

- `generate_urdf(xacro_path, output_path, args_dict)` = `subprocess.run(["xacro", xacro, k:=v...],
  stdout=file, check=True)`. Requires `xacro` on PATH (ROS). Prints the output path.
- `get_robot_urdf_path(arm_name, hand_name)`: `arm` only → `<arm>.urdf`; `hand` only →
  `<hand>_float.urdf`; both → `<arm>_<hand>.urdf`, under `rsc_path/robot/`. Pure path
  construction — **doesn't check existence**; a missing combo fails later at `RobotWrapper`.

---

## xarm_kinematic_calib.py — factory calibration → URDF

Reads UFACTORY per-unit kinematic calibration over TCP/Modbus (port 502; protocol mirrors
`xarm_ros2 gen_kinematics_params.py`) and patches URDF joint origins.
- `read_xarm_kinematic_params(ip)` sends a fixed 7-byte request, expects **exactly 179 bytes**
  with `recv[8]` truthy, unpacks `<42f` → `{joint_i: {x,y,z,roll,pitch,yaw}}`. Raises a
  descriptive `RuntimeError` on short read / clear flag (older units have no register).
- `apply_kinematics_to_urdf(urdf, params, joint_names=joint1..6)`: backs up to
  `<urdf>.original` on first call, **always parses the backup** (idempotent, never compounds),
  writes `origin/@xyz,@rpy` with `%.10g`, returns `{joint: {d_xyz_mm, d_rpy_deg}}`.
- `save_/load_kinematic_yaml` round-trip the `kinematics` block in xarm_ros2 YAML format.

Driver: `src/calibration/xarm_kinematic_calibration.py`; comparison check:
`src/validate/calibration/compare_xarm_kinematic_calib.py`.

---

## robot_wrapper_updating.py — WIP fork

A divergent copy of `RobotWrapper` whose `compute_forward_kinematics(qpos)` takes **no**
`link_list` and returns nothing (callers query pose getters separately). Not exported, not
imported anywhere. Don't build on it; if you're reconciling the two, `robot_wrapper.py` is the
live one (`__init__.py` re-exports it).

---

## Consumers (verify against these when you change signatures)
- `RobotWrapper`: `src/calibration/handeye/calculate.py` (FK `link6`), `src/inference/**`
  (grasp IK/FK), `src/validate/calibration/compare_xarm_kinematic_calib.py`, visualizers.
- `CuroboPlanner`: `src/util/robot/get_bounding_sphere.py`, `src/inference/**` grasp planning.
- `parse_inspire`: `src/validate/robot/inspire_*`, `src/process/miyungpa/*`.
- `get_robot_urdf_path` / `generate_urdf`: `src/util/robot/merge_urdf.py`, `src/inference/**`.
- `xarm_kinematic_calib`: `src/calibration/xarm_kinematic_calibration.py`.

No test suite — validate by re-running a hand-eye FK (`calculate.py`), a grasp IK, or a
`merge_urdf` build.
