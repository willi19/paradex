# Robot Control — API

Method reference for the robot stack: parameters (input) and return values (output)
per class. For the architecture and how these fit together, see the
{doc}`overview <robot>`.

Signatures are verified against the code; internal/private helpers are omitted.

Each entry is collapsed below — click to expand.

:::{dropdown} Factory: `get_arm` / `get_hand`
:open:

`paradex/io/robot_controller/__init__.py`. Resolves a device name to a concrete
controller, spreading connection params from `network_info`
(`system/current/network.json`).

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `get_arm(arm_name)` | `arm_name: str` | `XArmController` \| `None` | `"xarm"` → `XArmController(ip=...)`. `franka` branch is commented out; other names return `None`. |
| `get_hand(hand_name, tactile=False, ip=True)` | `hand_name: str`, `tactile: bool`, `ip: bool` | hand controller | `"inspire"`/`"inspire_left"` → `InspireControllerIP` (`ip=True`) or `InspireController` USB (`ip=False`); `"allegro"` → `AllegroController`. |
:::

:::{dropdown} `XArmController` (arm)

`paradex/io/robot_controller/xarm_controller.py`. Wraps `XArmAPI`; a 100 Hz loop
thread streams the latest target. Action is a `(6,)` joint vector (radians) **or**
a `(4,4)` homogeneous pose.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `XArmController(ip)` | `ip: str` | instance | Connect, enable servo mode, spawn control loop. |
| `.move(action, is_servo=True, speed=None)` | `action: ndarray (6,) \| (4,4)`, `is_servo: bool`, `speed: float=None` | `None` | Set target. `is_servo=True` streams (non-blocking); `is_servo=False` does a blocking position move and waits for arrival. Joint action clipped to ±2π. |
| `.start(save_path)` | `save_path: str` | `None` | Begin logging; makes dir, arms save event. |
| `.stop()` | — | `None` | Stop logging; write `time/position/velocity/torque/action/action_qpos` `.npy`. |
| `.end(set_break=False)` | `set_break: bool` | `None` | Exit loop, join thread, disconnect (`motion_enable(False)` if `set_break`). |
| `.get_data()` | — | `dict` | `{qpos: (6,), position: (4,4) homo, time}`. |
| `.clear_error()` | — | `None` | Clear err/warn, re-enable servo mode (no reconnect). |
| `.reset()` | — | `None` | (Re)connect `XArmAPI`, enable motion, enter servo mode. |
| `.is_error()` | — | `bool` | `True` if the arm reported an error/warning. |

**Attributes / events**: `.connect_event`, `.error_event`, `.position_control_event`,
`.save_event` (`threading.Event`); `.fps = 100`.

**Module helpers**: `homo2cart(h)`, `cart2homo(cart)`, `homo2aa(h)` convert between
`(4,4)` matrices and XArm's mm+rpy / mm+axis-angle cartesian vectors.
:::

:::{dropdown} `AllegroController` (hand, ROS2)

`paradex/io/robot_controller/allegro_controller.py`. Subclass of
`rclpy.node.Node`. 16-DoF, radians. Publishes `Float64MultiArray` on
`/allegro_hand_position_controller/commands`; subscribes `/joint_states`.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `AllegroController()` | — | instance | Create node, spawn spin + control (100 Hz) threads. Blocks internally until first `/joint_states`. |
| `.move(action)` | `action: array-like (16,)` | `None` | Set target joint positions (asserts DoF 16; clipped to `MAX_ANGLE=2.1`). |
| `.start(save_path)` | `save_path: str` | `None` | Begin logging (`action/time/position`). |
| `.stop()` | — | `None` | Stop logging; write `.npy` per key. |
| `.end()` | — | `None` | Exit loop, join threads, shut down executor. |
| `.get_data()` | — | `dict` | `{qpos: (16,) reordered via JS_TO_CMD, action: (16,), time}`. |

**Constants**: `action_dof = 16`, `MAX_ANGLE = 2.1`, `JS_TO_CMD` (joint-state →
command index map).
:::

:::{dropdown} `InspireControllerIP` (hand, Modbus TCP)

`paradex/io/robot_controller/inspire_controller_ip.py`. 6-DoF, motor units
0–1000 (0 = closed, 1000 = open). `move_hand` loop writes `angleSet`, reads
`angleAct` + `forceAct` (+ tactile when `tactile=True`).

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `InspireControllerIP(ip, port, tactile=False)` | `ip: str`, `port: int`, `tactile: bool` | instance | Open Modbus TCP client, preset speed/force/angle, spawn 100 Hz loop. |
| `.move(action)` | `action: ndarray (6,)` | `None` | Set target angles (0–1000). |
| `.start(save_path=None)` | `save_path: str=None` | `None` | Begin logging (`time/position/action/force/tactile`). |
| `.stop()` | — | `None` | Clear save event and `save()`. |
| `.save()` | — | `None` | Flush logged buffers to `.npy`. |
| `.end()` | — | `None` | Exit loop, join thread; `stop()` if still saving. |
| `.get_qpos()` | — | `ndarray (6,)` | Read `angleAct` register block. |
| `.get_data()` | — | `dict` | `{qpos: (6,), action: (6,), time}`. |
| `.get_force()` | — | `ndarray (6,)` | Read `forceAct` (signed). |
| `.get_tactile()` | — | `dict` | Per-pad arrays (requires `tactile=True`, else raises). |
| `.read_tactile(name)` | `name: str` | `ndarray (rows,cols)` | One pad from `TACTILE_LAYOUT`. |
| `.read_all_tactile()` | — | `dict[str, ndarray]` | Every pad. |
| `.home_robot(home_pose=None)` | `home_pose: ndarray (6,)=None` | `None` | Command the stored (or given) home pose. |
| `.set_homepose(home_pose)` | `home_pose: ndarray (6,)` | `None` | Replace the home pose. |
| `.calibrate_force(timeout=10.0, poll_interval=0.2)` | `timeout: float`, `poll_interval: float` | `int` | Trigger firmware force calibration, poll until done (raises on timeout). Hand must be open/untouched. |

**Register I/O**: `write6(reg_name, val)` / `read6(reg_name)` for
`angleSet/forceSet/speedSet/angleAct/forceAct`; `regdict` / `TACTILE_LAYOUT` map
names → Modbus addresses.
:::

:::{dropdown} `InspireController` (hand, USB serial)

`paradex/io/robot_controller/inspire_controller.py`. Same 6-DoF contract over a
serial link (`pyserial`), speaking the Inspire byte-frame protocol.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `InspireController(addr)` | `addr: str` (serial device) | instance | Open serial, preset speed/power/angle, spawn 100 Hz loop; blocks until connected. |
| `.move(action)` | `action: ndarray (6,)` | `None` | Set target angles (0–1000). |
| `.start(save_path)` | `save_path: str` | `None` | Begin logging (`time/position/action/force`). |
| `.stop()` | — | `None` | Stop logging; write `.npy`. |
| `.end()` | — | `None` | Exit loop, join thread; `stop()` if saving. |
| `.get_data()` | — | `dict` | `{action: (6,), joint_value: (6,), time}`. |
| `.get_force()` | — | `ndarray (6,)` | Read `getactforce`. |
| `.is_error()` | — | `bool` | Always `False`. |

**Layout constant**: `SENSOR_LAYOUT` (per-pad `addr/rows/cols`).
:::

:::{dropdown} `RobotWrapper` (kinematics, Pinocchio)

`paradex/robot/robot_wrapper.py`. FK / IK / Jacobian on a URDF model. No hardware.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `RobotWrapper(urdf_path)` | `urdf_path: str` | instance | Build Pinocchio model + data from URDF. |
| `.compute_forward_kinematics(qpos, link_list=[])` | `qpos: ndarray (dof,)`, `link_list: list[str]` | `dict[str, ndarray (4,4)]` | FK; returns homogeneous pose per requested link. |
| `.solve_ik(target_pose, end_effector_name, q_init=None, max_iter=1000, tol=1e-8, alpha=5e-2, try_num=30)` | `target_pose: ndarray (4,4)`, `end_effector_name: str`, `q_init: ndarray=None`, ... | `(ndarray (dof,), bool)` | Damped least-squares IK with random restarts; returns `(qpos, success)`. |
| `.get_link_pose(link_id)` | `link_id: int` | `ndarray (4,4)` | Frame placement as homogeneous matrix. |
| `.get_all_link_poses()` | — | `dict[str, ndarray (4,4)]` | Pose of every frame. |
| `.compute_single_link_local_jacobian(qpos, link_id)` | `qpos: ndarray`, `link_id: int` | `ndarray (6, dof)` | Frame Jacobian. |
| `.get_link_index(name)` | `name: str` | `int` | Frame id for a BODY link. |
| `.get_joint_index(name)` | `name: str` | `int` | Index within `dof_joint_names`. |
| `.integrate(q, v, dt)` | `q, v: ndarray`, `dt: float` | `ndarray` | `pin.integrate` (config-space step). |
| `.get_end_links()` | — | `list[str]` | Leaf links (never a joint parent). |
| `.get_root_link()` | — | `str` | Single root link (raises if ambiguous). |

**Properties**: `.joint_names`, `.dof_joint_names`, `.dof` (`nq`), `.link_names`,
`.joint_limits` (`(dof,2)` lower/upper).
:::

:::{dropdown} `CuroboPlanner` (motion planning, CuRobo)

`paradex/robot/curobo.py`. GPU planner over a world of cuboids + object meshes.
Heavyweight — construct once (it warms up CUDA graphs) and reuse.

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `CuroboPlanner(obstacle_dict, obj_dict, robot_cfg, tensor_args)` | `obstacle_dict: dict`, `obj_dict: dict`, `robot_cfg`, `tensor_args` | instance | Build world, load `MotionGen` + `IKSolver`, warm up. |
| `.plan_goalset(init_state, goal_pose)` | `init_state: array (dof,)`, `goal_pose: ndarray (N,4,4)` | `(goalset_index, trajectory ndarray)` | Plan to reach any of N candidate poses. |
| `.plan_to_joint_target(init_state, goal_joint_state)` | `init_state`, `goal_joint_state: array (dof,)` | `(success: bool, trajectory \| None)` | Joint-space plan via `plan_single_js`. |
| `.plan_full_step(current_state, target_positions, target_quats, num_grasp)` | `current_state: JointState`, `target_positions`, `target_quats: Tensor`, `num_grasp: int` | `(success: bool, tensor_ids \| None)` | Batch-plan grasp candidates, pick best per place. (References unset attrs — incomplete; see overview §11.) |
| `.update_world(obj_dict)` | `obj_dict: dict` | `None` | Rebuild + swap collision world. |
| `.get_robot_mesh(joint_state)` | `joint_state: array (dof,)` | visual meshes | Kinematics visual meshes at a config. |

**Module helpers**: `to_quat(obj_pose)` → `(7,)` position + **wxyz** quaternion;
`load_world_config(obstacle_dict, obj_dict)` → CuRobo world dict.
:::

:::{dropdown} URDF helpers

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `generate_urdf(xacro_path, output_path, args_dict)` | `xacro_path: str`, `output_path: str`, `args_dict: dict` | `None` | Run `xacro` with `key:=value` args, write URDF to `output_path`. (`paradex/robot/urdf.py`) |
| `get_robot_urdf_path(arm_name=None, hand_name=None)` | `arm_name: str=None`, `hand_name: str=None` | `str` | Resolve URDF path: hand-only → `<hand>_float.urdf`, arm-only → `<arm>.urdf`, both → `<arm>_<hand>.urdf`. (`paradex/robot/utils.py`) |
| `parse_inspire(inspire_traj, joint_order=...)` | `inspire_traj: ndarray (T,6)`, `joint_order: list[str]` | `ndarray (T, len(joint_order))` | Motor units (0–1000) → URDF joint radians via `limit`, filling mimic joints. (`paradex/robot/mimic_joint.py`) |
:::
