# CLAUDE.md — src/util/robot

## Purpose
Robot-model utilities: assemble combined arm+hand URDFs, visualize link geometry / collision spheres, and replay/inspect arm state in Viser.

## Files
- `merge_urdf.py` — args `--arm --hand`. Loads each URDF into `RobotWrapper`; `parent_link = arm.get_end_links()[0]`, `child_link = "wrist"`. Computes `arm2wrist = inv(DEVICE2WRIST[arm]) @ DEVICE2WRIST[hand]`, converts to xyz + xyz-euler rpy, then `generate_urdf(xacro_path, output_path, arg_dict)` -> `rsc/robot/<arm>_<hand>.urdf`.
- `visualize.py` — module-level script (no `__main__`/args). Hard-codes `arm_name="xarm"`, `hand_name="allegro"`, and only processes `link4`. Runs `RobotWrapper.compute_forward_kinematics(np.zeros(22))`, transforms each link visual mesh by its FK pose, builds icosphere meshes from `spheres/<arm>_<hand>.yml` `collision_spheres`, and shows them with Open3D (`visualize_meshlist`). Other helpers `visualize_meshes`/`visualize_single_robot` are defined but unused. Note the CuRobo `get_bounding_spheres` path is commented out.
- `replay.py` — args `--arm --hand`. `ViserViewer()`, `hand_dof = 6 if hand=="inspire" else 16`, reads `get_arm(arm).get_data()["qpos"]`, makes `action = concat(qpos, zeros(hand_dof))`, `add_robot` + `add_traj` (single frame), `arm.end(set_break=True)`, `start_viewer()`.
- `get_bounding_sphere.py` — EMPTY (0 bytes).
- `replay_sim.py` — EMPTY (0 bytes).

## paradex modules used
- `paradex.robot.robot_wrapper.RobotWrapper`, `paradex.robot.urdf.generate_urdf`, `paradex.robot.utils.get_robot_urdf_path`, `paradex.robot.curobo.to_quat`
- `paradex.utils.file_io.get_robot_urdf_path` / `rsc_path`
- `paradex.geometry.coordinate.DEVICE2WRIST`
- `paradex.io.robot_controller.get_arm`
- `paradex.visualization.visualizer.viser.ViserViewer`

## Data flow & IO
- merge_urdf: URDFs + `rsc/robot/robot_combined.urdf.xacro` -> `rsc/robot/<arm>_<hand>.urdf`.
- visualize: combined URDF + link meshes + `rsc/robot/spheres/<arm>_<hand>.yml` -> Open3D window only.
- replay: live arm qpos (network) + combined URDF -> Viser web viewer.

## When working here
- visualize.py is a scratch inspection script: change `link4` / `xarm_allegro` directly in source to inspect other links/robots.
- replay.py assumes 22 total DOF for xarm+allegro (6 arm + 16 hand); hand_dof branch only special-cases inspire (6).

## Gotchas
- Two stub files are empty — do not document behavior they don't have.
- `visualize.py` references an undefined `link_name` inside `visualize_meshlist`'s except clause (latent bug, only hit on mesh error).
- merge_urdf relies on `DEVICE2WRIST` having entries for both arm and hand names.
