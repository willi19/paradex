# CLAUDE.md — src/inference

## Purpose
Inference pipelines: estimate object 6D pose from the multi-camera rig, fetch/generate
a dexterous grasp, then visualize or execute it on XArm + Allegro. All variants share
one backbone (capture -> triangulate markers -> solve `obj_T` -> map grasp via `C2R` ->
IK -> Viser/GUI). The subdirs differ mainly in the grasp source and the execution UI.

## Subdirectories
- `bodex/` — grasps from `bodex/scale010_grasp.npy`; also `get_floor.py` (charuco floor probe).
- `grasp_eval/real/` — physical-robot grasp eval using `dexgraspnet/results/...`.
- `grasp_w_gui/` — BODex grasp with pregrasp/grasp/squeeze stages via GUI.
- `pringles_test/` — pick-and-place from refined lookup trajectories.

## Shared code flow (copied, not imported, across most scripts)
- `get_object_6d(obj_name, filename)`: `ImageDict.triangulate_markers()` -> match marker IDs
  against `marker_offset` -> `SOLVE_XA_B(A, B)` -> 4x4 `obj_T` in **camera** frame.
- Then `obj_T = inv(c2r) @ obj_T` to move into **robot** frame.
- `normalize_cylinder(obj_6D)`: snaps a cylindrical object's axis upright/lying — used
  before executing grasps so the grasp lines up with the symmetric can.
- Wrist mapped to robot frame: `wrist_6d = obj_T @ wrist_6d`; IK on `"palm_link"`.
- Allegro joint reorder: source qpos is `[index..., thumb]`; Allegro wants thumb first:
  `qpos[:4] = src[12:16]; qpos[4:] = src[:12]`.

## paradex modules used
- `paradex.io.camera_system.remote_camera_controller.remote_camera_controller`
- `paradex.image.image_dict.ImageDict` (`from_path`, `undistort`, `triangulate_markers`,
  `triangulate_charuco`, `project_mesh`, `save`, `merge`)
- `paradex.transforms.conversion.SOLVE_XA_B`
- `paradex.calibration.utils` — `load_current_C2R`, `save_current_camparam`, `load_current_camparam`
- `paradex.robot.robot_wrapper.RobotWrapper`, `paradex.robot.utils.get_robot_urdf_path`
- `paradex.io.robot_controller` — `get_arm("xarm")`, `get_hand("allegro")`, `RobotGUIController`
  (note: both `gui_controller` and the older `gui_controller_prev` are imported across files)
- `paradex.visualization.visualizer.viser.ViserViewer`

## Data flow & IO
- Captures images to `~/shared_data/inference/grasp_eval/<timestamp>/raw`, then `undistort` in place.
- Saves cam params via `save_current_camparam(...)`; some write `obj_T.npy`, `projected/`.
- Reads grasps from `dexgraspnet/results/<obj>/<idx>/{qpos,wrist_6d}.npy` or `bodex/*.npy`
  (relative paths -> must run from the cwd where those exist, typically repo root).

## When working here
- This is experimental code. Object (`pringles`), grasp `index`, and many paths are hardcoded.
- Do NOT "fix" the `dataset_acqusition` typo elsewhere; not relevant here but project-wide rule.
- Run from repo root unless a script's relative paths say otherwise.

## Gotchas
- `marker_offset` path is inconsistent: `bodex/`, `grasp_w_gui/`, `pringles_test/grasp.py`
  load a local `./marker_offset.npy`; `grasp_eval/real/object6d.py` & `visualize.py` load the
  proper `shared_dir/object/marker_offset/<obj>/0/marker_offset.npy`.
- `object6d.py` in `bodex/` and `get_floor.py` contain a `pdb.set_trace()` / debug prints.
- `asdf.py` files are scratch: `grasp_eval/real/asdf.py` references undefined functions and
  will not run as-is; `pringles_test/asdf.py` is a trivial load-only snippet.
- `viser`-based `visualize.py` scripts end with a dead `for i in range(10)` loop printing qpos.
