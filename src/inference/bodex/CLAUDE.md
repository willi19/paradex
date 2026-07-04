# CLAUDE.md — src/inference/bodex

## Purpose
Pringles grasp inference using BODex-generated grasps. Estimate object 6D pose from the
multi-cam rig, map a BODex wrist+hand grasp into robot frame, then project the mesh
(`object6d.py`) or visualize robot+object in Viser (`visualize.py`). `get_floor.py` is a
charuco floor-plane probe.

## Files
- `object6d.py` — `for index in [1]`: capture image -> `get_object_6d` (`triangulate_markers`
  + `SOLVE_XA_B`) -> `obj_mesh.apply_transform(obj_T)` -> `img_dict.project_mesh` -> `cv2.imshow`
  -> save `obj_T.npy` (robot frame, `inv(c2r) @ obj_T`). Also defines `normalize_cylinder`
  (unused here). Loads local `marker_offset.npy`.
- `visualize.py` — same pose estimation, loads `bodex/scale010_grasp.npy`
  (`data['robot_pose'][0, index]`), builds `wrist_6d` from quat+trans, reorders Allegro
  qpos (thumb-first), `robot.solve_ik(wrist_6d, "palm_link")`, then `ViserViewer` add_robot/
  add_object/add_traj. Ends with a dead `for i in range(10)` loop loading nonexistent
  `dexgraspnet/results/pringles/1/qpos.npy`.
- `get_floor.py` — capture -> `img_dict.triangulate_charuco()["1"]["checkerCorner"]` ->
  transform into robot frame via `inv(c2r)`. Has `import pdb; pdb.set_trace()` mid-script.

## paradex modules used
- `remote_camera_controller("object6d")`, `ImageDict` (from_path/undistort/triangulate_markers/
  triangulate_charuco/project_mesh/save/merge), `SOLVE_XA_B`, `load_current_C2R`,
  `save_current_camparam`, `RobotWrapper`+`get_robot_urdf_path("xarm","allegro")`, `ViserViewer`.
- `object6d.py` imports `gui_controller.RobotGUIController`; `visualize.py` imports the older
  `gui_controller_prev.RobotGUIController` — neither is actually used.

## Data flow & IO
- In: local `./marker_offset.npy`, `rsc/object/pringles/pringles.obj`,
  `bodex/scale010_grasp.npy`, current cam params + C2R.
- Capture dir: `~/shared_data/inference/grasp_eval/<timestamp>/raw`.
- Out: `object6d.py` writes `projected/` + `obj_T.npy`.

## When working here
- Run from repo root (relative `bodex/...` and `marker_offset.npy` paths).
- `index` is hardcoded to 1; object hardcoded to `pringles`.

## Gotchas
- `get_floor.py` blocks on `pdb.set_trace()` — interactive only.
- `marker_offset.npy` is loaded from cwd, not from `shared_dir` (differs from `grasp_eval/real`).
- `visualize.py`'s trailing loop is dead/broken (missing file) — ignore it.
- BODex qpos layout is index-joints-first then thumb; code reorders to Allegro thumb-first.
