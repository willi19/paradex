# CLAUDE.md — src/inference/grasp_eval/real

## Purpose
Execute / preview a DexGraspNet grasp of the pringles can on the real XArm + Allegro.
Capture -> object 6D pose -> map grasp into robot frame -> IK -> GUI execution
(`object6d.py`) or Viser preview (`visualize.py`).

## Files
- `object6d.py` — `for index in [6]`: capture image, `get_object_6d` (triangulate markers +
  `SOLVE_XA_B`), project mesh and save `projected/`, `obj_T = inv(c2r) @ obj_T`. Load
  `dexgraspnet/results/pringles/6/{qpos,wrist_6d}.npy`, `wrist_6d = obj_T @ wrist_6d`,
  `robot.solve_ik(wrist_6d,"palm_link")`, FK on `link6` for the arm target, then
  `RobotGUIController(arm, hand, {"grasp":pick_action}, {"start":zeros16,"grasp":qpos}).run()`.
  Loads marker offsets from `shared_dir/object/marker_offset/pringles/0/marker_offset.npy`.
- `visualize.py` — same pose estimation; loads `wrist_6d.npy` + `qpos.npy`, reorders Allegro
  qpos (thumb-first: `[12:16]`->first 4), IK on `palm_link`, `ViserViewer` add_robot/object/traj.
  Trailing dead `for i in range(10)` qpos-print loop.
- `grasp_test.py` — `RobotGUIController(get_arm("xarm"))` only; `.run()`. GUI smoke test.
- `asdf.py` — SCRATCH/BROKEN. Defines `get_current_object_6d_marker` and a `__main__`
  pick-place loop, but references many undefined names (`get_image`, `triangulate_marker`,
  `rigid_transform_3D`, `load_latest_C2R`, `get_pcinfo`, `sensors`, `start_event`,
  `RemoteCameraController`, `home_robot`, `get_traj`, `simulate`, `run_script`, etc.) and
  is missing nearly all imports. Useful only as a sketch of the intended capture+execute loop.

## paradex modules used
- `remote_camera_controller("object6d")`, `ImageDict`, `SOLVE_XA_B`, `load_current_C2R`,
  `save_current_camparam`, `RobotWrapper`+`get_robot_urdf_path("xarm","allegro")`,
  `gui_controller_prev.RobotGUIController`, `get_arm`/`get_hand`, `ViserViewer`.
- `asdf.py` imports only `load_current_camparam, load_current_C2R` — rest unimported.

## Data flow & IO
- In: `shared_dir/object/marker_offset/pringles/0/marker_offset.npy`,
  `rsc/object/pringles/pringles.obj`, `dexgraspnet/results/pringles/<index>/...`, cam params + C2R.
- Capture dir: `~/shared_data/inference/grasp_eval/<timestamp>/raw`; writes `projected/`.

## When working here
- Run from repo root. Object `pringles` and grasp `index=6` hardcoded.
- `object6d.py` drives the REAL robot via GUI — verify pose before pressing grasp.

## Gotchas
- `asdf.py` will NOT run (undefined symbols, missing imports). Treat as a design sketch.
- This dir uses `shared_dir` marker_offset path (unlike `bodex`/`grasp_w_gui` which use local).
- `object6d.py` here uses `gui_controller_prev` (older controller); `bodex/object6d.py` uses `gui_controller`.
- `visualize.py` trailing loop loads a likely-missing `qpos.npy` and only prints — ignore.
