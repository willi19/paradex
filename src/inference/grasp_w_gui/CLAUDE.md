# CLAUDE.md — src/inference/grasp_w_gui

## Purpose
Pringles grasp from BODex grasps, executed via `RobotGUIController` with multi-stage hand
poses (start/pregrasp/grasp/squeezed). Capture -> object 6D pose -> map grasp -> IK -> GUI.
`visualize.py` is the Viser dry-run.

## Files
- `object6d.py` — `for index in [1]`: capture, `get_object_6d` (triangulate + `SOLVE_XA_B`),
  project mesh -> `projected/`, `obj_T = inv(c2r) @ obj_T`. Load `bodex/scale010_grasp.npy`
  `data['robot_pose'][0, index]` (shape ~(3, ...)). Build `wrist_6d` from quat+trans of frame 0.
  Reorder Allegro qpos for all 3 frames (thumb-first). `wrist_6d = obj_T @ wrist_6d`, IK on
  `palm_link`, FK on `link6`. `squeezed_qpos = qpos[1]*8 - qpos[0]*7`. Launch
  `RobotGUIController(arm, hand, {"grasp":pick_action}, {"start":zeros, "pregrasp":qpos[0],
  "grasp":qpos[1], "squeezed":squeezed_qpos}).run()`. Loads local `./marker_offset.npy`.
- `visualize.py` — same pose estimation; uses BODex frame 0 wrist + frame 1 qpos (reordered),
  IK on `palm_link`, `ViserViewer` add_robot/object/traj. Has an unused `q_delta`/`R_delta`
  block and a trailing dead `for i in range(10)` qpos-print loop (missing dexgraspnet file).

## paradex modules used
- `remote_camera_controller("object6d")`, `ImageDict`, `SOLVE_XA_B`, `load_current_C2R`,
  `save_current_camparam`, `RobotWrapper`+`get_robot_urdf_path("xarm","allegro")`,
  `get_arm`/`get_hand`, `ViserViewer`.
- `object6d.py` imports `gui_controller.RobotGUIController`; `visualize.py` imports
  `gui_controller_prev.RobotGUIController` (unused in visualize).

## Data flow & IO
- In: local `./marker_offset.npy`, `rsc/object/pringles/pringles.obj`,
  `bodex/scale010_grasp.npy`, cam params + C2R.
- Capture dir: `~/shared_data/inference/grasp_eval/<timestamp>/raw`; writes `projected/`.

## When working here
- Run from repo root. Object `pringles`, index 1, hardcoded.
- `squeezed` is a linear extrapolation past the grasp — be cautious executing on hardware.

## Gotchas
- `marker_offset.npy` loaded from cwd (not `shared_dir`) — differs from `grasp_eval/real`.
- BODex qpos reorder: source `[index12, thumb4]` -> Allegro `[thumb4, index12]`.
- `visualize.py` trailing loop is dead/broken — ignore.
