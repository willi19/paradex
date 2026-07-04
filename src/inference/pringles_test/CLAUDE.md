# CLAUDE.md — src/inference/pringles_test

## Purpose
Pick-and-place the pringles can using pre-recorded ("refined") lookup arm+hand trajectories.
Estimate object 6D pose, transform the lookup trajectories into the current object frame,
and execute via `RobotGUIController` with `pick`/`place` stages.

## Files
- `grasp.py` — main. Loads `lookup/pringles/1/refined_{pick,place}_action.npy` (arm N×4×4)
  and `refined_{pick,place}_hand.npy` (N×16). Reorders Allegro hand qpos thumb-first
  (`[:, :4] = src[:, 12:]`, `[:, 4:] = src[:, :12]`) for both pick and place. Forces hand
  open for the last 70 place steps (`place_hand_traj[-70:] = zeros((70,16))`). Capture ->
  `get_object_6d` (triangulate markers + `SOLVE_XA_B`) -> project mesh to `projected/` ->
  `obj_T = inv(c2r) @ obj_T` -> `normalize_cylinder(obj_T)`. Transforms trajectories:
  `pick_traj = obj_T @ pick_traj`, `place_traj = obj_T @ place_traj`. Launch
  `RobotGUIController(arm, hand, {"pick":{"arm","hand"}, "place":{"arm","hand"}}).run()`.
  Loads local `./marker_offset.npy`.
- `asdf.py` — SCRATCH. Only loads the four refined `.npy` files into vars and stops. No
  transform, no execution. Inspect-the-format snippet.

## paradex modules used
- `remote_camera_controller("object6d")`, `ImageDict`, `SOLVE_XA_B`, `load_current_C2R`,
  `save_current_camparam`, `gui_controller_prev.RobotGUIController`, `get_arm`/`get_hand`.
  (`RobotWrapper`/`get_robot_urdf_path` imported but no IK used — trajectories are pre-baked.)

## Data flow & IO
- In: `lookup/pringles/1/refined_*` (relative to cwd), local `./marker_offset.npy`,
  `rsc/object/pringles/pringles.obj`, cam params + C2R.
- Capture dir: `~/shared_data/inference/grasp_eval/<timestamp>/raw`; writes `projected/`.

## When working here
- Run from repo root so `lookup/pringles/1/...` resolves. Index hardcoded `"1"`, object `pringles`.
- Trajectories are applied as `obj_T @ traj` (object-frame), so they're already full arm
  trajectories, not single grasp poses — no IK step here.

## Gotchas
- `asdf.py` is a non-functional load-only stub.
- `marker_offset.npy` from cwd (local), not `shared_dir`.
- The last-70-steps hand-open is hardcoded to the place trajectory length assumption.
