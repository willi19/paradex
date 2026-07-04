# CLAUDE.md — src/calibration/handeye

## Purpose
Camera-to-robot (C2R) hand-eye calibration for xArm via Tsai-Lenz AX=XB: drive arm through a trajectory observing a Charuco board, combine FK motions with triangulated board motions.

## Files
- `capture.py` — Main PC. Builds `XArmController` from `network_info["xarm"]["param"]`; `remote_camera_controller("handeye_calibration")`. `save_current_camparam(<root>/0)`. Iterates `*_qpos` files from `get_handeye_calib_traj(arm)` (sorted by leading int): `controller.move(action, is_servo=False)`, `rcc.start("image", ...)/stop()`, save `robot.npy`, `eef.npy` (=robot_data["position"]), `qpos.npy` per idx. Only `--arm` (default xarm; raises NotImplementedError otherwise).
- `calculate.py` — Main PC, runs at import (no `if __name__`). `undistort_and_detect_charuco` (ImageDict undistort → triangulate_charuco → `charuco_3d_{ids,corners}.npy` + detection overlay), `compute_fk` (RobotWrapper FK on link6 → `eef_fk.npy`), `compute_motion` (relative eef + camera motions; `SOLVE_XA_B` for cam point fit; floor board id range "1" excluded via `_floor_board_id_range`), `solve_ax_xb(motion_wrt_cam, motion_wrt_robot)` → `robot_wrt_cam`, save `C2R.npy`. `debug` overlays robot mesh + reprojected markers and prints FK/marker stats.

## paradex modules used
- `paradex.calibration.Tsai_Lenz.solve_ax_xb` — AX=XB hand-eye solve.
- `paradex.robot.robot_wrapper.RobotWrapper` + `paradex.robot.utils.get_robot_urdf_path` — FK on `link6`.
- `paradex.transforms.conversion.SOLVE_XA_B` — rigid fit between matched 3D point sets.
- `paradex.image.image_dict.ImageDict`, `paradex.image.aruco` (detect/merge, `find_common_indices`, `boardinfo_dict`).
- `paradex.visualization.robot.RobotModule` — debug mesh render.
- `paradex.calibration.utils` — `handeye_calib_path`, `save_current_camparam`, `load_camparam`, `get_handeye_calib_traj`.

## Data flow & IO
Trajectory `<n>_qpos*` → arm move + capture → `~/shared_data/handeye_calibration/<ts>/<idx>/{images,robot.npy,eef.npy,qpos.npy}`, camparam at `<ts>/0/`. calculate.py adds `eef_fk.npy`, `charuco_3d_*.npy`, `undistort/`, `detection/`, `debug/`, and `C2R.npy` (robot_wrt_cam) in first index dir. Consumed via `load_current_C2R()`.

## When working here
- Requires intrinsic + extrinsic done first (camparam loaded for triangulation/projection). FK uses the (possibly kinematically-calibrated) URDF from `get_robot_urdf_path`.
- `C2R.npy` stores `robot_wrt_cam`; `cam_wrt_robot = inv(C2R)`. Watch direction conventions in `debug`.

## Gotchas
- `capture.py` moves a real robot — run on Main PC with arm powered/clear.
- `calculate.py` has NO `__main__` guard: importing it RUNS the full pipeline. Invoke only as a script.
- Floor/static board id range is excluded (`_floor_board_id_range("1")`) from motion + debug; depends on `boardinfo_dict` ordering.
- eef link is hardcoded `link6`; only xarm is implemented.
