# CLAUDE.md — src/validate/calibration

## Purpose
Offline calibration-quality checks on captured hand-eye sessions. No camera/robot
hardware involved; reads `~/shared_data/handeye_calibration/<session>/<index>/`.

## Files
- `compare_xarm_kinematic_calib.py` — main entry `main()`:
  1. Load session (`handeye_calib_path` + `--name`/`find_latest_directory`), assert
     `charuco_3d_corners.npy` cached for all indices.
  2. `_compute_motion_wrt_cam` — charuco-to-charuco motion via `SOLVE_XA_B` (URDF-independent).
  3. `_eval_variant` per URDF: `RobotWrapper.compute_forward_kinematics` (link `link6`) →
     `motion_wrt_robot` → `solve_ax_xb` → metrics (AX-XB residual, FK vs `eef.npy`, marker std).
     Saves `C2R_<tag>.npy` into index `0`.
  4. `_render_overlay` (default on) — `RobotModule` + `ImageDict.from_path("undistort")` +
     `load_camparam`; projects robot mesh and red(pred)/blue(observed) markers, saves `debug_<tag>/`.
  5. `_print_side_by_side` — nominal vs calibrated table.
- `extrinsic_drift.py` — `compare_camera_poses()`: per session loads `C2R.npy` +
  `cam_param/extrinsics.json`, computes `T_cam_in_robot = inv(C2R) @ inv(extrinsic)`,
  `se3_diff` against a reference session. Pure print report.

## paradex modules used
- `paradex.calibration.Tsai_Lenz.solve_ax_xb`
- `paradex.calibration.utils` — `handeye_calib_path`, `load_camparam`
- `paradex.image.aruco.find_common_indices`, `paradex.image.image_dict.ImageDict`
- `paradex.robot.robot_wrapper.RobotWrapper`
- `paradex.transforms.conversion.SOLVE_XA_B`
- `paradex.utils.file_io.find_latest_directory`, `paradex.utils.path.shared_dir`
- `paradex.visualization.robot.RobotModule`

## Data flow & IO
- IN: `<session>/<idx>/` → `charuco_3d_{corners,ids}.npy`, `qpos.npy`, `eef.npy`,
  `undistort/`, `cam_param/extrinsics.json`, `C2R.npy`.
- OUT (compare script only): `<idx 0>/C2R_{nominal,calibrated}.npy`, `<idx>/debug_<tag>/`.

## When working here
- `EEF_LINK = "link6"` and URDF paths are resolved relative to repo root.
- `extrinsic_drift.py` appends `parents[3]` to sys.path so it can run as a bare file.

## Gotchas
- Convention note in `extrinsic_drift.py`: `C2R.npy` here is robot→COLMAP-world (it inverts it).
  `compare_xarm_kinematic_calib.py` instead saves `solve_ax_xb` output (`robot_wrt_cam`) directly
  as `C2R_<tag>.npy` — different meaning from the `C2R.npy` that `extrinsic_drift` reads. Don't conflate.
- FK-vs-TCP metric is partially circular (uses robot's own `eef.npy`); the AX-XB residual and
  marker-std metrics are the trustworthy ones.
