# CLAUDE.md — src/calibration/handeye

## Purpose
Camera-to-robot (C2R) hand-eye calibration via Tsai-Lenz AX=XB: drive arm through a trajectory observing a Charuco board, combine FK motions with triangulated board motions. Supports **xarm and franka** (`--arm`).

## Teaching the trajectory first
`capture.py` replays `*_qpos*` waypoints from `get_handeye_calib_traj(arm)` = `system/current/hecalib/<arm>/`. Record them by hand-guiding:
- xarm → `python src/capture/robot/xarm_teaching.py --save_path system/current/hecalib/xarm`
- franka → `python src/capture/robot/franka_teaching.py --save_path system/current/hecalib/franka --host 127.0.0.1`
  (**franka_daemon must be running first**: `./cpp/franka_daemon/run_daemon.sh` — see [`docs/franka.md`](../../../docs/franka.md))
  - The script only **reads** — pose the arm however you like (Desk jogging is easiest).
  - **Set the Desk mode before starting the daemon, and don't switch while it runs.** A
    mode switch kills the daemon's libfranka session: `ping` still answers but the state
    stream dies (`[STREAM] Error: Net Exception` in its log) and `c` saves nothing.
    Restart the daemon after any mode change. `capture.py` needs Execution mode.
  - `setGuidingMode` does **not** free the arm here (returns "success", arm stays locked
    — needs the X4 External Enabling Device), which is why the script no longer calls it.

Press `c` per pose, `q` to finish. The existing xarm set has 18 poses.

## Files
- `capture.py` — Main PC. Builds `XArmController` from `network_info["xarm"]["param"]`, or `FrankaController(network_info["franka"])` with a `ping()` precheck; `remote_camera_controller("handeye_calibration")`. `save_current_camparam(<root>/0)`. Iterates `*_qpos` files from `get_handeye_calib_traj(arm)` (sorted by leading int): `controller.move(action, is_servo=False)`, `rcc.start("image", ...)/stop()`, save `robot.npy`, `eef.npy` (=robot_data["position"]), `qpos.npy` per idx. `--arm xarm|franka`.
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
- eef link comes from `EEF_LINK` in `calculate.py`: xarm → `link6`, franka → `fr3_link8` (FR3's flange; `link6` does not exist in `franka.urdf`). Add an entry when supporting a new arm.
- franka's `network_info["franka"]` is a **plain IP string**, not a dict like xarm's — `network_info[arm]["name"]` breaks on it.
- `FrankaController.end()` takes no args (xarm's takes a bool).
