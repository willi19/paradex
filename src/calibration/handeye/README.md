# Hand-Eye Calibration

Computes the camera-to-robot transform (C2R) for an xArm. The arm is driven through a fixed trajectory while observing a Charuco board; per-pose forward kinematics and triangulated board poses feed a Tsai-Lenz AX=XB solver.

## Scripts
| File | Purpose |
|------|---------|
| `capture.py` | **Main PC**. Moves the arm through the hand-eye trajectory, capturing one multi-camera image set + robot state (eef pose, qpos) per waypoint. Saves the current camera params alongside. |
| `calculate.py` | **Main PC** offline solver. Undistorts/detects+triangulates Charuco per pose, computes FK, builds relative motions, solves Tsai-Lenz (`solve_ax_xb`), and writes `C2R.npy`. Also renders debug overlays of the robot mesh + reprojected markers. |

## Usage
Prerequisites: valid intrinsic **and** extrinsic calibration must exist (extrinsic provides the camera params loaded for triangulation/render), and a hand-eye trajectory must exist for the arm.

1. On the **Main PC**, capture (drives the live xArm):
   ```bash
   python src/calibration/handeye/capture.py --arm xarm
   ```
   No keyboard interaction — it auto-steps through every `*_qpos` trajectory file, moving the arm and capturing at each waypoint.
2. On the **Main PC**, solve:
   ```bash
   python src/calibration/handeye/calculate.py --arm xarm            # latest handeye dir
   python src/calibration/handeye/calculate.py --name 20250101_120000 --arm xarm
   ```
   Prints FK errors and AX=XB residuals (mm), writes `C2R.npy`, and saves per-pose debug images.

## Inputs & Outputs
- Capture writes `~/shared_data/handeye_calibration/<timestamp>/<idx>/`: `images/`, `robot.npy`, `eef.npy` (eef pose), `qpos.npy`; camera params saved to `<timestamp>/0/`.
- Trajectory read from `get_handeye_calib_traj(arm)` — files named `<n>_qpos*`.
- `calculate.py` adds per-pose `undistort/`, `detection/`, `debug/`, `charuco_3d_{ids,corners}.npy`, `eef_fk.npy`, and writes `C2R.npy` (robot_wrt_cam) into the first index dir.
- Result consumed elsewhere via `paradex.calibration.utils.load_current_C2R()`.

## Related
- `paradex/calibration/Tsai_Lenz.py` — `solve_ax_xb`, `solve_axb_cpu`.
- `paradex/robot/robot_wrapper.py` (`RobotWrapper.compute_forward_kinematics`), `paradex/robot/utils.get_robot_urdf_path`.
- `paradex/transforms/conversion.py` — `SOLVE_XA_B` (rigid fit between point sets).
- `paradex/image/{image_dict,aruco}.py`, `paradex/visualization/robot.RobotModule`.
- `paradex.calibration.utils` — `handeye_calib_path`, `save_current_camparam`, `load_camparam`, `get_handeye_calib_traj`.
- Prerequisite app: [`../extrinsic/`](../extrinsic/). URDF source for FK: [`../xarm_kinematic_calibration.py`](../xarm_kinematic_calibration.py).
