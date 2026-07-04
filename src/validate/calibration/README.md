# Calibration Validation

Re-evaluates calibration quality on *already-captured* data: how good the xArm
kinematic calibration is (nominal vs calibrated URDF) and how much each camera's
pose drifts in robot space across hand-eye sessions. No new capture is performed.

## Scripts
| File | Purpose |
|------|---------|
| `compare_xarm_kinematic_calib.py` | Re-solves the hand-eye AX=XB problem on one session using the nominal (`xarm.urdf.original`) vs calibrated (`xarm.urdf`) URDF and prints a side-by-side error comparison; optionally renders mesh overlays. |
| `extrinsic_drift.py` | Loads multiple hand-eye sessions, transforms each camera into robot space, and reports per-camera translation/rotation drift vs a reference session. |

## Usage
Runs on the **main PC** (where `~/shared_data/handeye_calibration/` lives). Read-only
on capture data except for writing `C2R_*.npy` and `debug_*/` overlays back into the session.

```bash
# Compare nominal vs calibrated URDF on the latest session (overlay on by default)
python src/validate/calibration/compare_xarm_kinematic_calib.py
python src/validate/calibration/compare_xarm_kinematic_calib.py --name 20251011_194400
python src/validate/calibration/compare_xarm_kinematic_calib.py --no_overlay

# Camera pose drift across sessions
python src/validate/calibration/extrinsic_drift.py                       # all sessions, earliest = ref
python src/validate/calibration/extrinsic_drift.py --sessions 20260318_083843 20260326_062502
python src/validate/calibration/extrinsic_drift.py --ref 20260318_083843
```

Prerequisites for `compare_xarm_kinematic_calib.py`:
- `src/calibration/xarm_kinematic_calibration.py` has run (so `rsc/robot/xarm.urdf.original` exists).
- `src/calibration/handeye/calculate.py` has run on the target session so each index has cached
  `charuco_3d_corners.npy` / `charuco_3d_ids.npy` (and `undistort/` for overlays).

## What it validates
- `compare_xarm_kinematic_calib.py`: lower is better. Watch the SUMMARY table —
  the calibrated URDF should reduce **AX-XB residual (mm)**, **FK vs robot TCP (mm)**,
  and **marker std |xyz| (mm)** vs nominal. The "Cam motion fit" line is a
  URDF-independent sanity check on the charuco data itself. Overlays show predicted
  markers (red) vs observed (blue) and a projected robot mesh.
- `extrinsic_drift.py`: small per-camera `dt (mm)` / `dR (deg)` across sessions means
  the rig is stable; large drift flags a moved/bumped camera. Look at the per-camera
  and ALL summary rows and the consecutive-session drift table.

## Related
- [`paradex/calibration/Tsai_Lenz.py`](../../../paradex/calibration/Tsai_Lenz.py) — `solve_ax_xb`
- [`paradex/calibration/utils.py`](../../../paradex/calibration/utils.py) — `handeye_calib_path`, `load_camparam`
- [`paradex/robot/robot_wrapper.py`](../../../paradex/robot/robot_wrapper.py) — FK
- [`paradex/visualization/robot.py`](../../../paradex/visualization/robot.py) — `RobotModule` overlay
- `src/calibration/handeye/calculate.py` — produces the cached session data consumed here
