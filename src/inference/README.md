# src/inference

Inference-time pipelines that estimate an object's 6D pose from the multi-camera
rig, look up / generate a dexterous grasp for it, and either visualize or execute
that grasp on the XArm + Allegro hand.

All pipelines share the same backbone:

1. Capture multi-view images of the scene (`remote_camera_controller`).
2. Triangulate ArUco markers and solve the object 6D pose against a per-object
   `marker_offset` table (`SOLVE_XA_B`).
3. Map a grasp (wrist pose + hand joints) from object frame into robot frame using
   the camera-to-robot transform `C2R`, then run IK (`RobotWrapper.solve_ik`).
4. Visualize (Viser / projected mesh) or drive the robot (`RobotGUIController`).

## Subdirectories
| Dir | Purpose |
|-----|---------|
| [`bodex/`](bodex/) | Grasps sourced from a BODex result file (`bodex/scale010_grasp.npy`). Includes a floor-plane probe. |
| [`grasp_eval/`](grasp_eval/) | Grasp evaluation pipelines. `real/` runs on the physical robot using DexGraspNet results. |
| [`grasp_w_gui/`](grasp_w_gui/) | BODex grasp driven through the GUI controller with pregrasp/grasp/squeeze stages. |
| [`pringles_test/`](pringles_test/) | Pringles can pick-and-place using a refined lookup trajectory. |

## Object & grasp sources
- Object mesh: `rsc/object/pringles/pringles.obj` (all current scripts target the `pringles` can).
- Marker offsets: `~/shared_data/object/marker_offset/<obj>/0/marker_offset.npy`, or a
  local `marker_offset.npy` in several scratch variants (see per-dir notes).
- Grasps: DexGraspNet (`dexgraspnet/results/<obj>/<index>/`) or BODex (`bodex/*.npy`).

## Status
This is a research / experimental tree. Several scripts hardcode object name,
grasp index, and relative paths (run from repo root or a specific cwd), and a few
files (`asdf.py`) are incomplete scratch. See each subdirectory's `CLAUDE.md` for
the exact gotchas.

## Related paradex modules
- [`paradex/image/image_dict.py`](../../paradex/image/image_dict.py) — `ImageDict` capture/undistort/triangulate/project
- [`paradex/transforms/conversion.py`](../../paradex/transforms/conversion.py) — `SOLVE_XA_B` rigid fit
- [`paradex/robot/robot_wrapper.py`](../../paradex/robot/robot_wrapper.py) — FK/IK
- [`paradex/io/robot_controller/`](../../paradex/io/robot_controller/) — `get_arm`, `get_hand`, `RobotGUIController`
- [`paradex/io/camera_system/remote_camera_controller.py`](../../paradex/io/camera_system/remote_camera_controller.py)
- [`paradex/calibration/utils.py`](../../paradex/calibration/utils.py) — `load_current_C2R`, `save_current_camparam`
