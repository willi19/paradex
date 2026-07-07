# CLAUDE.md - Paradex Project Guide

## What is Paradex?
Multi-camera vision + robot control framework for dexterous manipulation research.
- **`paradex/`**: Core reusable library (installed via `pip install -e .`)
- **`src/`**: Application scripts that combine paradex modules
- **`system/`**: System configs (camera, network, PC info) — active config in `system/current/`
- **`rsc/`**: Robot URDFs, object meshes, hand models

## Language & Runtime
- Python 3.8+, package name `paradex`
- No test suite — validate via `src/validate/` scripts

## Key Dependencies
- **Vision**: OpenCV (`cv2`), PyColmap, Open3D, trimesh
- **Robotics**: Pinocchio (FK/IK), CuRobo (GPU motion planning), yourdfpy
- **ROS2**: rclpy (Allegro hand control via `/joint_states`, `/allegro_hand_position_controller/commands`)
- **Simulation**: IsaacGym (PhysX)
- **Visualization**: Viser (web 3D viewer), Open3D
- **Math**: numpy, scipy, transforms3d, torch
- **Data**: JSON, YAML, pickle, numpy `.npy`

## Architecture

### Distributed Multi-PC System
- 6 capture PCs + 1 main PC
- Capture PCs run camera daemons, main PC orchestrates
- Communication: SSH, rsync, TCP/UDP
- Hardware sync via UTGE900 signal generator

### Core Modules (`paradex/`)
| Module | Purpose |
|--------|---------|
| `calibration/` | Camera calibration (COLMAP, hand-eye via Tsai-Lenz) |
| `image/` | ArUco detection, undistortion, projection, image merging |
| `io/camera_system/` | Camera control (Flir Blackfly via PySpin), distributed capture |
| `io/robot_controller/` | Robot arm (XArm) + hand (Allegro/Inspire) control |
| `io/capture_pc/` | SSH commands, rsync transfers, remote capture |
| `io/teleop/` | XSens motion capture input |
| `robot/` | Pinocchio FK/IK wrapper, CuRobo planning, URDF utils |
| `simulator/` | IsaacGym physics simulation |
| `transforms/` | Coordinate transforms, rotation conversions, triangulation |
| `retargetor/` | Hand pose retargeting from human demos |
| `dataset_acqusition/` | Capture session orchestration, frame sync |
| `video/` | Video processing (undistort, frame drop correction) |
| `visualization/` | Viser/Open3D 3D viewers, robot rendering, visual hull — see `docs/visualization.md` |
| `utils/` | Config loading, paths, file I/O, keyboard listener |

### Application Scripts (`src/`)
| Directory | Purpose |
|-----------|---------|
| `calibration/` | Intrinsic, extrinsic, hand-eye calibration workflows |
| `capture/` | Camera image/video capture (main + capture PC scripts) |
| `dataset_acquisition/` | Dataset-specific capture pipelines |
| `inference/` | Model inference (6D pose, grasp evaluation) |
| `object6d/` | 6D object pose estimation pipeline |
| `process/` | Data post-processing |
| `validate/` | System validation (camera sync, robot, teleop) |
| `util/` | ArUco markers, object registration, video upload |

## Common Patterns

### Config Loading
```python
from paradex.utils.system import config_dir, pc_info, network_info
from paradex.calibration.utils import load_current_camparam, load_current_C2R
from paradex.utils.file_io import load_yaml
```

### Camera Access
**On the main PC use `remote_camera_controller` (rcc)** — it drives the capture-PC
daemons. Do NOT use `CameraLoader`/`Camera` directly on the main PC (that's the
daemon's job). Full recipe / modes / error handling / gotchas:
[`agent_docs/camera_system/usage.md`](agent_docs/camera_system/usage.md)
(editing camera internals? see [`agent_docs/camera_system/internals.md`](agent_docs/camera_system/internals.md)).
```python
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
rcc = remote_camera_controller("my_app")                 # needs server_daemon.py up on each capture PC
rcc.start("image", False, save_path="dataset/001/raw")   # image | video | stream | full
rcc.stop(); rcc.end()                                    # end() releases the lock
# health while capturing: rcc.get_status() -> {'error', 'stalled', 'pc': {...}}
```

### Multi-Camera Images = Dict[serial_num: str, image: np.ndarray]
```python
image_dict = {"25305460": img1, "25305462": img2}
```

### Robot Control
```python
from paradex.io.robot_controller import get_arm, get_hand
arm = get_arm("xarm")
hand = get_hand("allegro")

from paradex.robot.robot_wrapper import RobotWrapper
robot = RobotWrapper("path/to/urdf")
fk = robot.compute_forward_kinematics(qpos, link_list=["tool"])
```

### Data Capture Session
```python
from paradex.dataset_acqusition.capture import CaptureSession
capture = CaptureSession(camera=True, arm="xarm", hand="allegro", teleop="xsens")
```

## Data Paths
```
~/shared_data/
├── cam_param/              # Calibration params (intrinsic, extrinsic)
├── handeye_calibration/    # Camera-to-robot transforms
├── [dataset-name]/         # Captured data (images/, videos/, arm/, hand/)
```

## Module API Reference Docs
Detailed API references are stored in Claude's memory directory (`~/.claude/projects/.../memory/`):
- `visualization.md` — ViserViewer, RobotModule, Open3DVideoRenderer, skeleton visualizers

## When doing X, look at Y (task map)

Every `src/<app>/` leaf dir has a `CLAUDE.md` (orientation for you) + `README.md` (run steps for humans). Use this map to jump to the right one.

| If the task is... | Start here | Notes |
|-------------------|-----------|-------|
| Calibrate camera intrinsics (K + distortion) | [`src/calibration/intrinsic/`](src/calibration/intrinsic/CLAUDE.md) | Charuco, distributed, **auto-capture (no save button)**. `capture.py` (main PC) → `calculate.py`. Run first. |
| Calibrate cameras (extrinsic) | [`src/calibration/extrinsic/`](src/calibration/extrinsic/CLAUDE.md) | Charuco + COLMAP; needs intrinsic first. `capture.py` (main PC) → `calculate.py`. |
| Calibrate camera→robot (hand-eye) | [`src/calibration/handeye/`](src/calibration/handeye/CLAUDE.md) | Tsai-Lenz AX=XB; needs extrinsic first. Produces `C2R.npy`. |
| Calibrate xArm kinematics | [`src/calibration/`](src/calibration/CLAUDE.md) | `xarm_kinematic_calibration.py` patches URDFs from live arm. |
| Capture images/video/stream | [`src/capture/camera/`](src/capture/camera/CLAUDE.md) | Main-PC `*_remote.py` drives capture-PC daemons in [`src/camera/`](src/camera/CLAUDE.md). |
| Run camera daemons on capture PCs | [`src/camera/`](src/camera/CLAUDE.md) | `server_daemon.py` must be up before any orchestrator connects. |
| Capture robot demos (teleop/teaching) | [`src/capture/robot/`](src/capture/robot/CLAUDE.md) | XSens teleop or manual xArm teaching. |
| Build a labeled dataset | [`src/dataset_acquisition/`](src/dataset_acquisition/CLAUDE.md) | Pick the pipeline (graphics/motion_blur/hri/miyungpa/object_turntable); built on `CaptureSession`. |
| Post-process captured data | [`src/process/`](src/process/CLAUDE.md) | `miyungpa` (sync+overlay+contact), `object_turntable` (COLMAP). Some files have known bugs — see their docs. |
| Estimate 6D object pose | [`src/object6d/`](src/object6d/CLAUDE.md) | LoFTR/template matching; needs external `_object_6d_tracking`. |
| Run grasp inference/eval | [`src/inference/`](src/inference/CLAUDE.md) | 6D pose → grasp → IK → execute. Watch for scratch `asdf.py` files. |
| Validate a subsystem | [`src/validate/`](src/validate/CLAUDE.md) | Smoke tests for cameras/sync/robots/teleop; mirror `paradex.io`. |
| Make charuco boards | [`src/util/marker/`](src/util/marker/CLAUDE.md) | Printable A4 PDFs. |
| Register an object's markers | [`src/util/register_object/`](src/util/register_object/CLAUDE.md) | Produces `marker_offset` table for 6D pose. |
| Merge/replay/visualize a robot URDF | [`src/util/robot/`](src/util/robot/CLAUDE.md) | URDF merge, Viser replay. |
| Upload/undistort raw video | [`src/util/upload_video/`](src/util/upload_video/CLAUDE.md) | Distributed; dashboard on main PC. |

Full application index: [`src/README.md`](src/README.md) / [`src/CLAUDE.md`](src/CLAUDE.md).
Generated API reference (Sphinx): [`docs/index.html`](docs/index.html). Project landing page: [`index.html`](index.html).

## Important Notes
- Calibration order: intrinsic -> extrinsic -> handeye_calibration
- Camera serial numbers are string keys everywhere
- `system/current/` is a symlink to the active system config (paradex1, paradex2, etc.)
- Typo in source: `dataset_acqusition` (missing 'i') — this is the actual directory/module name, do NOT "fix" it
- Robot hands: Allegro (ROS2/rclpy), Inspire (direct IP socket)
- Robot arms: XArm (SDK), Franka (Pinocchio)
- All color params in visualization are **0.0-1.0 float RGB** (or RGBA with alpha)
- Quaternion convention in viser: **wxyz** order. Convert from scipy: `quat_xyzw[[3,0,1,2]]`
