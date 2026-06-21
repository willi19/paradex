# Camera & Robot Calibration

Application scripts for the Paradex calibration pipeline. Calibration runs in a fixed order: **intrinsic → extrinsic → handeye**. This top-level directory also holds the xArm kinematic-calibration tool that patches the robot URDFs from the controller's factory parameters.

## Scripts
| File | Purpose |
|------|---------|
| `xarm_kinematic_calibration.py` | Reads factory kinematic calibration from a live xArm controller, saves it as YAML, and patches the paradex xArm URDFs in place (with `.original` backups). |

## Subdirectories
| Dir | Purpose |
|-----|---------|
| [`extrinsic/`](extrinsic/) | Multi-camera extrinsic calibration via Charuco + COLMAP (capture on Main PC, detection on Capture PCs, solve on Main PC). |
| [`handeye/`](handeye/) | Camera-to-robot hand-eye calibration (Tsai-Lenz) by moving the arm through a trajectory and observing a Charuco board. |

## Usage
Patch URDFs from a connected xArm (run on the Main PC, robot reachable on the network):
```bash
python src/calibration/xarm_kinematic_calibration.py --robot_ip 192.168.1.xxx
# default IP comes from system/current/network.json
python src/calibration/xarm_kinematic_calibration.py --no_apply   # only write YAML, don't patch
```
Flags: `--suffix` (YAML filename suffix, default today's date), `--urdf_paths` (override the default URDF list), `--no_apply`.

Full calibration order:
1. Intrinsic calibration (see `src/calibration/` siblings / `paradex/calibration`) — produces per-camera intrinsics.
2. Extrinsic — [`extrinsic/`](extrinsic/).
3. Hand-eye — [`handeye/`](handeye/).

## Inputs & Outputs
- Reads: live xArm controller over IP; default URDFs `rsc/robot/xarm.urdf`, `xarm_allegro.urdf`, `xarm_inspire.urdf`.
- Writes: `~/shared_data/xarm_kinematics/<robot_name>_kinematics_<suffix>.yaml`; patches the URDFs in place (originals saved as `*.urdf.original`).
- Rollback: `cp rsc/robot/xarm.urdf.original rsc/robot/xarm.urdf`.

## Related
- `paradex/robot/xarm_kinematic_calib.py` — `read_xarm_kinematic_params`, `save_kinematic_yaml`, `apply_kinematics_to_urdf`.
- `paradex/utils/system.py` — `network_info` (default robot IP).
- Sibling apps: [`extrinsic/`](extrinsic/), [`handeye/`](handeye/).
