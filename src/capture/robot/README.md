# Robot Capture

Scripts for collecting robot-arm/hand data: teleoperated demonstration
recording and manual waypoint teaching on the XArm.

## Scripts
| File | Purpose |
|------|---------|
| `teleop_real.py` | Record teleoperation sessions on the real robot via XSens/Oculus, optionally with arm and/or hand. Wraps `CaptureSession`. |
| `xarm_teaching.py` | Hand-guide ("manual mode") the XArm and save waypoint poses on a keypress — for building teaching trajectories. |

## Usage

### Teleoperation recording (`teleop_real.py`)
Runs on the **robot/main PC** connected to the teleop device and robot.
```bash
python src/capture/robot/teleop_real.py --device {xsens|occulus} \
    --arm <arm_name> --hand <hand_name> --save_path <dataset_root>
```
Flow per session (driven by teleop device gestures, with audio cues via `chime`):
1. Pre-record teleop loop — move the robot freely to get ready.
2. Gesture transitions out of the loop (`stop` → begin recording; `exit` → quit).
3. Recording starts to `<save_path>/<timestamp>`; another gesture stops it.
Repeats until an `exit` gesture. `--arm`/`--hand` may be omitted to record only
the available devices. Camera is disabled (`camera=False`).

### XArm waypoint teaching (`xarm_teaching.py`)
Runs on the **PC connected to the XArm**. Puts the arm in manual (gravity-comp)
mode so you physically move it.
```bash
python src/capture/robot/xarm_teaching.py --save_path <dir>
```
Keys: `c` = save current pose, `q` = quit. Each `c` writes the joint angles and
the wrist transform for the current arm pose.

## Inputs & Outputs
- `teleop_real.py`: reads teleop device + robot state; `CaptureSession` writes
  arm/hand/state recordings under `<save_path>/<timestamp>/`.
- `xarm_teaching.py`: reads XArm at `network_info["xarm"]["param"]["ip"]`; writes
  to `--save_path`:
  - `<idx>_qpos.npy` — 6-DOF joint angles (radians).
  - `<idx>_aa.npy` — 4x4 wrist pose matrix (from axis-angle position via `aa2mtx`).

## Related
- [`paradex/dataset_acqusition/capture.py`](../../../paradex/dataset_acqusition/capture.py) — `CaptureSession` (`teleop()`, `start()`, `stop()`, `end()`).
- [`paradex/io/robot_controller`](../../../paradex/io/robot_controller) — arm/hand drivers.
- [`paradex/io/teleop`](../../../paradex/io/teleop) — XSens/Oculus input.
- [`paradex/transforms/conversion.py`](../../../paradex/transforms/conversion.py) — `aa2mtx`.
- Sibling: camera capture in [`../camera`](../camera).
