# Robot Hardware Validation

Smoke-test scripts that exercise the physical robot hands (Allegro, Inspire) and the XArm arm, plus a tactile/force viewer and a URDF-overlay debug tool.

## Scripts
| File | Purpose |
|------|---------|
| `allegro.py` | Sweeps each of the 16 Allegro hand joints one at a time (flex out then back) to verify per-joint motion and joint ordering. |
| `inspire.py` | Live tactile heatmap viewer for the Inspire hand over USB Modbus (`InspireController`); 5x4 matplotlib grid of all sensor regions. |
| `inspire_left.py` | Scripted pose sequence for the Inspire LEFT hand over Modbus TCP/IP: HOME → OPEN → CLOSED → OPEN → per-finger sinusoidal wave → HOME, printing commanded vs. reported angles. |
| `inspire_left_gui.py` | Interactive matplotlib GUI for the Inspire LEFT hand: 6 DOF sliders, force bar plot, tactile heatmap grid, HOME/OPEN/CLOSED presets, and a force-calibration button. |
| `inspire_left_overlay.py` | URDF-tuning debug tool: captures a multi-camera snapshot + arm/hand qpos, then renders the inspire-left URDF mesh alpha-overlaid on the images with OpenCV trackbars for the 4 thumb joints and wrist xyz offset. |
| `xarm_base_wiggle.py` | Smoothly wiggles XArm joint1 ±60° (sine) about the current pose while holding all other joints, to validate base-joint servo motion. |

## Usage
Each script is run directly. Hardware required is noted per script.

```bash
# Allegro hand (ROS2/rclpy) — sweeps 16 joints
python src/validate/robot/allegro.py

# Inspire hand tactile viewer (USB Modbus) — opens matplotlib window
python src/validate/robot/inspire.py

# Inspire LEFT hand (Modbus TCP/IP) — scripted pose sequence
python src/validate/robot/inspire_left.py

# Inspire LEFT hand interactive GUI (Modbus TCP/IP, with tactile)
python src/validate/robot/inspire_left_gui.py

# XArm base joint wiggle (XArm SDK)
python src/validate/robot/xarm_base_wiggle.py

# Inspire LEFT URDF overlay (multi-camera capture + XArm + Inspire LEFT)
python src/validate/robot/inspire_left_overlay.py                 # fresh capture
python src/validate/robot/inspire_left_overlay.py --load <abs_dir>  # reuse a snapshot
```

### Hardware
- `allegro.py` — Allegro hand via ROS2 (`get_hand("allegro")`).
- `inspire.py` — Inspire hand via USB Modbus (`InspireController`, `network_info['inspire_usb']`).
- `inspire_left.py`, `inspire_left_gui.py` — Inspire LEFT hand via Modbus TCP/IP socket (`InspireControllerIP`, `network_info['inspire']`).
- `xarm_base_wiggle.py` — XArm arm via SDK (`get_arm("xarm")`).
- `inspire_left_overlay.py` — multi-camera rig (remote capture) **+** XArm **+** Inspire LEFT; requires current camera calibration and hand-eye (`C2R`).

### GUI / keyboard controls
- `inspire_left_gui.py`: drag the 6 sliders to command DOFs; click HOME/OPEN/CLOSED presets; click "CALIBRATE FORCE" (opens hand, zeros force sensors — keep hand untouched); close window or Ctrl+C to exit.
- `inspire_left_overlay.py`: `q` quit, `s` save trackbar state to `<save_dir>/tuned.json`, `r` reset trackbars to captured values, `c` re-capture a fresh snapshot.

## What it validates
- `allegro.py` — each joint visibly flexes and returns; confirms joint indexing and ROS2 command path.
- `inspire.py` — pressing each sensor region lights up the corresponding heatmap cell; confirms tactile data path.
- `inspire_left.py` — hand reaches each pose; printed `actual` angles track the `target` commands within tolerance.
- `inspire_left_gui.py` — slider moves track on the hand; force bars and tactile heatmaps respond to contact; viz loop runs at roughly 20 Hz.
- `xarm_base_wiggle.py` — base joint oscillates smoothly with no drift on other joints; returns to home on exit.
- `inspire_left_overlay.py` — rendered hand mesh should align with the real hand in the images; used to tune thumb joints and wrist mount offset.

## Related
- [`paradex/io/robot_controller/`](../../../paradex/io/robot_controller) — `get_arm`, `get_hand`, `XArmController`, `InspireController`, `InspireControllerIP`.
- [`paradex/visualization/robot.py`](../../../paradex/visualization/robot.py) — `RobotModule` (URDF rendering used by the overlay tool).
- [`paradex/image/image_dict.py`](../../../paradex/image/image_dict.py) — `ImageDict` (undistort, `project_mesh`, merge grid).
- [`paradex/calibration/utils.py`](../../../paradex/calibration/utils.py) — camparam / `C2R` load+save helpers.
- [`paradex/io/camera_system/remote_camera_controller.py`](../../../paradex/io/camera_system/remote_camera_controller.py) — remote multi-camera capture.
