# grasp_w_gui

Pringles grasp inference using **BODex** grasps, executed through the GUI controller with
distinct **pregrasp / grasp / squeeze** hand stages. Estimates the object 6D pose, maps the
BODex wrist+hand grasp into robot frame, runs IK, and launches a `RobotGUIController` to
drive the real XArm + Allegro.

## Scripts
| File | Purpose |
|------|---------|
| `object6d.py` | Main: capture -> object 6D pose -> load BODex grasp -> IK wrist -> launch GUI with `start`/`pregrasp`/`grasp`/`squeezed` hand poses. |
| `visualize.py` | Dry run: same pose estimation + BODex grasp, rendered in Viser (no robot motion). |

## Usage
Run from repo root (relative paths `bodex/scale010_grasp.npy`, `marker_offset.npy`).

```bash
# Preview in Viser
python src/inference/grasp_w_gui/visualize.py

# Execute via GUI (start / pregrasp / grasp / squeezed buttons)
python src/inference/grasp_w_gui/object6d.py
```

In `object6d.py` the GUI exposes hand stages: `start` (open), `pregrasp` (BODex frame 0),
`grasp` (BODex frame 1), and `squeezed` (extrapolated `grasp*8 - pregrasp*7` for a tighter
close). The arm `grasp` target is the FK of the IK solution.

## Inputs & Outputs
- **Reads:** local `marker_offset.npy`; `rsc/object/pringles/pringles.obj`;
  BODex grasps `bodex/scale010_grasp.npy` (`data['robot_pose'][0, index]`, index hardcoded to 1);
  current cam params + `C2R`.
- **Captures:** images to `~/shared_data/inference/grasp_eval/<timestamp>/raw`, undistorted in place.
- **Writes:** `projected/` overlay images (`object6d.py`).

## Related
- Same BODex grasp source as [`../bodex/`](../bodex/); `grasp_w_gui` adds the multi-stage GUI execution.
- [`paradex/io/robot_controller/`](../../../paradex/io/robot_controller/), [`paradex/robot/robot_wrapper.py`](../../../paradex/robot/robot_wrapper.py).
