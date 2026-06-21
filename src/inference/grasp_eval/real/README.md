# grasp_eval / real

Run grasp evaluation on the **physical** XArm + Allegro robot. Estimates the pringles
can's 6D pose from the camera rig, maps a DexGraspNet grasp into robot frame, runs IK,
and drives the robot through a `RobotGUIController`.

## Scripts
| File | Purpose |
|------|---------|
| `object6d.py` | Main: capture -> solve object 6D pose -> load DexGraspNet grasp -> IK wrist -> launch GUI with `start`/`grasp` stages on the real robot. |
| `visualize.py` | Dry run: same pose estimation + grasp, render robot + object in Viser instead of executing. |
| `grasp_test.py` | Minimal GUI smoke test: spins up `RobotGUIController` on just the XArm. |
| `asdf.py` | **Scratch / broken** — incomplete pick-and-place loop with undefined helpers; not runnable. |

## Usage
Run from repo root (needs `dexgraspnet/results/...` relative path).

```bash
# Preview the grasp in Viser first (no robot motion)
python src/inference/grasp_eval/real/visualize.py

# Execute on the real robot via GUI (start/grasp buttons)
python src/inference/grasp_eval/real/object6d.py

# Just test the arm GUI controller
python src/inference/grasp_eval/real/grasp_test.py
```

`object6d.py` opens a `RobotGUIController` GUI with named poses `start` (hand open) and
`grasp` (loaded qpos), with the arm `grasp` target = FK of the IK solution. Use the GUI
buttons to move the robot. `visualize.py` opens a Viser web viewer (open the printed URL).

## Inputs & Outputs
- **Reads:** marker offsets `~/shared_data/object/marker_offset/pringles/0/marker_offset.npy`;
  object mesh `rsc/object/pringles/pringles.obj`; grasp
  `dexgraspnet/results/pringles/<index>/{qpos.npy, wrist_6d.npy}` (index hardcoded to 6);
  current cam params + `C2R`.
- **Captures:** images to `~/shared_data/inference/grasp_eval/<timestamp>/raw`, undistorted in place.
- **Writes:** `projected/` overlay images. (`object6d.py` does not save `obj_T.npy`.)

## Related
- Sibling grasp sources: [`../../bodex/`](../../bodex/), [`../../grasp_w_gui/`](../../grasp_w_gui/).
- [`paradex/io/robot_controller/`](../../../../paradex/io/robot_controller/) — `RobotGUIController`, `get_arm`, `get_hand`.
- [`paradex/robot/robot_wrapper.py`](../../../../paradex/robot/robot_wrapper.py) — IK / FK.
