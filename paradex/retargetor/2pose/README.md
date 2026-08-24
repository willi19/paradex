# Allegro direct-anchor v2 / two-pose experiment

This directory preserves the direct-anchor v2 Allegro retargeter as it was
immediately before commit `21fe2f9a` (the VIVE-frame stabilization change).
It has its own immutable anchor JSON, so later edits to the default
`allegro_v5` calibration do not change this snapshot.

`allegro_v5_direct_anchor_v2.py` exports the historical `allegro_v5(frame,
ergonomics=None)` function.  It retains the original transform-derived raw
tie-breaker and is intended for comparison experiments, not the default VIVE
teleoperation path.

The pedal tools interpolate between two editable Allegro target poses without
MANUS hand teleoperation:

```bash
python paradex/retargetor/2pose/allegro_pedal_pose_ui.py --simulate
python paradex/retargetor/2pose/capture_allegro_pedal.py --name pedal_test
```

Use `--help` on either command for robot, xArm/VIVE, tactile, and endpoint
options.  The left and right outer pedals increase and decrease the blend
parameter; the middle pedal gates xArm teleoperation when `--with-xarm` is
enabled.
