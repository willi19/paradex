# Allegro direct-anchor v2 / two-pose experiment

This directory preserves the direct-anchor v2 Allegro retargeter as it was
immediately before commit `21fe2f9a` (the VIVE-frame stabilization change).
It has its own immutable anchor JSON, so later edits to the default
`allegro_v5` calibration do not change this snapshot.

`allegro_v5_direct_anchor_v2.py` exports the historical `allegro_v5(frame,
ergonomics=None)` function.  It retains the original transform-derived raw
tie-breaker and is intended for comparison experiments, not the default VIVE
teleoperation path.

The two-pose tools interpolate between two editable Allegro target poses
without MANUS hand teleoperation. Quest Touch is the default input: right
controller A moves toward pose B (grasp), while B moves back toward pose A
(open). They launch the neighbouring `quest-openxr` project's
`scripts/run-input.sh` wrapper in its focused streaming mode so the Touch
interaction profile remains active. Use `--quest-openxr-bin` if that launcher
is located elsewhere. `--quest-headless` is available for runtimes where a
headless session is known to retain controller focus.

```bash
python paradex/retargetor/2pose/allegro_pedal_pose_ui.py --simulate
python paradex/retargetor/2pose/capture_allegro_pedal.py --name quest_test
```

VIVE pose xArm teleoperation defaults to the Quest controller deadman and does
not require a pedal:

```bash
PYTHONPATH=. python paradex/retargetor/2pose/allegro_pedal_pose_ui.py \
  --with-xarm
```

Right VIVE supplies the xArm pose. Hold right Quest Grip to teleoperate; release
it to hold the arm. Quest A and B continue to grasp and open the Allegro hand.
Stale Grip input fails closed. Grip is exposed to `CaptureSession` as the same
state `0` (active) / state `1` (hold) transition used by the original left-hand
control input, so releasing it calls the standard retargeter stop path.

The capture command uses the same default gating:

```bash
PYTHONPATH=. python paradex/retargetor/2pose/capture_allegro_pedal.py \
  --name quest_vive_test
```

Use `--input-source pedal` to retain the Stream Deck workflow and use no Quest
controller input: left and right outer pedals move toward grasp and open, while
the middle pedal is the VIVE xArm deadman.

```bash
PYTHONPATH=. python paradex/retargetor/2pose/allegro_pedal_pose_ui.py \
  --with-xarm --input-source pedal
```

`--input-source slider` keeps manual UI control, but cannot be combined with
`--with-xarm` because it has no deadman. `--input-rate` controls the maximum
parameter speed; `--pedal-rate` remains a backward-compatible alias. The Quest
bridge requests 100 Hz analog samples and the standalone Allegro command loop
also runs on a fixed 100 Hz deadline to minimize added control-path latency.
Viser updates are capped at 20 Hz so browser traffic does not make physical
hand targets uneven. The Allegro publisher additionally slew-limits each joint
slightly above the endpoint interpolation velocity, smoothing missed command
ticks without slowing normal interpolation.

Quest pose xArm teleoperation is available without VIVE:

```bash
PYTHONPATH=. python paradex/retargetor/2pose/allegro_pedal_pose_ui.py \
  --with-quest-xarm
```

The Quest display uses WiVRn's alpha-blend camera passthrough by default instead
of the diagnostic color panels, while the desktop input monitor remains
available.

Hold right Grip to start moving the xArm and release Grip to hold it. Each Grip
press rebases the current controller pose to the actual current xArm pose, so
controller motion while released cannot create a re-enable jump. OpenXR axes
are mapped to xArm as controller forward/right/up to robot forward/right/up.
`--quest-arm-translation-scale` adjusts translation gain and
`--quest-grip-threshold` adjusts the deadman threshold.

If controller tracking is lost, xArm motion is held immediately. After the
controller is tracked again, release Grip once and press it again to rebase and
resume teleoperation safely.

With tactile contact stopping enabled, each finger latches after three
consecutive samples at or above its threshold. A latched finger resumes after
three consecutive fresh samples below that threshold; other latched fingers
remain held independently.

Allegro interpolation commands start enabled. Use the Viser `Pause hand`
button to hold feedback, and `Enable pose interpolation` to resume.
