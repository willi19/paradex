# pringles_test

Pick-and-place of the pringles can using a **refined lookup trajectory** (pre-recorded arm
+ hand trajectories), rather than a freshly generated grasp. Estimates the object 6D pose,
transforms the lookup trajectories into the current object frame, and runs them through a
`RobotGUIController` with `pick` and `place` stages.

## Scripts
| File | Purpose |
|------|---------|
| `grasp.py` | Main: load refined pick/place trajectories -> capture -> object 6D pose -> transform trajectories into object frame -> launch GUI with `pick`/`place` stages. |
| `asdf.py` | **Scratch** — just loads the four refined trajectory `.npy` files; no further logic. Useful only to inspect the lookup format. |

## Usage
Run from repo root. Trajectories are read from a relative `lookup/pringles/<index>/` dir
(index hardcoded to `"1"`).

```bash
# Execute pick-and-place via GUI
python src/inference/pringles_test/grasp.py
```

The GUI (`RobotGUIController`) exposes `pick` and `place` stages, each a dict of
`{"arm": (N,4,4), "hand": (N,16)}`. The object pose is `normalize_cylinder`-d before the
trajectories are applied, and the place trajectory force-opens the hand for the last 70
steps.

## Inputs & Outputs
- **Reads:** `lookup/pringles/1/refined_{pick,place}_action.npy` (arm, N×4×4) and
  `refined_{pick,place}_hand.npy` (Allegro qpos, N×16); local `marker_offset.npy`;
  object mesh `rsc/object/pringles/pringles.obj`; current cam params + `C2R`.
- **Captures:** images to `~/shared_data/inference/grasp_eval/<timestamp>/raw`, undistorted in place.
- **Writes:** `projected/` overlay images.

## Related
- Uses the same capture + object-6D backbone as the sibling inference dirs ([`../bodex/`](../bodex/), [`../grasp_w_gui/`](../grasp_w_gui/)).
- [`paradex/io/robot_controller/`](../../../paradex/io/robot_controller/), [`paradex/image/image_dict.py`](../../../paradex/image/image_dict.py).
