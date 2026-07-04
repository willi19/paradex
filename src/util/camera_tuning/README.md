# Camera Tuning

Interactive tools to dial in per-camera image settings.

## Scripts
| File | Purpose |
|------|---------|
| `live_tuner.py` | Live per-camera **gain/exposure** tuner. Shows all cameras in one window; adjust the selected camera's exposure/gain in real time and save to `camera.json`. |

## Usage
Run **on a capture PC** (needs PySpin + cameras + a display). Run it standalone —
not while a capture daemon holds the cameras.

```bash
python src/util/camera_tuning/live_tuner.py
python src/util/camera_tuning/live_tuner.py --serials 25305460 25305462 --fps 15
```

Controls (in the OpenCV window):

| Control | Action |
|---------|--------|
| trackbar `camera` | select which camera the sliders control |
| trackbar `exp_us` | exposure time (µs) of the selected camera |
| trackbar `gain_db` | gain (dB) of the selected camera |
| keys `[` `]` | previous / next camera |
| keys `-` `=` | exposure down / up (100 µs) |
| keys `;` `'` | gain down / up (1 dB) |
| key `s` | save all cameras' current values to `camera.json` |
| key `q` | quit |

The selected camera is highlighted (yellow border) and each tile shows its current
`exp`/`gain`.

## Inputs & Outputs
- Reads current per-camera baseline from `system/current/camera.json` (via `load_camera`).
- On `s`, writes the tuned `exposure`/`gain` per serial back to `system/current/camera.json`.

## Related
- [`paradex/io/camera_system/pyspin.py`](../../../paradex/io/camera_system/pyspin.py) — `load_camera`, `PyspinCamera` (exposure/gain applied here).
- [`system/current/camera.json`](../../../system/current/camera.json) — per-camera `gain`/`exposure` config.
- [`src/dataset_acquisition/graphics/`](../../dataset_acquisition/graphics/README.md) — non-interactive exposure×gain **sweep** capture.
- [`src/validate/camera_system/`](../../validate/camera_system/README.md) — camera smoke tests / sync check.
