# Teleop Input Validation

Verifies that teleop motion-capture input (XSens body suit or Oculus hand tracking) streams in and renders as a live skeleton.

## Scripts
| File | Purpose |
|------|---------|
| `io_connection.py` | Connects to a teleop receiver (XSens or Oculus), starts recording to `tmp/hand_pose`, and live-renders both hands' skeletons (recentered on the wrist midpoint) until `q` is pressed. |
| `xsens_body.py` | Same live skeleton visualization, but does **not** start/stop recording and uses `receiver.quit()` — a lighter connectivity check of the stream. |

## Usage
```bash
# XSens body suit (teleop motion capture)
python src/validate/teleop/io_connection.py --device xsens
python src/validate/teleop/xsens_body.py    --device xsens

# Oculus hand tracking
python src/validate/teleop/io_connection.py --device occulus
python src/validate/teleop/xsens_body.py    --device occulus
```

`--device` is required (`xsens` or `occulus`). Note the `occulus` spelling matches the code.

Hardware required:
- **XSens**: XSens motion-capture suit streaming to the network (`network_info['xsens']['param']` for `io_connection.py`; default ctor for `xsens_body.py`).
- **Oculus**: Oculus device providing hand tracking (`OculusReceiver`, default ctor).

### Controls
- Press `q` to stop (keyboard listener sets the stop event); the window closes and the receiver shuts down.

## What it validates
- The receiver connects and `get_data()` returns non-`None` `Left` and `Right` frames.
- A skeleton appears in the `HandVisualizer` window and moves with the operator at ~50 Hz (loop capped at 0.02 s).
- Joints are recentered so the wrist-midpoint of the first valid frame sits at the origin.

## Related
- [`paradex/io/teleop/`](../../../paradex/io/teleop) — `XSensReceiver`, `OculusReceiver`, joint-name/parent tables.
- [`paradex/visualization/skeleton/hand.py`](../../../paradex/visualization/skeleton/hand.py) — `HandVisualizer`.
- [`paradex/utils/keyboard_listener.py`](../../../paradex/utils/keyboard_listener.py) — `listen_keyboard`.
- [`paradex/utils/system.py`](../../../paradex/utils/system.py) — `network_info`.
