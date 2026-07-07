"""No-hardware logic test for the P4 get_image() timeout fix.

Fakes PySpin/cv2 so the real `paradex.io.camera_system.pyspin` module imports
without a camera, then checks the core contract of the fix:

  * get_image() passes a FINITE timeout (GRAB_TIMEOUT_MS) to GetNextImage, not the
    old infinite blocking call.
  * when GetNextImage raises a SpinnakerException (no frame — LAN drop / no
    trigger), get_image() returns (None, None) instead of raising or blocking.

Run anywhere (no cameras, no PySpin):

    python src/validate/camera_system/hang_recovery_mock.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# --- fake the hardware deps so pyspin.py can be imported ------------------------
if "PySpin" not in sys.modules:
    _ps = types.ModuleType("PySpin")
    class SpinnakerException(Exception):
        pass
    _ps.SpinnakerException = SpinnakerException
    _ps.System = types.SimpleNamespace(
        GetInstance=lambda: types.SimpleNamespace(GetCameras=lambda: [])
    )
    sys.modules["PySpin"] = _ps
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from paradex.io.camera_system import pyspin as py
from paradex.io.camera_system.pyspin import PyspinCamera, GRAB_TIMEOUT_MS


class _FakeCamNoFrame:
    """Simulates a camera that never delivers a frame (raises on grab)."""
    def __init__(self):
        self.timeout_arg = "NOT_CALLED"

    def GetNextImage(self, timeout=None):
        self.timeout_arg = timeout
        raise py.ps.SpinnakerException("simulated: no frame / grab timeout")


def _bare_camera():
    cam = PyspinCamera.__new__(PyspinCamera)   # skip __init__ (no hardware)
    cam.serial_num = "TEST"
    cam.init_time = 0.0
    return cam


def test_finite_timeout_and_none_on_loss():
    cam = _bare_camera()
    cam.cam = _FakeCamNoFrame()

    frame, data = cam.get_image()

    assert frame is None and data is None, f"expected (None, None), got {(frame, data)}"
    assert cam.cam.timeout_arg == GRAB_TIMEOUT_MS, (
        f"get_image() must pass finite GRAB_TIMEOUT_MS={GRAB_TIMEOUT_MS} to "
        f"GetNextImage; got {cam.cam.timeout_arg!r} (infinite block == the bug)")
    print(f"PASS: frame loss -> (None, None); finite timeout {GRAB_TIMEOUT_MS}ms passed to GetNextImage")


def main():
    tests = [test_finite_timeout_and_none_on_loss]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
