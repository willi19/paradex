"""Hardware-free test of the decoupled sink model in Camera.

Fakes PyspinCamera (via pyspin.load_camera) so the real capture thread + acquire
loop run, and exercises: arm with no sink -> stream on -> video on -> video off ->
snapshot, then stop/end. Asserts frames flow, sinks toggle live, and files land.
"""
import os, sys, time, glob, tempfile, types
import numpy as np

SHAPE = (240, 320, 3)


class FakeCam:
    """Minimal stand-in for PyspinCamera."""
    def __init__(self):
        self.fid = 0
        self.acquiring = False

    def start(self, mode, syncMode, frame_rate=None, gain=None, exposure_time=None):
        self.acquiring = True

    def get_image(self):
        if not self.acquiring:
            return None, None
        self.fid += 1
        time.sleep(0.005)  # ~200 fps, keeps the loop from pegging a core
        frame = np.full(SHAPE, self.fid % 255, dtype=np.uint8)
        return frame, {"frameID": self.fid, "pc_time": time.time()}

    def set_gain(self, gain):
        self.gain = float(gain)

    def set_exposure(self, exposure_time):
        self.exposure_time = float(exposure_time)

    def stop(self):
        self.acquiring = False

    def release(self):
        pass


# Inject a fake pyspin module so Camera.connect_camera's
# `from paradex.io.camera_system.pyspin import load_camera` gets FakeCam
# (the real module imports PySpin, which isn't installed here).
_fake = types.ModuleType("paradex.io.camera_system.pyspin")
_fake.load_camera = lambda name, cfg=None: FakeCam()
sys.modules["paradex.io.camera_system.pyspin"] = _fake

from paradex.io.camera_system.camera import Camera


def main():
    tmp = tempfile.mkdtemp(prefix="camsink_")
    cam = Camera("pyspin", "testcam", frame_shape=SHAPE)

    # 1) arm with NO sink
    cam.start("acquire", syncMode=False, fps=30)
    time.sleep(0.4)
    f1 = cam.get_frame_id()
    assert f1 > 0, "acquire loop not advancing frames"
    assert not glob.glob(os.path.join(tmp, "*")), "no sink yet but files appeared"
    print(f"[ok] armed, no sink; frame_id advancing ({f1})")

    # 2) stream sink on -> SHM should be written (fid buffers advance)
    cam.set_sink(stream=True)
    time.sleep(0.3)
    shm_fid = max(int(cam.fid_array_a[0]), int(cam.fid_array_b[0]))
    assert shm_fid > 0, "stream sink on but SHM fid not advancing"
    print(f"[ok] stream sink on; SHM fid={shm_fid}")

    # 2b) live gain/exposure change applied on the capture thread
    cam.set_param(gain=5.0, exposure=8000.0)
    time.sleep(0.2)
    assert cam.camera.gain == 5.0 and cam.camera.exposure_time == 8000.0, \
        f"live param not applied: gain={cam.camera.gain} exp={cam.camera.exposure_time}"
    print(f"[ok] live set_param applied: gain={cam.camera.gain} exp={cam.camera.exposure_time}")

    # 3) video sink on -> .avi is created
    vdir = os.path.join(tmp, "vid")
    cam.set_sink(video=True, save_path=vdir)
    time.sleep(0.6)
    vids = glob.glob(os.path.join(vdir, "*.avi"))
    assert vids, "video sink on but no .avi created"
    print(f"[ok] video sink on; {os.path.basename(vids[0])} created")

    # 4) video sink off -> writer released & file flushed (VideoWriter buffers
    #    until release, so the finalized file must be non-empty), acquisition continues
    cam.set_sink(video=False)
    time.sleep(0.4)
    fsz = os.path.getsize(vids[0])
    assert fsz > 0, f"video file empty after release ({fsz}B) — writer never flushed"
    still = cam.get_frame_id()
    time.sleep(0.2)
    assert cam.get_frame_id() > still, "acquisition stopped after closing video sink (should keep running)"
    print(f"[ok] video sink off; file finalized {fsz}B; acquisition still running")

    # 5) snapshot sink -> exactly 3 images
    sdir = os.path.join(tmp, "snap")
    cam.set_sink(snapshot=(sdir, 3))
    time.sleep(0.4)
    pngs = sorted(glob.glob(os.path.join(sdir, "*.png")))
    assert len(pngs) == 3, f"expected 3 snapshot images, got {len(pngs)}"
    print(f"[ok] snapshot sink wrote {len(pngs)} images")

    # 6) stop + end cleanly
    cam.stop(); time.sleep(0.2)
    t0 = time.time(); cam.end(timeout=3)
    assert time.time() - t0 < 6, "end() hung"
    assert not cam.capture_thread.is_alive(), "capture thread did not exit"
    print(f"[ok] stop+end clean in {time.time()-t0:.2f}s")

    print("\nALL PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e); sys.exit(1)
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(2)
