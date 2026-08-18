import numpy as np

from paradex.image.merge import merge_image
from paradex.io.camera_system.capture_pc_preview import CapturePcPreviewGui


class FakeReader:
    def __init__(self, host, port, request_timeout):
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.camera_names = ["camera-1"]

    def get_preview(self, camera_name):
        assert camera_name == "camera-1"
        return np.zeros((10, 10, 3), dtype=np.uint8)


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self):
        self.shown = []

    def imshow(self, window_name, image):
        self.shown.append((window_name, image))

    def waitKey(self, _delay):
        return -1

    def destroyWindow(self, _window_name):
        pass

    def putText(self, image, *_args):
        return image


def test_capture_pc_preview_collects_each_pc_frame_with_pc_label():
    preview = CapturePcPreviewGui(
        pc_list=["capture1"],
        reader_factory=FakeReader,
        host_lookup=lambda _: "127.0.0.1",
    )

    preview._connect_readers()
    images, frame_text = preview._collect_frames()

    assert list(images) == ["capture1:camera-1"]
    assert images["capture1:camera-1"].shape == (10, 10, 3)
    assert frame_text == {"capture1:camera-1": "preview"}


def test_capture_pc_preview_shows_latest_frame_only_when_called_from_main_loop():
    fake_cv2 = FakeCv2()
    preview = CapturePcPreviewGui(
        pc_list=[],
        cv2_module=fake_cv2,
    )
    preview._latest_display = np.zeros((10, 10, 3), dtype=np.uint8)

    preview.show()

    assert len(fake_cv2.shown) == 1


def test_capture_pc_preview_retains_frames_after_a_transient_request_failure():
    class FlakyReader:
        camera_names = ["camera-1"]

        def __init__(self):
            self.fail = False

        def get_preview(self, _camera_name):
            if self.fail:
                from paradex.io.camera_system.camera_daemon_reader import CameraDaemonReaderError

                raise CameraDaemonReaderError("temporary timeout")
            return np.full((10, 10, 3), 7, dtype=np.uint8)

    preview = CapturePcPreviewGui(pc_list=[])
    reader = FlakyReader()
    preview._readers["capture1"] = reader

    first_images, _ = preview._collect_frames()
    reader.fail = True
    second_images, frame_text = preview._collect_frames()

    assert "capture1" in preview._readers
    assert np.array_equal(second_images["capture1:camera-1"], first_images["capture1:camera-1"])
    assert frame_text["capture1:camera-1"] == "preview (stale)"


def test_camera_preview_grid_uses_four_columns_without_stretching_images():
    images = {
        f"camera-{index:02d}": np.zeros((30, 40, 3), dtype=np.uint8)
        for index in range(20)
    }

    merged = merge_image(
        images,
        put_text=False,
        grid_cols=4,
        preserve_aspect=True,
        target_height=1230,
    )

    # Four columns and five rows, retaining the old preview's total height.
    assert merged.shape == (1230, 2078, 3)

    first_cell = merged[:238, :512]
    image_mask = np.all(first_cell == 0, axis=2)
    ys, xs = np.where(image_mask)
    rendered_height = ys.max() - ys.min() + 1
    rendered_width = xs.max() - xs.min() + 1
    assert rendered_height == 238
    assert np.isclose(rendered_width / rendered_height, 4.0 / 3.0, atol=0.01)
