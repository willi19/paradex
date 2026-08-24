import json
import tempfile
import unittest
from pathlib import Path

from src.calibration.intrinsic.calculate import calibration_image_size


class IntrinsicCalculateTests(unittest.TestCase):
    def test_calibration_image_size_prefers_requested_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keypoints = Path(temp_dir) / "capture.npy"
            keypoints.touch()
            (Path(temp_dir) / "capture.json").write_text(
                json.dumps({"width": 2048, "height": 1536})
            )

            self.assertEqual(calibration_image_size(str(keypoints), (1024, 768)), (1024, 768))

    def test_calibration_image_size_reads_capture_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keypoints = Path(temp_dir) / "capture.npy"
            keypoints.touch()
            (Path(temp_dir) / "capture.json").write_text(
                json.dumps({"width": 1936, "height": 1216})
            )

            self.assertEqual(calibration_image_size(str(keypoints), None), (1936, 1216))

    def test_calibration_image_size_rejects_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keypoints = Path(temp_dir) / "capture.npy"
            keypoints.touch()
            (Path(temp_dir) / "capture.json").write_text(
                json.dumps({"width": 0, "height": 1536})
            )

            with self.assertRaisesRegex(ValueError, "invalid image size"):
                calibration_image_size(str(keypoints), None)

    def test_calibration_image_size_preserves_legacy_default_without_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            keypoints = Path(temp_dir) / "capture.npy"
            keypoints.touch()

            self.assertEqual(calibration_image_size(str(keypoints), None), (2048, 1536))


if __name__ == "__main__":
    unittest.main()
