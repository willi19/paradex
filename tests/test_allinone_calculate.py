from pathlib import Path
import threading

import numpy as np

from src.calibration.allinone import calculate


def test_saved_charuco_detections_are_reused_in_undistorted_coordinates(tmp_path):
    marker_dir = Path(tmp_path) / "markers_2d"
    marker_dir.mkdir()
    corners = np.array([[100.0, 120.0], [200.0, 220.0]], dtype=np.float32)
    np.save(marker_dir / "camera_corner.npy", corners)
    np.save(marker_dir / "camera_id.npy", np.array([0, 1]))

    camera_matrix = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )
    intrinsics = {
        "camera": {
            "original_intrinsics": camera_matrix,
            "intrinsics_undistort": camera_matrix,
            "dist_params": np.zeros(5),
        }
    }

    detections = calculate.load_saved_charuco_detections(tmp_path, intrinsics)
    board_detection = next(iter(detections["camera"].values()))

    np.testing.assert_array_equal(board_detection["checkerIDs"], [0, 1])
    np.testing.assert_allclose(
        board_detection["checkerCorner"],
        corners,
        atol=1e-5,
    )


def test_charuco_images_are_detected_in_parallel(monkeypatch, tmp_path):
    image_dir = tmp_path / "0" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "camera-a.png").touch()
    (image_dir / "camera-b.png").touch()

    barrier = threading.Barrier(2)
    thread_ids = set()
    board_id = next(iter(calculate.boardinfo_dict))

    def fake_detect(_image):
        thread_ids.add(threading.get_ident())
        barrier.wait(timeout=1.0)
        return {
            board_id: {
                "checkerCorner": np.array([[10.0, 20.0]]),
                "checkerIDs": np.array([0]),
            }
        }

    monkeypatch.setattr(
        calculate,
        "load_current_intrinsic",
        lambda: {"camera-a": {}, "camera-b": {}},
    )
    monkeypatch.setattr(calculate.cv2, "imread", lambda *_args: np.zeros((2, 2, 3)))
    monkeypatch.setattr(calculate, "detect_charuco", fake_detect)

    calculate.save_charuco_markers(str(tmp_path), ["0"])

    assert len(thread_ids) == 2
    assert (tmp_path / "0" / "markers_2d" / "camera-a_corner.npy").is_file()
    assert (tmp_path / "0" / "markers_2d" / "camera-b_corner.npy").is_file()


def test_undistortion_runs_images_in_parallel(monkeypatch, tmp_path):
    image_dir = tmp_path / "0" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "camera-a.png").touch()
    (image_dir / "camera-b.png").touch()

    barrier = threading.Barrier(2)
    thread_ids = set()

    def fake_undistort(*_args):
        thread_ids.add(threading.get_ident())
        barrier.wait(timeout=1.0)

    monkeypatch.setattr(
        calculate,
        "precomute_undistort_map",
        lambda _intrinsic: (None, np.zeros((1, 1)), np.zeros((1, 1))),
    )
    monkeypatch.setattr(calculate, "undistort_image_file", fake_undistort)

    cameras = {"camera-a": {}, "camera-b": {}}
    calculate.undistort_images_parallel(
        str(tmp_path), ["0"], cameras, cameras
    )

    assert len(thread_ids) == 2


def test_scaled_triangulation_is_reused_for_handeye(tmp_path):
    step_dir = tmp_path / "0"
    step_dir.mkdir()
    ids = np.array([2, 7])
    points = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    np.save(step_dir / "kypt_3d_id.npy", ids)
    np.save(step_dir / "kypt_3d_cor.npy", points)

    calculate.scale_triangulated_step(str(tmp_path), "0", 0.025)

    np.testing.assert_array_equal(np.load(step_dir / "charuco_3d_ids.npy"), ids)
    np.testing.assert_allclose(
        np.load(step_dir / "charuco_3d_corners.npy"),
        points * 0.025,
    )


def test_timing_summary_reports_each_stage(capsys):
    calculate.print_timing_summary(65.25, 5.5, 70.75)

    assert capsys.readouterr().out == (
        "Calibration timing:\n"
        "  Extrinsic: 1m 5.25s\n"
        "  Hand-eye:  5.50s\n"
        "  Total:     1m 10.75s\n"
    )


def test_handeye_result_uses_canonical_current_c2r_layout(monkeypatch, tmp_path):
    from src.calibration.handeye.xarm import calculate as handeye_calculate

    capture_root = tmp_path / "extrinsic" / "20260820_120000"
    (capture_root / "0").mkdir(parents=True)
    canonical_root = tmp_path / "handeye_calibration"
    saved_paths = []

    monkeypatch.setattr(calculate, "handeye_calib_path", str(canonical_root))
    monkeypatch.setattr(
        calculate,
        "scale_triangulated_charuco",
        lambda *_args, **_kwargs: None,
    )

    def fake_calculate_sequence(root_dir, arm, save_path, precomputed_charuco):
        assert root_dir == str(capture_root)
        assert arm == "xarm"
        assert precomputed_charuco is True
        saved_paths.append(Path(save_path))
        np.save(save_path, np.eye(4))

    monkeypatch.setattr(
        handeye_calculate,
        "calculate_sequence",
        fake_calculate_sequence,
    )

    output_path = calculate.calculate_handeye_stage(
        str(capture_root), "xarm", ["0"], 1.0
    )

    expected = canonical_root / "20260820_120000" / "0" / "C2R.npy"
    assert Path(output_path) == expected
    assert saved_paths == [expected]
    np.testing.assert_array_equal(np.load(expected), np.eye(4))
