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


def test_c2r_only_loads_saved_triangulation_and_recovers_scale(
    monkeypatch,
    tmp_path,
):
    import src.calibration.extrinsic.calculate as camera_calculate

    for step in ("Right/0", "Left/0"):
        step_dir = tmp_path / step
        step_dir.mkdir(parents=True)
        np.save(step_dir / "kypt_3d_id.npy", np.array([0, 1]))
        np.save(
            step_dir / "kypt_3d_cor.npy",
            np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]),
        )
    monkeypatch.setattr(
        camera_calculate,
        "get_adjecent_ids",
        lambda: {0: [1], 1: [0]},
    )

    scale, triangulated = calculate.load_completed_camera_stage(
        str(tmp_path),
        ["Right/0", "Left/0"],
    )

    assert scale == 0.5
    assert set(triangulated) == {"Right/0", "Left/0"}


def test_c2r_only_rejects_incomplete_camera_stage(tmp_path):
    step_dir = tmp_path / "Right" / "0"
    step_dir.mkdir(parents=True)
    np.save(step_dir / "kypt_3d_id.npy", np.array([0, 1]))

    try:
        calculate.load_completed_camera_stage(str(tmp_path), ["Right/0"])
    except FileNotFoundError as error:
        assert "Camera calibration is incomplete" in str(error)
        assert "kypt_3d_cor.npy" in str(error)
    else:
        raise AssertionError("Expected incomplete camera stage to be rejected")


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


def test_bimanual_steps_are_combined_as_relative_paths(tmp_path):
    for side in ("Right", "Left"):
        for step in ("0", "1"):
            step_dir = tmp_path / side / step
            (step_dir / "images").mkdir(parents=True)
            np.save(step_dir / "qpos.npy", np.zeros(6))
            np.save(step_dir / "eef.npy", np.eye(4))

    steps_by_side = calculate.validate_bimanual_allinone_capture(str(tmp_path))

    assert steps_by_side == {"Right": ["0", "1"], "Left": ["0", "1"]}
    assert calculate.combine_bimanual_steps(steps_by_side) == [
        "Right/0",
        "Right/1",
        "Left/0",
        "Left/1",
    ]


def test_camera_stage_passes_bimanual_relative_steps_to_colmap(
    monkeypatch,
    tmp_path,
):
    import src.calibration.extrinsic.calculate as camera_calculate

    calls = []
    monkeypatch.setattr(calculate, "cam_param_dir", str(tmp_path / "cam_param"))
    monkeypatch.setattr(calculate, "save_charuco_markers", lambda *_args: None)
    monkeypatch.setattr(calculate, "undistort_images_parallel", lambda *_args: None)
    monkeypatch.setattr(
        calculate,
        "load_all_saved_charuco_detections",
        lambda _root, steps, _intrinsics: {step: {} for step in steps},
    )
    monkeypatch.setattr(
        calculate,
        "triangulate_saved_charuco",
        lambda _root, steps, *_args: {
            step: {
                "ids": np.array([0, 1]),
                "corners": np.array([[0.0, 0.0, 0.0], [0.025, 0.0, 0.0]]),
            }
            for step in steps
        },
    )
    monkeypatch.setattr(
        calculate,
        "get_reconstructed_length",
        lambda *_args: calculate.HAND_EYE_SQUARE_LENGTH_M,
    )
    monkeypatch.setattr(
        calculate,
        "calculate_reprojection_errors",
        lambda *_args: {"camera": [0.1]},
    )
    monkeypatch.setattr(calculate, "save_reprojection_errors", lambda *_args: None)
    monkeypatch.setattr(
        camera_calculate,
        "load_colmap_camparam",
        lambda _path: ({"camera": {}}, {"camera": np.eye(4)}),
    )
    monkeypatch.setattr(camera_calculate, "get_adjecent_ids", lambda: {0: [1], 1: [0]})

    def fake_run_calibration(name, root_dir=None, index_list=None):
        calls.append((name, Path(root_dir), list(index_list)))

    monkeypatch.setattr(camera_calculate, "run_calibration", fake_run_calibration)

    steps = ["Right/0", "Right/1", "Left/0", "Left/1"]
    scale, triangulated = calculate.calculate_camera_stage(
        "session",
        str(tmp_path),
        steps,
        ["Right/0", "Left/0"],
    )

    assert scale == 1.0
    assert calls == [("session", tmp_path, steps)]
    assert set(triangulated) == set(steps)
    assert (tmp_path / "Right" / "0" / "cam_param" / "intrinsics.json").is_file()
    assert (tmp_path / "Left" / "0" / "cam_param" / "extrinsics.json").is_file()


def test_bimanual_handeye_saves_right_and_left_outputs(monkeypatch, tmp_path):
    from src.calibration.handeye.xarm import calculate as handeye_calculate

    session_root = tmp_path / "extrinsic" / "20260820_120000"
    for side in ("Right", "Left"):
        for step in ("0", "1"):
            (session_root / side / step).mkdir(parents=True)

    output_root = tmp_path / "handeye_calibration_bimanual"
    calls = []
    debug_calls = []
    scaled = []

    monkeypatch.setattr(calculate, "handeye_calib_bimanual_path", str(output_root))
    monkeypatch.setattr(
        calculate,
        "scale_triangulated_charuco",
        lambda root, steps, scale, triangulated: scaled.append(
            (Path(root), list(steps), scale, sorted(triangulated))
        ),
    )

    def fake_calculate_sequence(
        root_dir,
        arm,
        save_path,
        precomputed_charuco,
        render_debug,
    ):
        calls.append(
            (
                Path(root_dir),
                arm,
                Path(save_path),
                precomputed_charuco,
                render_debug,
            )
        )
        np.save(save_path, np.eye(4))
        return np.eye(4)

    monkeypatch.setattr(
        handeye_calculate,
        "calculate_sequence",
        fake_calculate_sequence,
    )
    monkeypatch.setattr(
        handeye_calculate,
        "debug",
        lambda root_dir, arm, transform: debug_calls.append(
            (Path(root_dir), arm, transform.copy())
        ),
    )

    steps_by_side = {"Right": ["0", "1"], "Left": ["0", "1"]}
    triangulated = {
        f"{side}/{step}": {"ids": np.array([0]), "corners": np.zeros((1, 3))}
        for side in ("Right", "Left")
        for step in ("0", "1")
    }

    outputs = calculate.calculate_bimanual_handeye_stage(
        str(session_root),
        "xarm",
        steps_by_side,
        2.0,
        triangulated,
    )

    assert outputs == {
        "Right": str(output_root / "20260820_120000" / "C2R_R.npy"),
        "Left": str(output_root / "20260820_120000" / "C2R_L.npy"),
    }
    assert {
        (root, tuple(steps), scale, tuple(keys))
        for root, steps, scale, keys in scaled
    } == {
        (session_root / "Right", ("0", "1"), 2.0, ("0", "1")),
        (session_root / "Left", ("0", "1"), 2.0, ("0", "1")),
    }
    assert [
        (root, arm, save_path.name, precomputed, render_debug)
        for root, arm, save_path, precomputed, render_debug in calls
    ] == [
        (session_root / "Right", "xarm", "C2R_R.npy", True, False),
        (session_root / "Left", "xarm_left", "C2R_L.npy", True, False),
    ]
    assert {
        (root, arm) for root, arm, _transform in debug_calls
    } == {
        (session_root / "Right", "xarm"),
        (session_root / "Left", "xarm_left"),
    }
    assert all(np.array_equal(transform, np.eye(4)) for _, _, transform in debug_calls)
    assert (output_root / "20260820_120000" / "C2R_R.npy").is_file()
    assert (output_root / "20260820_120000" / "C2R_L.npy").is_file()
