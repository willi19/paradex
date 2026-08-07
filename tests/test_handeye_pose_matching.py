import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from paradex.calibration.handeye_pose_matching import (
    align_keypoints,
    build_servo_target,
    estimate_camera_from_robot_rotation,
    fit_multiview_pose,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_optimizer_script():
    path = REPO_ROOT / "src/calibration/handeye/xarm/optimize_bimanual_poses.py"
    spec = importlib.util.spec_from_file_location("optimize_bimanual_poses_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_detector_uses_one_shot_image_session(monkeypatch, tmp_path):
    optimizer = load_optimizer_script()
    calls = []

    class FakeRemoteCameraController:
        def start(self, mode, sync_mode, save_path=None, fps=30):
            calls.append(("start", mode, sync_mode, save_path, fps))

        def stop(self):
            calls.append(("stop",))

    expected = optimizer.KeypointObservation(
        ids=np.arange(6),
        points=np.zeros((6, 3)),
        path=tmp_path / "expected",
    )
    monkeypatch.setattr(optimizer, "remove_home", lambda path: "relative/" + path.name)
    monkeypatch.setattr(optimizer, "extract_keypoints", lambda **_kwargs: expected)
    detector = optimizer.SnapshotKeypointDetector(
        rcc=FakeRemoteCameraController(),
        intrinsic={},
        extrinsic={},
        observation_root=tmp_path / "observations",
        workers=1,
        checker_length_m=0.025,
        fps=20,
    )

    actual = detector.capture("Left/pose_000/iteration_00")

    assert actual is expected
    assert calls == [
        ("start", "image", False, "relative/iteration_00", 20),
        ("stop",),
    ]


def test_reference_overlay_blends_matching_camera_images(tmp_path):
    optimizer = load_optimizer_script()
    reference_dir = tmp_path / "reference/images"
    current_dir = tmp_path / "current/images"
    reference_dir.mkdir(parents=True)
    current_dir.mkdir(parents=True)
    reference_image = np.full((8, 10, 3), 40, dtype=np.uint8)
    current_image = np.full((8, 10, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(reference_dir / "camera_a.png"), reference_image)
    cv2.imwrite(str(current_dir / "camera_a.png"), current_image)

    output_dir = optimizer.save_reference_overlays(
        reference_path=reference_dir.parent,
        current_path=current_dir.parent,
        output_dir=tmp_path / "overlay",
        alpha=0.5,
    )

    overlay = cv2.imread(str(output_dir / "camera_a.png"))
    np.testing.assert_array_equal(overlay, np.full((8, 10, 3), 120, dtype=np.uint8))


def test_both_side_specs_run_left_before_right(tmp_path):
    optimizer = load_optimizer_script()
    args = SimpleNamespace(
        side="both",
        left_ip="left-ip",
        right_ip="right-ip",
        left_seed_dir=str(tmp_path / "left"),
        right_seed_dir=str(tmp_path / "right"),
    )

    specs = optimizer._side_specs(args)

    assert [name for name, _ip, _seed_dir in specs] == ["Left", "Right"]


def test_optimizer_defaults_use_coarse_bounded_convergence():
    optimizer = load_optimizer_script()

    args = optimizer.build_parser().parse_args([])

    assert args.resume_latest is False
    assert args.side is None
    assert args.translation_gain == 0.9
    assert args.rotation_gain == 0.9
    assert args.max_translation_step_mm == 35.0
    assert args.max_rotation_step_deg == 15.0
    assert args.one_shot_max_translation_mm == 150.0
    assert args.one_shot_max_rotation_deg == 30.0


def test_resume_latest_must_be_enabled_explicitly():
    optimizer = load_optimizer_script()

    args = optimizer.build_parser().parse_args(["--resume-latest"])

    assert args.resume_latest is True


def test_completed_pose_prefix_returns_last_successful_qpos(tmp_path):
    optimizer = load_optimizer_script()
    side_dir = tmp_path / "Left"
    side_dir.mkdir()
    expected_qpos = np.linspace(0.1, 0.6, 6)
    optimizer._write_json(
        side_dir / "0_metrics.json",
        {"pose_index": 0, "success": True},
    )
    np.save(side_dir / "0_qpos.npy", expected_qpos)
    # A later saved pose is intentionally ignored because pose 1 is missing:
    # resume must preserve the contiguous calibration order.
    optimizer._write_json(
        side_dir / "2_metrics.json",
        {"pose_index": 2, "success": True},
    )
    np.save(side_dir / "2_qpos.npy", np.ones(6))

    reports, completed, pending, warm_qpos = optimizer._completed_pose_prefix(
        tmp_path, "Left", [0, 1, 2]
    )

    assert list(reports) == ["0"]
    assert completed == [0]
    assert pending == [1, 2]
    np.testing.assert_allclose(warm_qpos, expected_qpos)


def test_latest_resume_dir_uses_newest_session_even_if_marked_complete(
    monkeypatch, tmp_path
):
    optimizer = load_optimizer_script()
    incomplete = tmp_path / "20260803_120000"
    complete = tmp_path / "20260803_130000"
    incomplete.mkdir()
    complete.mkdir()
    optimizer._write_json(incomplete / "manifest.json", {"status": "failed"})
    optimizer._write_json(complete / "manifest.json", {"status": "complete"})
    monkeypatch.setattr(optimizer, "DEFAULT_OUTPUT_ROOT", tmp_path)

    assert optimizer._latest_resume_dir() == complete


def test_resume_defaults_to_both_sides_even_if_manifest_only_has_left(
    monkeypatch, tmp_path, capsys
):
    optimizer = load_optimizer_script()
    output_dir = tmp_path / "20260803_125942"
    output_dir.mkdir()
    optimizer._write_json(
        output_dir / "manifest.json",
        {
            "reference_dir": str(tmp_path / "reference"),
            "pose_indices": [0],
            "side_order": ["Left"],
            "status": "complete",
            "sides": {"Left": {"poses": {}}},
        },
    )
    left_dir = output_dir / "Left"
    left_dir.mkdir()
    optimizer._write_json(
        left_dir / "0_metrics.json",
        {"pose_index": 0, "success": True},
    )
    np.save(left_dir / "0_qpos.npy", np.zeros(6))
    monkeypatch.setattr(optimizer, "_latest_resume_dir", lambda: output_dir)
    monkeypatch.setattr(optimizer, "_load_seed", lambda *_args: np.zeros(6))
    monkeypatch.setattr(
        "sys.argv", ["optimize_bimanual_poses.py", "--resume-latest"]
    )

    optimizer.main()

    output = capsys.readouterr().out
    assert "Resume Left: 1 completed, 0 pending" in output
    assert "Resume Right: 0 completed, 1 pending (next pose 0)" in output
    assert "Sides: ['Left', 'Right']" in output


def test_prepare_reference_keypoints_reuses_resume_cache(monkeypatch, tmp_path):
    optimizer = load_optimizer_script()
    cached_path = tmp_path / "output/reference_keypoints/4"
    cached_path.mkdir(parents=True)
    expected_ids = np.arange(6)
    expected_points = np.arange(18, dtype=np.float64).reshape(6, 3)
    np.save(cached_path / "charuco_3d_ids.npy", expected_ids)
    np.save(cached_path / "charuco_3d_corners.npy", expected_points)
    optimizer._write_json(
        cached_path / "pose_fit.json",
        {
            "board_id": "3",
            "reprojection_rmse_px": 1.25,
        },
    )

    def fail_extract(**_kwargs):
        raise AssertionError("cached reference should not be recomputed")

    monkeypatch.setattr(optimizer, "extract_keypoints", fail_extract)
    prepared = optimizer.prepare_reference_keypoints(
        reference_dir=tmp_path / "reference",
        pose_indices=[4],
        intrinsic={},
        extrinsic={},
        output_dir=tmp_path / "output",
        workers=1,
        checker_length_m=0.025,
        reuse_existing=True,
    )

    np.testing.assert_array_equal(prepared[4].ids, expected_ids)
    np.testing.assert_array_equal(prepared[4].points, expected_points)
    assert prepared[4].reprojection_rmse_px == 1.25
    assert prepared[4].image_path == tmp_path / "reference/4"


def test_coarse_defaults_need_less_than_half_the_legacy_translation_steps():
    target_x = 0.375
    tolerance_m = 0.003

    def count_steps(gain, max_step_m):
        current = np.eye(4)
        steps = 0
        while target_x - current[0, 3] > tolerance_m:
            current = build_servo_target(
                current_eef=current,
                camera_from_robot_rotation=np.eye(3),
                centroid_error_camera=np.array(
                    [target_x - current[0, 3], 0.0, 0.0]
                ),
                rotation_error_camera=np.eye(3),
                translation_gain=gain,
                rotation_gain=0.9,
                max_translation_step_m=max_step_m,
                max_rotation_step_rad=np.deg2rad(15.0),
            )
            steps += 1
        return steps

    legacy_steps = count_steps(gain=0.85, max_step_m=0.015)
    coarse_steps = count_steps(gain=0.9, max_step_m=0.035)

    assert coarse_steps * 2 <= legacy_steps


def test_align_keypoints_recovers_reference_transform_with_shuffled_ids():
    current_ids = np.array([10, 11, 12, 13, 14, 15])
    current_points = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.08, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.04, 0.04, 0.00],
            [0.08, 0.04, 0.00],
        ]
    )
    rotation = Rotation.from_euler("xyz", [12.0, -8.0, 17.0], degrees=True).as_matrix()
    translation = np.array([0.025, -0.014, 0.033])
    target_points_ordered = (rotation @ current_points.T).T + translation
    shuffle = np.array([3, 0, 5, 2, 1, 4])

    result = align_keypoints(
        target_ids=current_ids[shuffle],
        target_points=target_points_ordered[shuffle],
        current_ids=current_ids,
        current_points=current_points,
        min_common=6,
    )

    np.testing.assert_allclose(result.transform[:3, :3], rotation, atol=1e-10)
    np.testing.assert_allclose(result.transform[:3, 3], translation, atol=1e-10)
    assert result.raw_rmse_m > 0.02
    assert result.registered_rmse_m < 1e-10
    assert result.robust_loss_m2 > 0


def test_estimate_camera_from_robot_rotation_recovers_probe_frame():
    expected = Rotation.from_euler("xyz", [23.0, -31.0, 48.0], degrees=True).as_matrix()
    robot_deltas = np.eye(3) * 0.004
    camera_deltas = (expected @ robot_deltas.T).T

    actual = estimate_camera_from_robot_rotation(robot_deltas, camera_deltas)

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12)
    assert np.linalg.det(actual) > 0


def test_build_servo_target_converts_axes_and_clamps_both_steps():
    camera_from_robot = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    current_eef = np.eye(4)
    rotation_error_camera = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()

    target = build_servo_target(
        current_eef=current_eef,
        camera_from_robot_rotation=camera_from_robot,
        centroid_error_camera=np.array([0.020, 0.0, 0.0]),
        rotation_error_camera=rotation_error_camera,
        translation_gain=1.0,
        rotation_gain=1.0,
        max_translation_step_m=0.010,
        max_rotation_step_rad=np.deg2rad(5.0),
    )

    np.testing.assert_allclose(target[:3, 3], [0.0, -0.010, 0.0], atol=1e-12)
    step_angle = Rotation.from_matrix(target[:3, :3]).magnitude()
    np.testing.assert_allclose(step_angle, np.deg2rad(5.0), atol=1e-12)


def test_fit_multiview_pose_recovers_rigid_board_from_pixels():
    object_ids = np.arange(6)
    object_points = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.08, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.04, 0.04, 0.00],
            [0.08, 0.04, 0.00],
        ]
    )
    intrinsic = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    extrinsics = {
        "a": np.hstack([np.eye(3), np.zeros((3, 1))]),
        "b": np.hstack([np.eye(3), np.array([[-0.18], [0.0], [0.0]])]),
        "c": np.hstack([np.eye(3), np.array([[0.0], [-0.15], [0.0]])]),
    }
    projections = {name: intrinsic @ extrinsic for name, extrinsic in extrinsics.items()}
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler(
        "xyz", [8.0, -11.0, 17.0], degrees=True
    ).as_matrix()
    expected[:3, 3] = [0.03, -0.02, 0.9]
    world_points = (expected[:3, :3] @ object_points.T).T + expected[:3, 3]
    observations = {}
    for name, projection in projections.items():
        homogeneous = np.hstack([world_points, np.ones((len(world_points), 1))])
        projected = (projection @ homogeneous.T).T
        observations[name] = (object_ids, projected[:, :2] / projected[:, 2:])

    initial = expected.copy()
    initial[:3, 3] += [0.02, -0.01, 0.03]
    initial[:3, :3] = Rotation.from_euler("z", 4.0, degrees=True).as_matrix() @ initial[
        :3, :3
    ]
    fit = fit_multiview_pose(
        object_ids,
        object_points,
        observations,
        projections,
        initial,
    )

    np.testing.assert_allclose(fit.transform, expected, atol=1e-8)
    assert fit.reprojection_rmse_px < 1e-7
    assert fit.camera_count == 3
    assert fit.observation_count == 18


def test_optimize_pose_applies_one_shot_cap_then_saves_qpos(monkeypatch, tmp_path):
    optimizer = load_optimizer_script()
    ids = np.arange(6)
    target_points = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.08, 0.00, 0.00],
            [0.00, 0.04, 0.00],
            [0.04, 0.04, 0.00],
            [0.08, 0.04, 0.00],
        ]
    )
    reference = optimizer.KeypointObservation(ids, target_points, tmp_path / "reference")

    class FakeController:
        def __init__(self):
            self.moves = []
            self.eef = np.eye(4)

        def move(self, target, is_servo):
            self.moves.append((np.asarray(target).copy(), is_servo))
            if np.asarray(target).shape == (4, 4):
                self.eef = np.asarray(target).copy()

        def get_data(self):
            return {
                "qpos": np.arange(6, dtype=np.float64),
                "position": self.eef.copy(),
            }

        def is_error(self):
            return False

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def capture(self, label):
            self.calls += 1
            points = target_points + ([0.300, 0.0, 0.0] if self.calls == 1 else 0.0)
            return optimizer.KeypointObservation(ids, points, tmp_path / label)

    args = SimpleNamespace(
        settle_sec=0.0,
        max_moves=2,
        min_common=6,
        huber_delta_mm=4.0,
        max_registered_rmse_mm=1.0,
        max_live_reprojection_rmse_px=5.0,
        keypoint_tolerance_mm=1.0,
        rotation_tolerance_deg=1.0,
        translation_gain=1.0,
        rotation_gain=1.0,
        max_translation_step_mm=10.0,
        max_rotation_step_deg=5.0,
        one_shot_max_translation_mm=150.0,
        one_shot_max_rotation_deg=30.0,
    )
    monkeypatch.setattr(optimizer.time, "sleep", lambda _seconds: None)
    controller = FakeController()

    report = optimizer.optimize_pose(
        controller=controller,
        detector=FakeDetector(),
        side_name="Right",
        pose_index=7,
        reference=reference,
        seed_qpos=np.zeros(6),
        camera_from_robot_rotation=np.eye(3),
        output_dir=tmp_path / "output",
        args=args,
    )

    assert report["success"] is True
    assert len(report["history"]) == 2
    assert report["history"][0]["next_move_mode"] == "one_shot"
    np.testing.assert_allclose(
        report["history"][0]["command_translation_mm"], 150.0
    )
    assert len(controller.moves) == 2
    np.testing.assert_allclose(controller.moves[1][0][:3, 3], [-0.150, 0.0, 0.0])
    np.testing.assert_array_equal(
        np.load(tmp_path / "output/Right/7_qpos.npy"),
        np.arange(6, dtype=np.float64),
    )
