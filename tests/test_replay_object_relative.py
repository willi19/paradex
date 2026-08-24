import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "src" / "dataset_acquisition" / "hri" / "replay_object_relative.py"
SPEC = importlib.util.spec_from_file_location("replay_object_relative", MODULE_PATH)
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def test_relative_arm_actions_applies_object_translation_to_every_eef_target():
    source_object = np.eye(4)
    source_object[:3, 3] = [0.4, -0.1, 0.2]
    current_object = source_object.copy()
    current_object[:3, 3] += [0.05, 0.0, -0.02]
    source_actions = np.repeat(np.eye(4)[None], 2, axis=0)
    source_actions[0, :3, 3] = [0.5, 0.0, 0.3]
    source_actions[1, :3, 3] = [0.6, 0.1, 0.3]

    actual = replay.relative_arm_actions(source_object, current_object, source_actions)

    np.testing.assert_allclose(
        actual[:, :3, 3] - source_actions[:, :3, 3],
        np.tile([0.05, 0.0, -0.02], (2, 1)),
    )


def test_relative_arm_actions_preserves_object_relative_eef_transform():
    source_object = np.eye(4)
    current_object = np.eye(4)
    current_object[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    current_object[:3, 3] = [0.2, 0.1, 0.0]
    source_action = np.eye(4)[None]
    source_action[0, :3, 3] = [0.1, 0.0, 0.4]

    target_action = replay.relative_arm_actions(source_object, current_object, source_action)[0]

    np.testing.assert_allclose(np.linalg.inv(current_object) @ target_action, source_action[0])


def test_zoh_resample_holds_last_command():
    values = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    actual = replay._zoh_resample(np.array([1.0, 2.0, 3.0]), values, np.array([0.5, 1.5, 2.0, 2.9, 4.0]))
    np.testing.assert_array_equal(actual, [[1, 1], [1, 1], [2, 2], [2, 2], [3, 3]])


def test_capture_object6d_rpc_uses_one_shot_infer_and_saves_response(tmp_path, monkeypatch):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    args = type("Args", (), {
        "mesh_name": "apple",
        "rpc_addr": "tcp://pose:5570",
        "rpc_timeout_ms": 123,
        "debug_image_mode": "none",
    })()
    sent = {}

    def fake_rpc(addr, request, timeout_ms):
        sent.update(addr=addr, request=request, timeout_ms=timeout_ms)
        return {"object_6d": {"pose_world": np.eye(4).tolist(), "t_world": [0, 0, 0]}}

    monkeypatch.setattr(replay, "_send_rpc_once", fake_rpc)
    monkeypatch.setattr(replay, "_shared_rel_path", lambda _: "shared_data/capture/live")

    pose = replay._infer_capture_object6d_rpc(args, capture_dir)

    np.testing.assert_array_equal(pose, np.eye(4))
    assert sent == {
        "addr": "tcp://pose:5570",
        "request": {"command": "infer", "image_path": "shared_data/capture/live", "mesh_name": "apple"},
        "timeout_ms": 123,
    }
    assert json.loads((capture_dir / "object_6d.json").read_text()) == {
        "pose_world": np.eye(4).tolist(), "pose_left_cam": None, "R_world": None, "t_world": [0, 0, 0]
    }


def test_open_rpc_debug_images_opens_server_overlay_files(tmp_path, monkeypatch):
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    (debug_dir / "projection_1.png").touch()
    (debug_dir / "mask_overlay_2.jpg").touch()
    opened = []

    monkeypatch.setattr(replay.subprocess, "Popen", lambda command, **_: opened.append(command))

    assert replay._open_rpc_debug_images(debug_dir) == 2
    assert opened == [
        ["xdg-open", str(debug_dir / "mask_overlay_2.jpg")],
        ["xdg-open", str(debug_dir / "projection_1.png")],
    ]


def test_debug_projection_selects_the_camera_matching_pose_left_cam():
    pose_world = np.eye(4)
    pose_world[2, 3] = 1.0
    pose_left_cam = pose_world.copy()
    pose_left_cam[0, 3] = 0.2

    matching_extrinsic = np.eye(4)
    matching_extrinsic[0, 3] = 0.2
    other_extrinsic = np.eye(4)
    other_extrinsic[1, 3] = 0.5

    assert replay._debug_projection_camera_id(
        pose_world,
        pose_left_cam,
        {"left": matching_extrinsic, "other": other_extrinsic},
    ) == "left"


def test_local_debug_projection_writes_mesh_overlay(tmp_path):
    cv2 = pytest.importorskip("cv2")
    capture_dir = tmp_path / "capture"
    raw_images = capture_dir / "raw" / "images"
    raw_images.mkdir(parents=True)
    assert cv2.imwrite(str(raw_images / "left.png"), np.zeros((100, 100, 3), dtype=np.uint8))
    cam_param = capture_dir / "cam_param"
    cam_param.mkdir()
    (cam_param / "intrinsics.json").write_text(json.dumps({
        "left": {
            "original_intrinsics": [[80.0, 0.0, 50.0], [0.0, 80.0, 50.0], [0.0, 0.0, 1.0]],
            "dist_params": [],
        }
    }))
    (cam_param / "extrinsics.json").write_text(json.dumps({
        "left": np.eye(4)[:3].tolist(),
    }))
    mesh_path = tmp_path / "mesh_blender" / "apple" / "apple.obj"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("v -0.1 -0.1 0\nv 0.1 -0.1 0\nv 0.0 0.1 0\nf 1 2 3\n")
    pose = np.eye(4)
    pose[2, 3] = 1.0

    output = replay._write_local_projection_debug_image(
        capture_dir,
        mesh_name="apple",
        mesh_root_dir=tmp_path / "mesh_blender",
        pose_out={"pose_world": pose.tolist(), "pose_left_cam": pose.tolist()},
    )

    assert output == capture_dir / "debug" / "local_projection_apple_left.png"
    rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert rendered is not None
    assert np.count_nonzero(rendered) > 0


def test_capture_object6d_mesh_check_uses_viser_aligned_mesh(tmp_path):
    mesh = tmp_path / "mesh_blender" / "apple" / "apple_viser.obj"
    mesh.parent.mkdir(parents=True)
    mesh.touch()

    assert replay._resolve_capture_object6d_mesh("apple", tmp_path / "mesh_blender") == mesh


def test_preview_indices_keep_endpoints_when_downsampling():
    np.testing.assert_array_equal(replay._preview_indices(10, 3), [0, 4, 9])
    np.testing.assert_array_equal(replay._preview_indices(3, 10), [0, 1, 2])


def test_allegro_v5_actions_preserve_the_recorded_driver_order():
    action = np.arange(32, dtype=float).reshape(2, 16)
    np.testing.assert_array_equal(
        replay._as_allegro_v5_actions(action, label="test"),
        action,
    )
    with pytest.raises(ValueError, match="N, 16"):
        replay._as_allegro_v5_actions(np.zeros((2, 15)), label="test")


def test_episode_source_pose_prefers_frame_zero_npz_when_capture_txt_is_absent(tmp_path):
    np.savez(tmp_path / "object_6d_pose.npz", frame_0=np.eye(4))

    assert replay._episode_source_pose_path(tmp_path, None) == tmp_path / "object_6d_pose.npz"


def test_episode_source_pose_prefers_v2_over_v1_when_both_are_available(tmp_path):
    np.savez(tmp_path / "object_6d_pose.npz", frame_0=np.eye(4))
    np.savez(tmp_path / "object_6d_pose_v2.npz", frame_0=np.eye(4))

    assert replay._episode_source_pose_path(tmp_path, None) == tmp_path / "object_6d_pose_v2.npz"


def test_viser_object_pose_uses_inverse_of_viser_mesh_alignment(tmp_path):
    mesh_path = tmp_path / "apple_viser.obj"
    mesh_path.touch()
    viser_from_original = np.eye(4)
    viser_from_original[0, 3] = 0.01
    viser_from_original[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    np.save(tmp_path / "apple_viser_align.npy", viser_from_original)
    object_pose = np.eye(4)
    object_pose[:3, 3] = [0.5, 0.1, 0.2]

    actual = replay._viser_object_pose(object_pose, mesh_path, apply_mesh_alignment=True)

    np.testing.assert_allclose(actual @ viser_from_original, object_pose)


def test_cartesian_approach_uses_speed_bounded_translation_and_rotation():
    current = np.eye(4)
    target = np.eye(4)
    target[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    target[:3, 3] = [0.2, 0.0, 0.0]

    poses, seconds = replay._cartesian_approach_trajectory(
        current,
        target,
        linear_speed_mps=0.1,
        angular_speed_rps=np.pi / 4,
        min_seconds=0.5,
        rate_hz=10.0,
    )

    assert seconds == pytest.approx(2.0)
    assert poses.shape == (20, 4, 4)
    np.testing.assert_allclose(poses[-1], target, atol=1e-12)
    np.testing.assert_allclose(poses[0][:3, 3], [0.01, 0.0, 0.0])


def test_xarm_position_to_transform_matches_xarm_mm_and_rpy_convention():
    position = np.array([100.0, -200.0, 300.0, 0.0, 0.0, np.pi / 2])

    actual = replay._xarm_position_to_transform(position)

    np.testing.assert_allclose(actual[:3, 3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(
        actual[:3, :3],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_hand_approach_blends_current_hand_state_to_episode_start():
    current = np.zeros(16)
    target = np.full(16, 1.0)

    actual = replay._hand_approach_trajectory(current, target, frame_count=4)

    np.testing.assert_allclose(actual[:, 0], [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(actual[-1], target)


def test_hand_approach_slew_rate_reaches_each_joint_target_in_requested_duration():
    current = np.array([0.0, -1.0] + [0.0] * 14)
    target = np.array([1.0, 0.0] + [0.0] * 14)

    actual = replay._hand_approach_slew_rate(current, target, seconds=5.0)

    np.testing.assert_allclose(actual[:2], [0.2, 0.2])
    np.testing.assert_array_equal(actual[2:], np.zeros(14))


def test_stationary_arm_extension_keeps_the_replay_start_pose_for_hand_approach():
    pose = np.eye(4)[None]

    actual = replay._extend_stationary_approach(pose, required_frames=3)

    assert actual.shape == (3, 4, 4)
    np.testing.assert_allclose(actual, np.repeat(pose, 3, axis=0))


def test_preview_starts_at_the_live_arm_and_hand_state_before_interpolation():
    interpolated = np.ones((2, 22))
    arm = np.arange(6, dtype=float)
    hand = np.arange(16, dtype=float)

    actual = replay._prepend_preview_start_state(interpolated, arm, hand)

    assert actual.shape == (3, 22)
    np.testing.assert_array_equal(actual[0], np.concatenate((arm, hand)))
    np.testing.assert_array_equal(actual[1:], interpolated)


def test_preview_hand_approach_ends_exactly_at_the_episode_first_hand_pose():
    current = np.zeros(16)
    target = np.ones(16)
    interpolated = replay._hand_approach_trajectory(current, target, frame_count=4)
    joint_trajectory = np.zeros((4, 22))
    joint_trajectory[:, 6:] = interpolated

    actual = replay._prepend_preview_start_state(joint_trajectory, np.zeros(6), current)

    np.testing.assert_array_equal(actual[0, 6:], current)
    np.testing.assert_array_equal(actual[-1, 6:], target)
