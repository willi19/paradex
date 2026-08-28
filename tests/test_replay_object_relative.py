import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import trimesh

from paradex.visualization.robot import simplify_mesh


REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "paradex" / "retargetor" / "experiment" / "replay_object_relative.py"
SPEC = importlib.util.spec_from_file_location("replay_object_relative", MODULE_PATH)
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def test_preview_mesh_simplification_caps_faces_without_changing_bounds():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.1)

    simplified = simplify_mesh(mesh, 300)

    assert len(simplified.faces) <= 300
    np.testing.assert_allclose(simplified.bounds, mesh.bounds, atol=2e-3)


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


def _episode(tmp_path, name, arm_poses, hand_commands=None):
    arm_poses = np.asarray(arm_poses, dtype=float)
    count = len(arm_poses)
    if hand_commands is None:
        hand_commands = np.zeros((count, 16), dtype=float)
    return replay.Episode(
        root=tmp_path / str(name),
        arm_poses=arm_poses,
        arm_times=np.arange(count, dtype=float),
        hand_commands=np.asarray(hand_commands, dtype=float),
        hand_times=np.arange(count, dtype=float),
        source_object_world=np.eye(4),
        source_c2r=np.eye(4),
    )


def test_episode_matching_uses_closest_distance_frame_then_relative_6d_pose(tmp_path):
    current_object = np.eye(4)
    current_wrist = np.eye(4)
    current_wrist[0, 3] = 0.3

    episode_a_poses = np.repeat(np.eye(4)[None], 2, axis=0)
    episode_a_poses[:, 0, 3] = [0.3, 0.1]
    episode_a_poses[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    episode_b_poses = np.repeat(np.eye(4)[None], 2, axis=0)
    episode_b_poses[:, 0, 3] = [0.3, 0.1]
    angle = np.deg2rad(10.0)
    episode_b_poses[0, :3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]

    match = replay._select_episode_match(
        current_object,
        current_wrist,
        [
            _episode(tmp_path, 1, episode_a_poses),
            _episode(tmp_path, 2, episode_b_poses),
        ],
        position_scale_m=0.05,
        rotation_scale_rad=0.5,
    )

    assert match.episode.root.name == "2"
    assert match.frame_index == 0
    assert match.grasp_frame_index == 1
    assert match.rotation_error_rad == pytest.approx(angle)


def test_episode_match_ranking_exposes_all_candidates_in_score_order(tmp_path):
    current_object = np.eye(4)
    current_wrist = np.eye(4)
    current_wrist[0, 3] = 0.3
    close = np.repeat(np.eye(4)[None], 2, axis=0)
    close[:, 0, 3] = [0.3, 0.1]
    far = np.repeat(np.eye(4)[None], 2, axis=0)
    far[:, 0, 3] = [-0.3, 0.1]

    ranked = replay._rank_episode_matches(
        current_object,
        current_wrist,
        [_episode(tmp_path, 1, far), _episode(tmp_path, 2, close)],
        position_scale_m=0.05,
        rotation_scale_rad=0.5,
    )

    assert [match.episode.root.name for match in ranked] == ["2", "1"]


def test_episode_matching_never_selects_a_frame_at_or_after_minimum_distance(tmp_path):
    current_object = np.eye(4)
    current_wrist = np.eye(4)
    current_wrist[0, 3] = 0.3
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = [0.5, 0.3, 0.1, 0.3]

    match = replay._select_episode_match(
        current_object,
        current_wrist,
        [_episode(tmp_path, 1, poses)],
        position_scale_m=0.05,
        rotation_scale_rad=0.5,
    )

    assert match.frame_index == 1
    assert match.grasp_frame_index == 2


def test_episode_remainder_starts_at_matched_frame_and_rebases_time(tmp_path):
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0.1, 0.2, 0.3]
    hands = np.repeat(np.arange(16, dtype=float)[None], 3, axis=0)
    hands += np.arange(3)[:, None]
    episode = _episode(tmp_path, 4, poses, hands)

    arm, hand, times = replay._episode_remainder(episode, 1)

    np.testing.assert_array_equal(arm, poses[1:])
    np.testing.assert_array_equal(hand, hands[1:])
    np.testing.assert_array_equal(times, [0.0, 1.0])


def test_candidate_episode_roots_prefer_successful_grasps(tmp_path):
    object_root = tmp_path / "allegro_v5" / "apple"
    for index in (1, 2, 3):
        (object_root / str(index)).mkdir(parents=True)
    (object_root / "1" / "grasp_result.json").write_text(json.dumps({"grasp_success": False}))
    (object_root / "2" / "grasp_result.json").write_text(json.dumps({"grasp_success": True}))

    roots = replay._candidate_episode_roots(tmp_path, "allegro_v5", "apple")

    assert roots == [object_root / "2"]


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


def test_capture_object6d_rpc_prints_and_saves_remote_timing(tmp_path, monkeypatch, capsys):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    args = type("Args", (), {
        "mesh_name": "apple",
        "rpc_addr": "tcp://pose:5570",
        "rpc_timeout_ms": 123,
        "debug_image_mode": "none",
    })()
    timing = {
        "runtime_setup": 0.1,
        "sam3": 3.42,
        "foundationpose": 0.84,
        "total_request": 6.42,
    }
    monkeypatch.setattr(
        replay,
        "_send_rpc_once",
        lambda *_args, **_kwargs: {
            "object_6d": {"pose_world": np.eye(4).tolist()},
            "timing_seconds": timing,
        },
    )
    monkeypatch.setattr(replay, "_shared_rel_path", lambda _: "shared_data/capture/live")

    replay._infer_capture_object6d_rpc(args, capture_dir)

    output = capsys.readouterr().out
    assert "[object6d timing] remote inference breakdown:" in output
    assert "sam3: 3.420 s" in output
    assert "total_request: 6.420 s" in output
    saved = json.loads((capture_dir / "object_6d.json").read_text())
    assert saved["timing_seconds"] == timing


def test_capture_object6d_rpc_save_mode_generates_projection(tmp_path, monkeypatch):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    args = type("Args", (), {
        "mesh_name": "apple",
        "mesh_root_dir": tmp_path / "meshes",
        "rpc_addr": "tcp://pose:5570",
        "rpc_timeout_ms": 123,
        "debug_image_mode": "save",
    })()
    projection = capture_dir / "debug" / "projection.png"
    calls = []

    monkeypatch.setattr(
        replay,
        "_send_rpc_once",
        lambda *_args, **_kwargs: {"object_6d": {"pose_world": np.eye(4).tolist()}},
    )
    monkeypatch.setattr(replay, "_shared_rel_path", lambda _: "shared_data/capture/live")
    monkeypatch.setattr(
        replay,
        "_write_local_projection_debug_image",
        lambda *args, **kwargs: calls.append((args, kwargs)) or projection,
    )
    monkeypatch.setattr(
        replay,
        "_open_rpc_debug_images",
        lambda *_: pytest.fail("save mode must not open a popup"),
    )

    replay._infer_capture_object6d_rpc(args, capture_dir)

    assert len(calls) == 1


def test_debug_image_mode_defaults_to_save():
    args = replay._parser().parse_args(["--object", "apple", "--plan-only"])

    assert args.debug_image_mode == "save"


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


def test_plan_only_state_read_does_not_start_teleoperation(monkeypatch):
    from threading import Event

    calls = []
    arm_pose = np.eye(4)
    arm_qpos = np.arange(6, dtype=float)
    hand_qpos = np.arange(16, dtype=float) / 10.0

    class FakeArm:
        def get_data(self):
            return {"position": arm_pose, "qpos": arm_qpos}

        def end(self):
            calls.append("arm.end")

    class FakeHand:
        connection_event = Event()
        connection_event.set()

        def get_data(self):
            return {"qpos": hand_qpos}

        def end(self):
            calls.append("hand.end")

    import paradex.io.robot_controller as controllers

    monkeypatch.setattr(controllers, "get_arm", lambda *args, **kwargs: FakeArm())

    def fake_get_hand(*args, **kwargs):
        calls.append(("get_hand", args, kwargs))
        return FakeHand()

    monkeypatch.setattr(controllers, "get_hand", fake_get_hand)

    actual_pose, actual_arm_qpos, actual_hand_qpos = replay._read_live_robot_state_once()

    np.testing.assert_array_equal(actual_pose, arm_pose)
    np.testing.assert_array_equal(actual_arm_qpos, arm_qpos)
    np.testing.assert_array_equal(actual_hand_qpos, hand_qpos)
    assert calls == [
        ("get_hand", ("allegro_v5",), {"hand_side": "right", "command_enabled": False}),
        "hand.end",
        "arm.end",
    ]


def test_live_state_read_from_open_controllers_does_not_close_them():
    from threading import Event

    arm_pose = np.eye(4)
    arm_qpos = np.arange(6, dtype=float)
    hand_qpos = np.arange(16, dtype=float) / 10.0

    class OpenArm:
        def get_data(self):
            return {"position": arm_pose, "qpos": arm_qpos}

        def end(self):
            raise AssertionError("state read must keep the arm controller open")

    class OpenHand:
        connection_event = Event()
        connection_event.set()

        def get_data(self):
            return {"qpos": hand_qpos}

        def end(self):
            raise AssertionError("state read must keep the hand controller open")

    actual = replay._read_live_robot_state(OpenArm(), OpenHand())

    np.testing.assert_array_equal(actual[0], arm_pose)
    np.testing.assert_array_equal(actual[1], arm_qpos)
    np.testing.assert_array_equal(actual[2], hand_qpos)


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


def test_allegro_v5_preview_maps_driver_actions_by_urdf_joint_name():
    action = np.arange(16, dtype=float)[None]
    names = tuple(reversed([f"joint_{index}_0" for index in range(16)]))
    limits = {name: (-100.0, 100.0) for name in names}

    actual = replay._allegro_v5_preview_qpos(
        action,
        urdf_hand_joint_names=names,
        joint_limits=limits,
    )

    np.testing.assert_array_equal(actual[0], np.arange(15, -1, -1))


def test_allegro_v5_preview_supports_semantic_thumb_first_urdf_order():
    action = np.arange(16, dtype=float)[None]
    names = (
        "thumb_base", "thumb_proximal", "thumb_medial", "thumb_distal",
        "index_base", "index_proximal", "index_medial", "index_distal",
        "middle_base", "middle_proximal", "middle_medial", "middle_distal",
        "ring_base", "ring_proximal", "ring_medial", "ring_distal",
    )
    limits = {name: (-100.0, 100.0) for name in names}

    actual = replay._allegro_v5_preview_qpos(
        action,
        urdf_hand_joint_names=names,
        joint_limits=limits,
    )

    np.testing.assert_array_equal(actual[0], np.r_[np.arange(12, 16), np.arange(12)])


def test_allegro_v5_preview_clamps_historical_commands_to_physical_limits():
    action = np.zeros((1, 16), dtype=float)
    action[0, 12:14] = [1.81, -2.1]
    names = tuple(f"joint_{index}_0" for index in range(16))
    limits = {name: (-10.0, 10.0) for name in names}
    limits["joint_12_0"] = (0.0, 1.78)
    limits["joint_13_0"] = (-0.26, 1.78)

    actual = replay._allegro_v5_preview_qpos(
        action,
        urdf_hand_joint_names=names,
        joint_limits=limits,
    )

    assert actual[0, 12] == pytest.approx(1.78)
    assert actual[0, 13] == pytest.approx(-0.26)


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


def test_composed_plan_contains_current_state_transition_and_episode_remainder():
    args = type(
        "Args",
        (),
        {
            "approach_linear_speed_mps": 0.1,
            "approach_angular_speed_rps": 0.5,
            "approach_min_seconds": 5.0,
            "approach_rate_hz": 10.0,
            "rate_scale": 2.0,
        },
    )()
    live_arm = np.eye(4)
    live_hand = np.zeros(16)
    episode_arm = np.repeat(np.eye(4)[None], 2, axis=0)
    episode_arm[:, 0, 3] = [0.1, 0.2]
    episode_hand = np.repeat(np.ones((1, 16)), 2, axis=0)

    trajectory = replay._compose_replay_trajectory(
        args,
        live_arm_pose=live_arm,
        live_hand_qpos=live_hand,
        episode_arm_poses=episode_arm,
        episode_hand_actions=episode_hand,
        episode_arm_times=np.array([0.0, 2.0]),
    )

    boundary = trajectory.transition_frame_count - 1
    np.testing.assert_array_equal(trajectory.arm_poses[0], live_arm)
    np.testing.assert_array_equal(trajectory.hand_actions[0], live_hand)
    np.testing.assert_allclose(trajectory.arm_poses[boundary], episode_arm[0])
    np.testing.assert_allclose(trajectory.hand_actions[boundary], episode_hand[0])
    np.testing.assert_allclose(trajectory.arm_poses[-1], episode_arm[-1])
    np.testing.assert_allclose(trajectory.hand_actions[-1], episode_hand[-1])
    assert trajectory.transition_seconds == pytest.approx(5.0)
    assert trajectory.times[boundary] == pytest.approx(5.0)
    assert trajectory.times[-1] == pytest.approx(6.0)
    assert len(trajectory.arm_poses) == len(trajectory.hand_actions) == len(trajectory.times)
    assert np.all(np.diff(trajectory.times) > 0)


def test_execute_streams_the_saved_combined_plan_without_regenerating_transition(monkeypatch):
    args = type("Args", (), {"settle_seconds": 0.25})()
    arm_poses = np.repeat(np.eye(4)[None], 4, axis=0)
    arm_poses[:, 0, 3] = [0.0, 0.1, 0.2, 0.3]
    hand_actions = np.arange(64, dtype=float).reshape(4, 16)
    arm_times = np.array([0.0, 0.1, 0.2, 0.5])
    moved_arm = []
    moved_hand = []
    sleeps = []

    class FakeArm:
        def move(self, pose):
            moved_arm.append(np.asarray(pose).copy())

        def is_error(self):
            return False

    class FakeHand:
        def move(self, action):
            moved_hand.append(np.asarray(action).copy())

        def is_error(self):
            return False

    monkeypatch.setattr(replay.time, "sleep", sleeps.append)

    replay._execute(
        args,
        arm_poses,
        hand_actions,
        arm_times,
        transition_frame_count=3,
        arm=FakeArm(),
        hand=FakeHand(),
    )

    np.testing.assert_array_equal(moved_arm, arm_poses)
    np.testing.assert_array_equal(moved_hand, hand_actions)
    np.testing.assert_allclose(sleeps, [0.1, 0.1, 0.25, 0.3])


def test_joint_return_trajectory_respects_speed_and_endpoints():
    current = np.zeros(6)
    target = np.array([0.2, -0.4, 0.1, 0.0, 0.3, -0.2])

    positions, times, seconds = replay._joint_return_trajectory(
        current,
        target,
        speed_rps=0.2,
        min_seconds=0.5,
        rate_hz=20.0,
    )

    assert seconds == pytest.approx(2.0)
    np.testing.assert_array_equal(positions[0], current)
    np.testing.assert_array_equal(positions[-1], target)
    assert np.all(np.diff(times) > 0)
    joint_speeds = np.abs(np.diff(positions, axis=0)) / np.diff(times)[:, None]
    assert np.max(joint_speeds) <= 0.2 + 1.0e-12


def test_execute_returns_from_live_post_release_joints_to_saved_initial(monkeypatch):
    args = type(
        "Args",
        (),
        {
            "settle_seconds": 0.0,
            "return_joint_speed_rps": 0.5,
            "return_min_seconds": 0.2,
            "return_rate_hz": 10.0,
        },
    )()
    arm_poses = np.repeat(np.eye(4)[None], 2, axis=0)
    hand_actions = np.zeros((2, 16))
    arm_times = np.array([0.0, 0.1])
    post_release_qpos = np.array([0.5, -0.2, 0.3, 0.0, 0.1, -0.4])
    initial_qpos = np.array([0.1, -0.1, 0.0, 0.2, -0.2, 0.1])

    class FakeArm:
        def __init__(self):
            self.return_trajectory = None

        def move(self, _pose):
            pass

        def get_data(self):
            return {"qpos": post_release_qpos.copy()}

        def move_joint_trajectory(self, positions, times):
            self.return_trajectory = (positions.copy(), times.copy())

        def is_error(self):
            return False

    class FakeHand:
        def move(self, _action):
            pass

        def is_error(self):
            return False

    arm = FakeArm()
    monkeypatch.setattr(replay.time, "sleep", lambda _seconds: None)

    replay._execute(
        args,
        arm_poses,
        hand_actions,
        arm_times,
        transition_frame_count=2,
        arm=arm,
        hand=FakeHand(),
        return_arm_qpos=initial_qpos,
    )

    positions, times = arm.return_trajectory
    np.testing.assert_array_equal(positions[0], post_release_qpos)
    np.testing.assert_array_equal(positions[-1], initial_qpos)
    assert np.all(np.diff(times) > 0)


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
