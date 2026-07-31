from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.validate.grasp import (
    CaptureKind,
    ContactPhase,
    ContactLossKind,
    EpisodePaths,
    EpisodeValidationReport,
    HandType,
    ValidationThresholds,
    discover_episode_paths,
    validate_dataset_episode,
)
from src.validate.grasp.episode_validation import (
    _evaluate_pregrasp_motion,
    _evaluate_states,
    _FrameState,
)
from src.validate.grasp.grasp_sequence import build_parser


def _make_object_mesh(mesh_root: Path, object_name: str) -> None:
    directory = mesh_root / object_name
    directory.mkdir(parents=True)
    trimesh.creation.box(extents=(0.08, 0.08, 0.08)).export(
        directory / f"{object_name}.obj"
    )


def _write_object_poses(path: Path, poses: np.ndarray) -> None:
    np.savez(path, **{f"frame_{index}": pose for index, pose in enumerate(poses)})


def _load_test_poses(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.stack(
            [payload[f"frame_{index}"] for index in range(len(payload.files))]
        )


def _make_human_episode(
    capture_root: Path, object_name: str, episode: int, frame_count: int = 20
) -> None:
    root = capture_root / "human" / object_name / str(episode)
    mano = root / "hand" / "mano"
    params = root / "hand" / "mano_params"
    mano.mkdir(parents=True)
    params.mkdir(parents=True)

    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.001
    poses[:, 2, 3] = 1.0
    _write_object_poses(root / "object_6d_pose.npz", poses)
    np.save(root / "C2R.npy", np.eye(4))

    base_vertices = np.zeros((778, 3), dtype=float)
    base_vertices[:, 0] = 0.2
    base_vertices[:, 2] = 1.0
    base_vertices[0:3] = np.array(
        [[0.04, 0.0, 1.0], [0.04, 0.01, 1.0], [0.04, 0.0, 1.01]]
    )
    base_vertices[744] = [0.04, 0.0, 1.0]
    base_vertices[320] = [0.04, 0.01, 1.0]
    base_vertices[443] = [0.04, -0.01, 1.0]
    # Reference every vertex so OBJ round-tripping preserves MANO's 778-index
    # topology contract even though this is only a tiny synthetic fixture.
    faces = np.asarray([[0, index, index + 1] for index in range(1, 777)])

    for frame in range(frame_count):
        translation = np.array([frame * 0.001, 0.0, 0.0])
        hand_mesh = trimesh.Trimesh(
            vertices=base_vertices + translation,
            faces=faces,
            process=False,
        )
        hand_mesh.export(mano / f"{frame:05d}.obj")
        joints = np.zeros((21, 3), dtype=float)
        joints[0] = translation
        (params / f"{frame:05d}.json").write_text(
            json.dumps(
                {
                    "global_orient": np.eye(3).reshape(1, 1, 3, 3).tolist(),
                    "joints": joints.tolist(),
                }
            ),
            encoding="utf-8",
        )


def test_discovery_uses_explicit_hand_to_select_overlapping_episode(
    tmp_path: Path,
) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    _make_object_mesh(mesh_root, "apple")

    robot = capture_root / "allegro_v5" / "apple" / "3"
    for relative in (
        "raw/hand/position.npy",
        "raw/hand/time.npy",
        "raw/arm/position.npy",
        "raw/arm/time.npy",
        "raw/timestamps/timestamp.npy",
        "C2R.npy",
        "object_6d_pose.npz",
    ):
        path = robot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    human = capture_root / "human" / "apple" / "3"
    (human / "hand" / "mano").mkdir(parents=True)
    (human / "hand" / "mano_params").mkdir()
    (human / "object_6d_pose.npz").touch()
    (human / "C2R.npy").touch()

    robot_paths = discover_episode_paths(
        "apple",
        3,
        "allegro",
        mesh_root=mesh_root,
        capture_root=capture_root,
    )
    human_paths = discover_episode_paths(
        "apple",
        3,
        "human",
        mesh_root=mesh_root,
        capture_root=capture_root,
    )

    assert robot_paths.kind is CaptureKind.ROBOT
    assert robot_paths.hand_trajectory == robot / "raw" / "hand"
    assert human_paths.kind is CaptureKind.HUMAN
    assert human_paths.hand_trajectory == human / "hand"


def test_single_episode_cli_help_excludes_projection_options() -> None:
    help_text = build_parser().format_help()

    assert "projection" not in help_text
    assert "max-object-position-jump" in help_text


@pytest.mark.parametrize(
    ("hand", "source", "state_file", "time_file", "canonical"),
    (
        ("allegro_v5", "allegro_v5", "position.npy", "time.npy", HandType.ALLEGRO),
        ("inspire", "inspire_dftp", "position.npy", "time.npy", HandType.INSPIRE),
        (
            "inspire_f1",
            "inspire_f1",
            "right_joint_states.npy",
            "right_joint_states_time.npy",
            HandType.INSPIRE_F1,
        ),
    ),
)
def test_robot_hand_names_map_to_dataset_layout(
    tmp_path: Path,
    hand: str,
    source: str,
    state_file: str,
    time_file: str,
    canonical: HandType,
) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    _make_object_mesh(mesh_root, "apple")
    root = capture_root / source / "apple" / "2"
    required = (
        root / "raw" / "hand" / state_file,
        root / "raw" / "hand" / time_file,
        root / "raw" / "arm" / "position.npy",
        root / "raw" / "arm" / "time.npy",
        root / "raw" / "timestamps" / "timestamp.npy",
        root / "C2R.npy",
        root / "object_6d_pose.npz",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    paths = discover_episode_paths(
        "apple",
        2,
        hand,
        mesh_root=mesh_root,
        capture_root=capture_root,
    )

    assert paths.hand is canonical
    assert paths.episode_root == root


@pytest.mark.parametrize("object_name", ("../apple", "apple/3", "..", ""))
def test_discovery_rejects_path_traversal(
    tmp_path: Path, object_name: str
) -> None:
    with pytest.raises(ValueError):
        discover_episode_paths(
            object_name,
            0,
            "human",
            mesh_root=tmp_path / "mesh",
            capture_root=tmp_path / "capture",
        )


def test_discovery_uses_v1_when_unsuffixed_pose_is_absent(
    tmp_path: Path,
) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    _make_object_mesh(mesh_root, "apple")
    root = capture_root / "human" / "apple" / "0"
    (root / "hand" / "mano").mkdir(parents=True)
    (root / "hand" / "mano_params").mkdir()
    (root / "C2R.npy").touch()
    (root / "object_6d_pose_v1.npz").touch()
    (root / "object_6d_pose_v2.npz").touch()

    paths = discover_episode_paths(
        "apple",
        0,
        "human",
        mesh_root=mesh_root,
        capture_root=capture_root,
    )

    assert paths.object_poses == root / "object_6d_pose_v1.npz"


def test_discovery_rejects_v2_only_pose(
    tmp_path: Path,
) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    _make_object_mesh(mesh_root, "apple")
    root = capture_root / "human" / "apple" / "0"
    (root / "hand" / "mano").mkdir(parents=True)
    (root / "hand" / "mano_params").mkdir()
    (root / "C2R.npy").touch()
    (root / "object_6d_pose_v2.npz").touch()

    with pytest.raises(FileNotFoundError, match="non-v2 object pose"):
        discover_episode_paths(
            "apple",
            0,
            "human",
            mesh_root=mesh_root,
            capture_root=capture_root,
        )


def test_human_episode_validation_combines_gravity_and_jump_checks(
    tmp_path: Path,
) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    _make_object_mesh(mesh_root, "cube")
    _make_human_episode(capture_root, "cube", 7)

    report = validate_dataset_episode(
        "cube",
        7,
        "human",
        mesh_root=mesh_root,
        capture_root=capture_root,
        thresholds=ValidationThresholds(
            max_frames=20,
            contact_distance_m=0.02,
            min_contact_fingers=1,
            min_grasp_frames=10,
            min_object_motion_m=0.01,
        ),
    )

    assert report.valid
    assert report.contact_phase is not None
    assert report.sampled_frame_count == 20
    assert report.projection_alignment is None
    assert report.metrics["gravity_check_status"] == (
        "not_applicable_no_persistent_contact_loss"
    )
    assert report.metrics["object_position_jump_count"] == 0


def test_smooth_pose_change_is_not_checked_by_projection(
    tmp_path: Path,
) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    _make_object_mesh(mesh_root, "cube")
    _make_human_episode(capture_root, "cube", 7)
    pose_path = (
        capture_root / "human" / "cube" / "7" / "object_6d_pose.npz"
    )
    poses = _load_test_poses(pose_path)
    poses[:, 0, 3] += np.linspace(0.0, 0.5, len(poses))
    _write_object_poses(pose_path, poses)
    overlay_dir = tmp_path / "projection_overlays"

    report = validate_dataset_episode(
        "cube",
        7,
        "human",
        mesh_root=mesh_root,
        capture_root=capture_root,
        thresholds=ValidationThresholds(
            max_frames=20,
            contact_distance_m=0.02,
            min_grasp_frames=1_000,
        ),
        projection_overlay_dir=overlay_dir,
    )

    assert report.valid
    assert report.projection_alignment is None
    assert not any(
        issue.code == "OBJECT_POSITION_JUMP" for issue in report.issues
    )
    assert not overlay_dir.exists()


def test_excessive_pregrasp_translation_is_invalid() -> None:
    poses = np.repeat(np.eye(4)[None, ...], 60, axis=0)
    poses[20:, 0, 3] = 0.08
    phase = ContactPhase(
        start_frame=45,
        end_frame=59,
        start_sample=45,
        end_sample=59,
        sampled_frame_count=15,
    )

    metrics, issues = _evaluate_pregrasp_motion(
        poses,
        phase,
        ValidationThresholds(),
    )

    assert metrics["pregrasp_motion_status"] == "excessive"
    assert metrics["pregrasp_last_evaluated_frame"] == 29
    assert metrics["max_pregrasp_translation_m"] == pytest.approx(0.08)
    assert {issue.code for issue in issues} == {
        "OBJECT_PRE_GRASP_TRANSLATION_EXCESSIVE"
    }


def test_excessive_pregrasp_rotation_is_invalid() -> None:
    poses = np.repeat(np.eye(4)[None, ...], 60, axis=0)
    angle = np.deg2rad(150.0)
    poses[20:, 0, 0] = np.cos(angle)
    poses[20:, 0, 1] = -np.sin(angle)
    poses[20:, 1, 0] = np.sin(angle)
    poses[20:, 1, 1] = np.cos(angle)
    phase = ContactPhase(
        start_frame=45,
        end_frame=59,
        start_sample=45,
        end_sample=59,
        sampled_frame_count=15,
    )

    metrics, issues = _evaluate_pregrasp_motion(
        poses,
        phase,
        ValidationThresholds(),
    )

    assert metrics["pregrasp_motion_status"] == "excessive"
    assert metrics["max_pregrasp_rotation_deg"] == pytest.approx(150.0)
    assert {issue.code for issue in issues} == {
        "OBJECT_PRE_GRASP_ROTATION_EXCESSIVE"
    }


def test_pregrasp_grace_interval_ignores_immediate_grasp_approach() -> None:
    poses = np.repeat(np.eye(4)[None, ...], 60, axis=0)
    poses[35:, 0, 3] = 0.08
    phase = ContactPhase(
        start_frame=45,
        end_frame=59,
        start_sample=45,
        end_sample=59,
        sampled_frame_count=15,
    )

    metrics, issues = _evaluate_pregrasp_motion(
        poses,
        phase,
        ValidationThresholds(),
    )

    assert metrics["pregrasp_motion_status"] == "normal"
    assert metrics["pregrasp_last_evaluated_frame"] == 29
    assert metrics["max_pregrasp_translation_m"] == pytest.approx(0.0)
    assert issues == []


def test_pregrasp_motion_is_not_applicable_without_stable_grasp() -> None:
    poses = np.repeat(np.eye(4)[None, ...], 20, axis=0)

    metrics, issues = _evaluate_pregrasp_motion(
        poses,
        None,
        ValidationThresholds(),
    )

    assert metrics["pregrasp_motion_status"] == (
        "not_applicable_no_stable_grasp"
    )
    assert issues == []


def test_abrupt_object_position_jump_invalidates_episode(
    tmp_path: Path,
) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    _make_object_mesh(mesh_root, "cube")
    _make_human_episode(capture_root, "cube", 7)
    pose_path = (
        capture_root / "human" / "cube" / "7" / "object_6d_pose.npz"
    )
    poses = _load_test_poses(pose_path)
    poses[10, 0, 3] += 0.5
    _write_object_poses(pose_path, poses)

    report = validate_dataset_episode(
        "cube",
        7,
        "human",
        mesh_root=mesh_root,
        capture_root=capture_root,
        thresholds=ValidationThresholds(
            max_frames=20,
            contact_distance_m=0.02,
            min_grasp_frames=1_000,
        ),
    )

    assert not report.valid
    assert report.projection_alignment is None
    assert report.metrics["object_position_jump_count"] == 2
    assert report.metrics["object_position_jump_frames"] == [10, 11]
    assert {
        issue.code for issue in report.issues
    } == {"OBJECT_POSITION_JUMP"}


def test_post_release_gravity_issue_is_part_of_combined_validity(
    tmp_path: Path,
) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    _make_object_mesh(mesh_root, "cube")
    _make_human_episode(capture_root, "cube", 7)
    episode_root = capture_root / "human" / "cube" / "7"
    pose_path = episode_root / "object_6d_pose.npz"
    poses = _load_test_poses(pose_path)
    lift = np.minimum(np.arange(len(poses)), 9) * 0.02
    poses[:, 2, 3] += lift
    _write_object_poses(pose_path, poses)

    mano_dir = episode_root / "hand" / "mano"
    for frame, z_offset in enumerate(lift):
        hand_mesh = trimesh.load(
            mano_dir / f"{frame:05d}.obj",
            force="mesh",
            process=False,
        )
        hand_mesh.apply_translation(
            [
                0.0 if frame < 10 else 0.3,
                0.0,
                z_offset,
            ]
        )
        hand_mesh.export(mano_dir / f"{frame:05d}.obj")

    report = validate_dataset_episode(
        "cube",
        7,
        "human",
        mesh_root=mesh_root,
        capture_root=capture_root,
        thresholds=ValidationThresholds(
            max_frames=20,
            contact_distance_m=0.02,
        ),
    )

    assert not report.valid
    assert report.projection_alignment is None
    assert report.metrics["object_position_jump_count"] == 0
    assert report.metrics["gravity_check_status"] == "tracking_error"
    assert {
        issue.code for issue in report.issues
    } == {"OBJECT_POST_CONTACT_GRAVITY_INCONSISTENT"}


def test_negative_episode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        discover_episode_paths(
            "apple",
            -1,
            "human",
            mesh_root=tmp_path / "mesh",
            capture_root=tmp_path / "capture",
        )


def test_relative_object_slip_is_diagnostic_only(tmp_path: Path) -> None:
    paths = EpisodePaths(
        kind=CaptureKind.HUMAN,
        hand=HandType.HUMAN,
        object_name="cube",
        episode=0,
        episode_root=tmp_path,
        object_mesh=tmp_path / "cube.obj",
        object_poses=tmp_path / "object_6d_pose.npz",
        hand_trajectory=tmp_path / "hand",
    )
    states: list[_FrameState] = []
    frame_count = 20
    for frame in range(frame_count):
        angle = np.deg2rad(90.0 * frame / (frame_count - 1))
        cosine, sine = np.cos(angle), np.sin(angle)
        object_pose = np.eye(4)
        object_pose[:3, :3] = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        object_pose[0, 3] = frame * 0.01
        states.append(
            _FrameState(
                frame=frame,
                time_s=float(frame),
                contact_fingers=("thumb", "index"),
                min_hand_object_distance_m=0.0,
                world_from_reference=np.eye(4),
                world_from_object=object_pose,
            )
        )

    report = _evaluate_states(
        paths,
        states,
        frame_count,
        ValidationThresholds(max_frames=frame_count),
    )

    assert report.valid
    assert report.issues == []
    assert report.metrics["max_relative_translation_m"] == pytest.approx(0.19)
    assert report.metrics["max_relative_rotation_deg"] == pytest.approx(90.0)


def test_single_finger_contact_defines_grasp_phase(tmp_path: Path) -> None:
    frame_count = 20
    paths = EpisodePaths(
        kind=CaptureKind.HUMAN,
        hand=HandType.HUMAN,
        object_name="cube",
        episode=0,
        episode_root=tmp_path,
        object_mesh=tmp_path / "cube.obj",
        object_poses=tmp_path / "object_6d_pose.npz",
        hand_trajectory=tmp_path / "hand",
    )
    states = [
        _FrameState(
            frame=frame,
            time_s=frame / 30.0,
            contact_fingers=("thumb",) if 5 <= frame <= 14 else (),
            min_hand_object_distance_m=0.0 if 5 <= frame <= 14 else 0.1,
            world_from_reference=np.eye(4),
            world_from_object=np.eye(4),
        )
        for frame in range(frame_count)
    ]

    report = _evaluate_states(
        paths,
        states,
        frame_count,
        ValidationThresholds(max_frames=frame_count),
    )

    assert report.contact_phase is not None
    assert report.contact_phase.start_frame == 5
    assert report.contact_phase.end_frame == 14


def test_contact_finger_threshold_is_fixed_to_one() -> None:
    with pytest.raises(ValueError, match="fixed to 1"):
        ValidationThresholds(min_contact_fingers=2)


def _evaluate_contact_loss(
    tmp_path: Path,
    object_poses: np.ndarray,
    object_vertices: np.ndarray | None = None,
) -> EpisodeValidationReport:
    frame_count = len(object_poses)
    paths = EpisodePaths(
        kind=CaptureKind.HUMAN,
        hand=HandType.HUMAN,
        object_name="cube",
        episode=0,
        episode_root=tmp_path,
        object_mesh=tmp_path / "cube.obj",
        object_poses=tmp_path / "object_6d_pose.npz",
        hand_trajectory=tmp_path / "hand",
    )
    states = [
        _FrameState(
            frame=frame,
            time_s=frame / 30.0,
            contact_fingers=(
                ("thumb", "index") if frame < 15 else ()
            ),
            min_hand_object_distance_m=0.0 if frame < 15 else 0.1,
            world_from_reference=np.eye(4),
            world_from_object=object_poses[frame],
        )
        for frame in range(frame_count)
    ]
    return _evaluate_states(
        paths,
        states,
        frame_count,
        ValidationThresholds(max_frames=frame_count),
        dense_object_poses=object_poses,
        dense_object_times=np.arange(frame_count, dtype=float) / 30.0,
        object_vertices=object_vertices,
    )


def test_contact_loss_with_normal_object_motion_does_not_invalidate(
    tmp_path: Path,
) -> None:
    frame_count = 30
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002
    released_time = np.maximum(np.arange(frame_count) - 14, 0) / 30.0
    poses[:, 2, 3] = -0.5 * 9.81 * released_time**2

    report = _evaluate_contact_loss(tmp_path, poses)

    assert report.valid
    assert report.issues == []
    assert report.contact_loss_event is not None
    assert report.contact_loss_event.kind is ContactLossKind.NORMAL_MOTION
    assert report.contact_loss_event.frame == 15
    assert report.contact_loss_event.gravity_displacement_m > 0.010
    assert report.contact_loss_event.gravity_velocity_change_m_s > 0.05


def test_lateral_velocity_and_rotation_are_allowed_when_gravity_is_present(
    tmp_path: Path,
) -> None:
    frame_count = 30
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002
    released_time = np.maximum(np.arange(frame_count) - 14, 0) / 30.0
    poses[:, 0, 3] += 2.0 * released_time
    poses[:, 2, 3] = -0.5 * 9.81 * released_time**2
    angles = np.linspace(0.0, np.pi, frame_count)
    poses[:, :3, :3] = np.stack(
        [
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
            for angle in angles
        ]
    )

    report = _evaluate_contact_loss(tmp_path, poses)

    assert report.valid
    assert report.issues == []
    assert report.contact_loss_event is not None
    assert report.contact_loss_event.kind is ContactLossKind.NORMAL_MOTION


@pytest.mark.parametrize("post_loss_motion", ("floating", "constant_upward"))
def test_motion_without_gravity_effect_after_contact_loss_is_tracking_error(
    tmp_path: Path,
    post_loss_motion: str,
) -> None:
    frame_count = 30
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002
    released_time = np.maximum(np.arange(frame_count) - 14, 0) / 30.0
    if post_loss_motion == "constant_upward":
        poses[:, 2, 3] = released_time

    report = _evaluate_contact_loss(tmp_path, poses)

    assert not report.valid
    assert report.contact_loss_event is not None
    assert report.contact_loss_event.kind is ContactLossKind.TRACKING_ERROR
    assert {issue.code for issue in report.issues} == {
        "OBJECT_POST_CONTACT_GRAVITY_INCONSISTENT",
    }
    assert "minimum acceptance thresholds" in report.issues[0].message


def test_upward_throw_is_allowed_when_velocity_trends_toward_gravity(
    tmp_path: Path,
) -> None:
    frame_count = 30
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002
    released_time = np.maximum(np.arange(frame_count) - 14, 0) / 30.0
    poses[:, 2, 3] = (
        3.0 * released_time - 0.5 * 9.81 * released_time**2
    )

    report = _evaluate_contact_loss(tmp_path, poses)

    assert report.valid
    assert report.contact_loss_event is not None
    assert report.contact_loss_event.kind is ContactLossKind.NORMAL_MOTION
    assert report.contact_loss_event.gravity_displacement_m < 0.0
    assert report.contact_loss_event.gravity_velocity_change_m_s > 0.05


def test_short_post_loss_observation_does_not_invalidate(
    tmp_path: Path,
) -> None:
    frame_count = 17
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002

    report = _evaluate_contact_loss(tmp_path, poses)

    assert report.valid
    assert report.contact_loss_event is not None
    assert (
        report.contact_loss_event.kind
        is ContactLossKind.INSUFFICIENT_OBSERVATION
    )


def test_floor_supported_object_does_not_need_to_fall(
    tmp_path: Path,
) -> None:
    frame_count = 30
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002
    vertices = np.asarray(trimesh.creation.box(extents=(0.08,) * 3).vertices)

    report = _evaluate_contact_loss(tmp_path, poses, vertices)

    assert report.valid
    assert report.issues == []
    assert report.contact_loss_event is not None
    assert (
        report.contact_loss_event.kind
        is ContactLossKind.FLOOR_SUPPORTED
    )
    assert report.contact_loss_event.floor_height_at_loss_m == pytest.approx(0.0)
    assert report.contact_loss_event.floor_contact_frame == 14


def test_airborne_object_that_never_reaches_floor_is_tracking_error(
    tmp_path: Path,
) -> None:
    frame_count = 60
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002
    poses[:15, 2, 3] = np.linspace(0.0, 0.30, 15)
    poses[15:, 2, 3] = 0.30
    vertices = np.asarray(trimesh.creation.box(extents=(0.08,) * 3).vertices)

    report = _evaluate_contact_loss(tmp_path, poses, vertices)

    assert not report.valid
    assert report.contact_loss_event is not None
    assert report.contact_loss_event.kind is ContactLossKind.TRACKING_ERROR
    assert report.contact_loss_event.floor_contact_frame is None
    assert report.contact_loss_event.floor_height_at_loss_m == pytest.approx(0.30)
    assert {issue.code for issue in report.issues} == {
        "OBJECT_DID_NOT_REACH_INFERRED_FLOOR",
    }
    assert report.issues[0].threshold == pytest.approx(0.2)


def test_falling_object_may_stop_after_reaching_floor(
    tmp_path: Path,
) -> None:
    frame_count = 60
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002
    poses[:15, 2, 3] = np.linspace(0.0, 0.15, 15)
    released_time = np.maximum(np.arange(frame_count) - 14, 0) / 30.0
    poses[15:, 2, 3] = np.maximum(
        0.15 - 0.5 * 9.81 * released_time[15:] ** 2,
        0.0,
    )
    vertices = np.asarray(trimesh.creation.box(extents=(0.08,) * 3).vertices)

    report = _evaluate_contact_loss(tmp_path, poses, vertices)

    assert report.valid
    assert report.issues == []
    assert report.contact_loss_event is not None
    assert report.contact_loss_event.kind is ContactLossKind.NORMAL_MOTION
    assert report.contact_loss_event.floor_contact_frame is not None
    assert report.contact_loss_event.minimum_floor_height_m == pytest.approx(0.0)


def test_configured_gravity_axis_is_used(tmp_path: Path) -> None:
    frame_count = 30
    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.002
    released_time = np.maximum(np.arange(frame_count) - 14, 0) / 30.0
    poses[:, 1, 3] = -0.5 * 9.81 * released_time**2

    paths = EpisodePaths(
        kind=CaptureKind.HUMAN,
        hand=HandType.HUMAN,
        object_name="cube",
        episode=0,
        episode_root=tmp_path,
        object_mesh=tmp_path / "cube.obj",
        object_poses=tmp_path / "object_6d_pose.npz",
        hand_trajectory=tmp_path / "hand",
    )
    states = [
        _FrameState(
            frame=frame,
            time_s=frame / 30.0,
            contact_fingers=("thumb", "index") if frame < 15 else (),
            min_hand_object_distance_m=0.0 if frame < 15 else 0.1,
            world_from_reference=np.eye(4),
            world_from_object=poses[frame],
        )
        for frame in range(frame_count)
    ]
    report = _evaluate_states(
        paths,
        states,
        frame_count,
        ValidationThresholds(max_frames=frame_count),
        dense_object_poses=poses,
        dense_object_times=np.arange(frame_count, dtype=float) / 30.0,
        gravity_direction=np.array([0.0, -1.0, 0.0]),
    )

    assert report.valid
    assert report.contact_loss_event is not None
    assert report.contact_loss_event.gravity_direction == (0.0, -1.0, 0.0)
