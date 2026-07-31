from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.validate.grasp.batch_grasp_sequence import (
    discover_batch_jobs,
    main,
    validate_output_path,
)


def _tree_snapshot(root: Path) -> list[tuple[str, bool, int, int]]:
    return [
        (
            str(path.relative_to(root)),
            path.is_dir(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    ]


def _make_object_mesh(mesh_root: Path, object_name: str) -> None:
    directory = mesh_root / object_name
    directory.mkdir(parents=True)
    trimesh.creation.box(extents=(0.08, 0.08, 0.08)).export(
        directory / f"{object_name}.obj"
    )


def _make_human_episode(
    capture_root: Path,
    object_name: str,
    episode: int,
    frame_count: int = 20,
) -> None:
    root = capture_root / "human" / object_name / str(episode)
    mano = root / "hand" / "mano"
    params = root / "hand" / "mano_params"
    mano.mkdir(parents=True)
    params.mkdir(parents=True)

    poses = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)
    poses[:, 0, 3] = np.arange(frame_count) * 0.001
    poses[:, 2, 3] = 1.0
    np.savez(
        root / "object_6d_pose.npz",
        **{f"frame_{index}": pose for index, pose in enumerate(poses)},
    )
    np.save(root / "C2R.npy", np.eye(4))

    vertices = np.zeros((778, 3), dtype=float)
    vertices[:, 0] = 0.2
    vertices[:, 2] = 1.0
    vertices[0:3] = np.array(
        [[0.04, 0.0, 1.0], [0.04, 0.01, 1.0], [0.04, 0.0, 1.01]]
    )
    vertices[744] = [0.04, 0.0, 1.0]
    vertices[320] = [0.04, 0.01, 1.0]
    vertices[443] = [0.04, -0.01, 1.0]
    faces = np.asarray([[0, index, index + 1] for index in range(1, 777)])

    for frame in range(frame_count):
        translation = np.array([frame * 0.001, 0.0, 0.0])
        trimesh.Trimesh(
            vertices=vertices + translation,
            faces=faces,
            process=False,
        ).export(mano / f"{frame:05d}.obj")
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


def test_discover_batch_jobs_finds_every_numeric_episode(tmp_path: Path) -> None:
    capture_root = tmp_path / "capture"
    expected = (
        ("allegro_v5", "apple", 3),
        ("human", "apple", 0),
        ("inspire_dftp", "banana", 2),
        ("inspire_f1", "apple", 4),
    )
    for hand, object_name, episode in expected:
        (capture_root / hand / object_name / str(episode)).mkdir(parents=True)
    (capture_root / "human" / "apple" / "notes").mkdir()

    jobs = discover_batch_jobs(
        capture_root,
        ("allegro_v5", "human", "inspire_dftp", "inspire_f1"),
    )

    assert tuple(job.key for job in jobs) == expected


def test_output_path_must_be_outside_source_roots(tmp_path: Path) -> None:
    mesh_root = tmp_path / "mesh"
    capture_root = tmp_path / "capture"
    mesh_root.mkdir()
    capture_root.mkdir()

    with pytest.raises(ValueError, match="source data root"):
        validate_output_path(
            mesh_root / "result.json",
            mesh_root=mesh_root,
            capture_root=capture_root,
        )
    with pytest.raises(ValueError, match="source data root"):
        validate_output_path(
            capture_root / "result.json",
            mesh_root=mesh_root,
            capture_root=capture_root,
        )

    assert validate_output_path(
        tmp_path / "result.json",
        mesh_root=mesh_root,
        capture_root=capture_root,
    ) == tmp_path / "result.json"


def test_batch_writes_one_file_and_preserves_source_data(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    mesh_root = source_root / "mesh"
    capture_root = source_root / "capture"
    output = tmp_path / "result.json"
    _make_object_mesh(mesh_root, "cube")
    _make_human_episode(capture_root, "cube", 7)
    (capture_root / "human" / "incomplete" / "2").mkdir(parents=True)
    before = _tree_snapshot(source_root)

    exit_code = main(
        [
            "--hands",
            "human",
            "--mesh-root",
            str(mesh_root),
            "--capture-root",
            str(capture_root),
            "--output",
            str(output),
            "--checkpoint-every",
            "1",
            "--pretty",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["complete"] is True
    assert payload["schema_version"] == 19
    assert payload["validity_criterion"] == (
        "pregrasp_motion_post_release_gravity_and_position_jump"
    )
    assert payload["excluded_validity_checks"] == [
        "OBJECT_SLIP_TRANSLATION",
        "OBJECT_SLIP_ROTATION",
        "NO_STABLE_CONTACT_PHASE",
        "GRASP_TOO_SHORT",
        "OBJECT_NOT_MOVED",
        "OBJECT_CAMERA_PROJECTION_MISMATCH",
        "ROBOT_SELF_COLLISION",
        "ROBOT_JOINT_LIMITS",
    ]
    assert payload["thresholds"]["max_object_position_jump_m"] == 0.1
    assert payload["thresholds"]["min_contact_fingers"] == 1
    assert payload["thresholds"]["max_pregrasp_translation_m"] == 0.05
    assert payload["thresholds"]["max_pregrasp_rotation_deg"] == 120.0
    assert payload["thresholds"]["pregrasp_baseline_frames"] == 15
    assert payload["thresholds"]["pregrasp_grace_frames"] == 15
    assert payload["thresholds"]["object_jump_local_factor"] == 8.0
    assert payload["thresholds"]["object_jump_window_frames"] == 5
    assert payload["thresholds"]["max_unreached_floor_height_m"] == 0.2
    assert payload["thresholds"]["min_gravity_displacement_m"] == 0.010
    assert payload["summary"] == {
        "total_discovered": 2,
        "completed": 2,
        "pending": 0,
        "valid": 1,
        "invalid": 0,
        "error": 1,
        "gravity_or_floor_mismatch": 0,
        "object_position_jump": 0,
        "excessive_pregrasp_motion": 0,
        "by_hand": {
            "human": {
                "completed": 2,
                "valid": 1,
                "invalid": 0,
                "error": 1,
                "gravity_or_floor_mismatch": 0,
                "object_position_jump": 0,
                "excessive_pregrasp_motion": 0,
            }
        },
    }
    assert [result["status"] for result in payload["results"]] == [
        "valid",
        "error",
    ]
    assert _tree_snapshot(source_root) == before
    assert list(tmp_path.glob(".result.json.*.tmp")) == []
