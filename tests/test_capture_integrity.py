from pathlib import Path

import numpy as np

from paradex.dataset_acqusition.capture_integrity import (
    format_integrity_report,
    validate_immediate_capture,
)


def _write_stream(root: Path, length=5):
    root.mkdir(parents=True)
    np.save(root / "time.npy", np.arange(length, dtype=float) * 0.1)
    np.save(root / "position.npy", np.zeros((length, 6)))


def _make_capture(root: Path):
    _write_stream(root / "raw" / "arm")
    _write_stream(root / "raw" / "hand")
    teleop = root / "raw" / "teleop"
    teleop.mkdir(parents=True)
    np.save(teleop / "time.npy", np.arange(5, dtype=float) * 0.1)
    np.save(teleop / "right.npy", np.zeros(5, dtype=object))
    state = root / "raw" / "state"
    state.mkdir(parents=True)
    np.save(state / "state_hist.npy", np.zeros(5))
    np.save(state / "state_time.npy", np.arange(5, dtype=float) * 0.1)
    timestamps = root / "raw" / "timestamps"
    timestamps.mkdir(parents=True)
    np.save(timestamps / "timestamp.npy", np.arange(5, dtype=float) * 0.1)
    np.save(timestamps / "frame_id.npy", np.arange(10, 15))
    (root / "cam_param").mkdir()
    (root / "cam_param" / "intrinsics.json").write_text("{}")
    np.save(root / "C2R.npy", np.eye(4))


def test_valid_immediate_capture_reports_video_upload_pending(tmp_path):
    _make_capture(tmp_path)
    report = validate_immediate_capture(tmp_path)
    assert report["immediate_ok"]
    assert report["videos"]["status"] == "upload_pending"
    assert not any("video" in warning for warning in report["warnings"])
    assert report["streams"]["timestamps"]["length"] == 5


def test_corrupt_and_mismatched_stream_fails(tmp_path):
    _make_capture(tmp_path)
    np.save(tmp_path / "raw" / "arm" / "position.npy", np.zeros((3, 6)))
    (tmp_path / "raw" / "hand" / "time.npy").write_bytes(b"broken")
    report = validate_immediate_capture(tmp_path)
    assert not report["immediate_ok"]
    assert any("arm: sample length mismatch" in error for error in report["errors"])
    assert any("hand/time.npy: unreadable" in error for error in report["errors"])
    formatted = format_integrity_report(report)
    assert "CAPTURE INTEGRITY CHECK FAILED" in formatted
    assert "RETAKE IT" in formatted
    assert "INTEGRITY FAILED" in formatted
    assert len(formatted.splitlines()) >= 10


def test_timestamp_gaps_are_warning_not_failure(tmp_path):
    _make_capture(tmp_path)
    np.save(tmp_path / "raw" / "timestamps" / "frame_id.npy", [10, 11, 14, 15, 16])
    report = validate_immediate_capture(tmp_path)
    assert report["immediate_ok"]
    assert report["streams"]["timestamps"]["dropped_frame_ids"] == 2
    assert any("2 trigger frame ID(s) missing" in warning for warning in report["warnings"])


def test_vive_mapping_payload_uses_leaf_sample_length(tmp_path):
    _make_capture(tmp_path)
    right = {
        "wrist": np.repeat(np.eye(4)[None, ...], 5, axis=0),
        "index": np.repeat(np.eye(4)[None, ...], 5, axis=0),
    }
    np.save(tmp_path / "raw" / "teleop" / "right.npy", right)
    report = validate_immediate_capture(tmp_path)
    assert report["immediate_ok"]
    right_report = report["streams"]["teleop"]["files"]["right.npy"]
    assert right_report["length"] == 5
    assert right_report["mapping_keys"] == 2
