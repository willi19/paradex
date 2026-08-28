"""Immediate, read-only integrity checks for a freshly stopped capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_array(path):
    try:
        return np.load(path, allow_pickle=False)
    except ValueError:
        # Several robot controllers intentionally save object arrays.
        return np.load(path, allow_pickle=True)


def _check_time_array(path, array, errors, warnings):
    try:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        errors.append(f"{path}: time values are not numeric")
        return
    if not np.all(np.isfinite(values)):
        errors.append(f"{path}: time values contain NaN/Inf")
        return
    if len(values) > 1:
        differences = np.diff(values)
        if np.any(differences < 0):
            errors.append(f"{path}: time values move backwards")
        elif path.name == "timestamp.npy" and np.any(differences == 0):
            warnings.append(f"{path}: duplicate time values detected")


def _sample_length(path, array, errors):
    if array.ndim > 0:
        return int(len(array)), {}
    try:
        value = array.item()
    except ValueError:
        errors.append(f"{path}: scalar array, expected samples")
        return None, {}
    if not isinstance(value, dict) or not value:
        errors.append(f"{path}: scalar array, expected samples")
        return None, {}
    lengths = {}
    for key, samples in value.items():
        samples = np.asarray(samples)
        if samples.ndim == 0:
            errors.append(f"{path}: mapping entry {key!r} is scalar")
            return None, {}
        lengths[str(key)] = int(len(samples))
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        detail = ", ".join(
            f"{name}={length}" for name, length in sorted(lengths.items())
        )
        errors.append(f"{path}: mapping sample length mismatch ({detail})")
        return None, {"mapping_keys": len(lengths)}
    return unique_lengths.pop(), {"mapping_keys": len(lengths)}


def _check_stream(root, label, required_names, errors, warnings):
    root = Path(root)
    result = {"path": str(root), "files": {}, "length": None}
    if not root.is_dir():
        errors.append(f"{label}: missing directory {root}")
        return result

    paths = sorted(root.glob("*.npy"))
    existing_names = {path.name for path in paths}
    missing = sorted(set(required_names) - existing_names)
    if missing:
        errors.append(f"{label}: missing {', '.join(missing)}")

    lengths = {}
    for path in paths:
        try:
            array = _load_array(path)
        except Exception as exc:
            errors.append(f"{label}/{path.name}: unreadable ({exc})")
            continue
        length, extra = _sample_length(path, array, errors)
        if length is None:
            continue
        lengths[path.name] = length
        result["files"][path.name] = {
            "length": length,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            **extra,
        }
        if length == 0:
            errors.append(f"{label}/{path.name}: empty array")
        if (
            path.name in ("time.npy", "timestamp.npy")
            or path.stem.endswith("_time")
        ) and length:
            _check_time_array(path, array, errors, warnings)

    aligned_lengths = {
        name: lengths[name]
        for name in required_names
        if name in lengths
    }
    nonempty_lengths = {
        length for length in aligned_lengths.values() if length > 0
    }
    if len(nonempty_lengths) > 1:
        detail = ", ".join(
            f"{name}={length}"
            for name, length in sorted(aligned_lengths.items())
        )
        errors.append(f"{label}: sample length mismatch ({detail})")
    elif nonempty_lengths:
        result["length"] = nonempty_lengths.pop()
    return result


def _check_timestamps(root, errors, warnings):
    result = _check_stream(
        root,
        "timestamps",
        ("timestamp.npy", "frame_id.npy"),
        errors,
        warnings,
    )
    frame_path = Path(root) / "frame_id.npy"
    if not frame_path.is_file():
        return result
    try:
        frame_ids = np.asarray(_load_array(frame_path), dtype=np.int64).reshape(-1)
    except Exception as exc:
        errors.append(f"timestamps/frame_id.npy: invalid frame IDs ({exc})")
        return result
    if len(frame_ids) > 1:
        differences = np.diff(frame_ids)
        if np.any(differences <= 0):
            errors.append("timestamps/frame_id.npy: frame IDs are not increasing")
        else:
            dropped = int(np.sum(np.maximum(differences - 1, 0)))
            result["dropped_frame_ids"] = dropped
            if dropped:
                warnings.append(
                    f"timestamps: {dropped} trigger frame ID(s) missing"
                )
    return result


def validate_immediate_capture(
    episode_root,
    *,
    arm_enabled=True,
    hand_enabled=True,
    bimanual=False,
    teleop_enabled=True,
    camera_enabled=True,
    timestamp_expected=True,
):
    """Validate files available immediately after ``CaptureSession.stop``."""

    episode_root = Path(episode_root)
    raw_root = episode_root / "raw"
    errors = []
    warnings = []
    streams = {}

    if not raw_root.is_dir():
        errors.append(f"missing raw directory {raw_root}")

    if arm_enabled:
        arm_names = ("arm_left", "arm_right") if bimanual else ("arm",)
        for name in arm_names:
            streams[name] = _check_stream(
                raw_root / name,
                name,
                ("time.npy", "position.npy"),
                errors,
                warnings,
            )

    if hand_enabled:
        hand_names = ("hand_left", "hand_right") if bimanual else ("hand",)
        for name in hand_names:
            streams[name] = _check_stream(
                raw_root / name,
                name,
                ("time.npy", "position.npy"),
                errors,
                warnings,
            )

    if teleop_enabled:
        required = ["time.npy"]
        required.extend(("left.npy", "right.npy") if bimanual else ("right.npy",))
        streams["teleop"] = _check_stream(
            raw_root / "teleop",
            "teleop",
            required,
            errors,
            warnings,
        )
        streams["state"] = _check_stream(
            raw_root / "state",
            "state",
            ("state_hist.npy", "state_time.npy"),
            errors,
            warnings,
        )

    if timestamp_expected:
        streams["timestamps"] = _check_timestamps(
            raw_root / "timestamps", errors, warnings
        )

    metadata = {}
    if camera_enabled:
        cam_param = episode_root / "cam_param"
        metadata["cam_param"] = str(cam_param)
        if not cam_param.is_dir() or not any(cam_param.iterdir()):
            errors.append(f"camera metadata missing or empty: {cam_param}")
        if arm_enabled:
            c2r = episode_root / "C2R.npy"
            metadata["C2R"] = str(c2r)
            try:
                matrix = np.asarray(_load_array(c2r), dtype=np.float64)
                if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                    errors.append(f"C2R.npy: expected finite 4x4, got {matrix.shape}")
            except Exception as exc:
                errors.append(f"C2R.npy: unreadable ({exc})")

    videos_root = episode_root / "videos"
    videos = sorted(videos_root.glob("*.avi")) if videos_root.is_dir() else []
    if videos:
        empty_videos = [path.name for path in videos if path.stat().st_size <= 0]
        video_status = "available"
        if empty_videos:
            errors.append(f"empty uploaded video(s): {', '.join(empty_videos)}")
    elif camera_enabled:
        video_status = "upload_pending"
    else:
        video_status = "disabled"

    report = {
        "version": 1,
        "episode_root": str(episode_root),
        "immediate_ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "streams": streams,
        "metadata": metadata,
        "videos": {
            "status": video_status,
            "count": len(videos),
            "files": [path.name for path in videos],
        },
    }
    return report


def format_integrity_report(report):
    if not report["immediate_ok"]:
        lines = [
            "!" * 72,
            "!!!              CAPTURE INTEGRITY CHECK FAILED              !!!",
            "!!!  THIS EPISODE MAY BE INCOMPLETE OR CORRUPTED. RETAKE IT.  !!!",
            "!" * 72,
            f"Episode: {report['episode_root']}",
            f"Detected errors: {len(report['errors'])}",
            "-" * 72,
        ]
        for index, error in enumerate(report["errors"], start=1):
            lines.append(f"ERROR {index}: {error}")
        for warning in report["warnings"]:
            lines.append(f"WARN: {warning}")
        lines.extend(
            [
                "-" * 72,
                "The captured files were kept for diagnosis.",
                "Recommended action: verify devices, then recapture this episode.",
                "!!!                    INTEGRITY FAILED                    !!!",
                "!" * 72,
            ]
        )
        return "\n".join(lines)

    lines = ["Immediate capture integrity: PASS"]
    for name, stream in report["streams"].items():
        length = stream.get("length")
        lines.append(f"  {name}: {length if length is not None else 'invalid'} samples")
    for warning in report["warnings"]:
        lines.append(f"  WARN: {warning}")
    return "\n".join(lines)
