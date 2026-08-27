"""Physically measure shifted bimanual xArm hand-eye trajectories.

This script is intentionally separate from capture.py. It sends each original
single-arm xArm qpos through the real robot's Cartesian controller after a
base-frame Y offset, then records the actual qpos returned by the robot.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from paradex.utils.system import config_dir, network_info


LEFT_XARM_IP = "192.168.1.221"
RIGHT_Y_OFFSET_M = 0.32
LEFT_Y_OFFSET_M = -0.32
DEFAULT_SOURCE_DIR = Path(config_dir) / "hecalib" / "xarm"
DEFAULT_OUTPUT_DIR = Path(config_dir) / "hecalib" / "xarm_bimanual"


class SideConfig:
    def __init__(self, name: str, ip: str, y_offset_m: float):
        self.name = name
        self.ip = ip
        self.y_offset_m = y_offset_m


def right_ip_default() -> str:
    return network_info["xarm"]["param"]["ip"]


def create_controller(ip: str):
    from paradex.io.robot_controller.xarm_controller import XArmController

    return XArmController(ip=ip)


def cartesian_to_homo(cartesian: np.ndarray) -> np.ndarray:
    """Convert xArm SDK cartesian [mm, mm, mm, r, p, y] to a 4x4 pose in meters."""
    cartesian = np.asarray(cartesian, dtype=np.float64).reshape(-1)
    if cartesian.shape[0] < 6 or not np.all(np.isfinite(cartesian[:6])):
        raise ValueError(
            f"Invalid xArm FK cartesian pose: expected at least 6 finite values, got {cartesian.shape}"
        )

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = cartesian[:3] / 1000.0
    pose[:3, :3] = Rotation.from_euler("xyz", cartesian[3:6]).as_matrix()
    return pose


def target_pose_with_y_offset(controller, qpos: np.ndarray, y_offset_m: float) -> np.ndarray:
    qpos = validate_vector(qpos, (6,), "source qpos")
    code, cartesian = controller.arm.get_forward_kinematics(
        qpos.tolist(),
        input_is_radian=True,
        return_is_radian=True,
    )
    if code != 0:
        raise RuntimeError(f"xArm forward kinematics failed with code {code}")

    pose = cartesian_to_homo(np.asarray(cartesian, dtype=np.float64))
    pose[:3, 3] += np.array([0.0, y_offset_m, 0.0], dtype=np.float64)
    return pose


def validate_vector(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid {label}: expected finite shape {shape}, got {array.shape}")
    return array


def validate_pose(value, label: str) -> np.ndarray:
    return validate_vector(value, (4, 4), label)


def verify_reached_pose(
    target_eef: np.ndarray,
    actual_eef: np.ndarray,
    max_translation_error_m: float,
    max_rotation_error_rad: float,
) -> tuple[float, float]:
    target_eef = validate_pose(target_eef, "target eef")
    actual_eef = validate_pose(actual_eef, "actual eef")
    translation_error = float(
        np.linalg.norm(actual_eef[:3, 3] - target_eef[:3, 3])
    )
    relative_rotation = target_eef[:3, :3].T @ actual_eef[:3, :3]
    rotation_error = float(
        np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    )
    if (
        translation_error > max_translation_error_m
        or rotation_error > max_rotation_error_rad
    ):
        raise RuntimeError(
            "xArm did not reach the commanded pose: "
            f"translation={translation_error * 1000.0:.2f}mm "
            f"(limit {max_translation_error_m * 1000.0:.2f}mm), "
            f"rotation={np.degrees(rotation_error):.2f}deg "
            f"(limit {np.degrees(max_rotation_error_rad):.2f}deg)"
        )
    return translation_error, rotation_error


def pose_files(source_dir: Path, start: int | None, end: int | None) -> list[Path]:
    files = sorted(
        source_dir.glob("*_qpos.npy"),
        key=lambda path: int(path.name.split("_", 1)[0]),
    )
    if not files:
        raise FileNotFoundError(f"No *_qpos.npy files found in {source_dir}")

    if start is not None:
        files = [path for path in files if int(path.name.split("_", 1)[0]) >= start]
    if end is not None:
        files = [path for path in files if int(path.name.split("_", 1)[0]) < end]
    if not files:
        raise FileNotFoundError(
            f"No *_qpos.npy files selected in {source_dir} for start={start}, end={end}"
        )
    return files


def readiness_phrase(side_name: str) -> str:
    return f"MOVE {side_name.upper()}"


def require_ready(side_name: str, prompt_fn: Callable[[str], str]) -> None:
    phrase = readiness_phrase(side_name)
    answer = prompt_fn(
        f"About to move the {side_name} xArm. Type '{phrase}' "
        f"or '{phrase.replace(' ', '_')}' to continue: "
    )
    normalized_answer = " ".join(answer.strip().upper().replace("_", " ").split())
    if normalized_answer != phrase:
        raise RuntimeError(f"Aborted before {side_name}: readiness phrase did not match")


def output_exists(output_dir: Path, side_name: str, pose_path: Path) -> bool:
    return (output_dir / side_name / pose_path.name).exists()


def selected_pose_files(
    source_dir: Path,
    output_dir: Path,
    side_name: str,
    start: int | None,
    end: int | None,
    resume: bool,
) -> list[Path]:
    files = pose_files(source_dir, start, end)
    if resume:
        files = [
            pose_path
            for pose_path in files
            if not output_exists(output_dir, side_name, pose_path)
        ]
    return files


def save_measurement(
    side_dir: Path,
    pose_path: Path,
    source_qpos: np.ndarray,
    target_eef: np.ndarray,
    robot_data: dict,
    side: SideConfig,
) -> None:
    actual_qpos = validate_vector(robot_data["qpos"], (6,), "actual qpos")
    actual_eef = validate_pose(robot_data["position"], "actual eef")
    timestamp = float(robot_data.get("time", time.time()))

    side_dir.mkdir(parents=True, exist_ok=True)
    np.save(side_dir / pose_path.name, actual_qpos)
    np.save(side_dir / pose_path.name.replace("_qpos.npy", "_eef.npy"), actual_eef)
    with open(side_dir / pose_path.name.replace("_qpos.npy", "_meta.json"), "w") as f:
        json.dump(
            {
                "side": side.name,
                "ip": side.ip,
                "source_file": str(pose_path),
                "y_offset_m": side.y_offset_m,
                "timestamp": timestamp,
                "source_qpos": source_qpos.tolist(),
                "target_eef": target_eef.tolist(),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def record_side(
    side: SideConfig,
    source_dir: Path,
    output_dir: Path,
    files: Iterable[Path],
    *,
    settle_seconds: float,
    max_translation_error_m: float,
    max_rotation_error_rad: float,
    overwrite: bool,
    controller_factory: Callable[[str], object] = create_controller,
    prompt_fn: Callable[[str], str] = input,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    files = list(files)
    if not files:
        print(f"{side.name}: nothing to record")
        return 0

    require_ready(side.name, prompt_fn)
    controller = controller_factory(side.ip)
    side_dir = output_dir / side.name
    try:
        for step, pose_path in enumerate(files, start=1):
            if output_exists(output_dir, side.name, pose_path) and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {side_dir / pose_path.name}. "
                    "Use --resume to skip it or --overwrite to replace it."
                )

            source_qpos = validate_vector(np.load(pose_path), (6,), f"{pose_path} qpos")
            target_eef = target_pose_with_y_offset(controller, source_qpos, side.y_offset_m)
            print(
                f"{side.name}: {step}/{len(files)} moving {pose_path.name} "
                f"with base Y offset {side.y_offset_m:+.3f} m"
            )
            controller.move(target_eef, is_servo=False)
            sleep_fn(settle_seconds)
            robot_data = controller.get_data()
            verify_reached_pose(
                target_eef,
                robot_data["position"],
                max_translation_error_m,
                max_rotation_error_rad,
            )
            save_measurement(
                side_dir,
                pose_path,
                source_qpos,
                target_eef,
                robot_data,
                side,
            )
            print(f"{side.name}: saved {side_dir / pose_path.name}")
    finally:
        controller.end(set_break=False)

    return len(files)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Right/Left xArm bimanual qpos files by physically moving to "
            "base-frame Y-shifted Cartesian poses. No IK is solved in this script."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--right-ip", default=right_ip_default())
    parser.add_argument("--left-ip", default=LEFT_XARM_IP)
    parser.add_argument("--start", type=int, default=None, help="First numeric pose id, inclusive.")
    parser.add_argument("--end", type=int, default=None, help="Last numeric pose id, exclusive.")
    parser.add_argument(
        "--side",
        choices=("both", "Right", "Left"),
        default="both",
        help="Record both arms in order, or only one side.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-recorded *_qpos.npy outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files.")
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--max-translation-error-mm", type=float, default=5.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=3.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually connect to and move the robots. Without this, only prints the plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.start is not None and args.end is not None and args.start >= args.end:
        raise ValueError("--start must be smaller than --end")
    if args.settle_seconds < 0:
        raise ValueError("--settle-seconds must be non-negative")
    if args.max_translation_error_mm <= 0 or args.max_rotation_error_deg <= 0:
        raise ValueError("pose error limits must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")

    sides = [
        SideConfig("Right", args.right_ip, RIGHT_Y_OFFSET_M),
        SideConfig("Left", args.left_ip, LEFT_Y_OFFSET_M),
    ]
    if args.side != "both":
        sides = [side for side in sides if side.name == args.side]

    selected = {
        side.name: selected_pose_files(
            args.source_dir,
            args.output_dir,
            side.name,
            args.start,
            args.end,
            args.resume,
        )
        for side in sides
    }

    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    for side in sides:
        files = selected[side.name]
        print(
            f"{side.name}: ip={side.ip}, y_offset={side.y_offset_m:+.3f} m, "
            f"poses={len(files)}"
        )

    if not args.execute:
        print("Dry run only. Re-run with --execute to move the robots.")
        return 0

    total = 0
    for side in sides:
        total += record_side(
            side,
            args.source_dir,
            args.output_dir,
            selected[side.name],
            settle_seconds=args.settle_seconds,
            max_translation_error_m=args.max_translation_error_mm / 1000.0,
            max_rotation_error_rad=np.deg2rad(args.max_rotation_error_deg),
            overwrite=args.overwrite,
        )
    print(f"Recorded {total} measured poses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
