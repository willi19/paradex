#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


DEFAULT_EXTENSIONS = {".avi"}


def collect_one_video_per_videos_dir(base_path: Path, rng: random.Random, all_files: bool):
    selected = []

    for robot_dir in sorted(base_path.iterdir()):
        if not robot_dir.is_dir():
            continue

        for object_dir in sorted(robot_dir.iterdir()):
            if not object_dir.is_dir():
                continue

            for index_dir in sorted(object_dir.iterdir()):
                if not index_dir.is_dir():
                    continue

                videos_dir = index_dir / "videos"
                if not videos_dir.is_dir():
                    continue

                candidates = []
                for p in videos_dir.iterdir():
                    if not p.is_file():
                        continue
                    if all_files or p.suffix.lower() in DEFAULT_EXTENSIONS:
                        candidates.append(p)

                if not candidates:
                    continue

                selected.append(rng.choice(candidates))

    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "base_path/robot/object/index/videos 구조를 순회하면서 각 videos 폴더에서 "
            "랜덤 비디오 1개씩 뽑아 개수를 출력합니다."
        )
    )
    parser.add_argument("base_path", type=Path, help="robot 폴더들이 있는 최상위 경로")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드(재현 가능한 결과가 필요할 때 사용)",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="확장자 필터 없이 videos 폴더의 모든 파일을 후보로 사용",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="선택된 비디오 경로도 함께 출력",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = args.base_path.expanduser().resolve()

    if not base_path.is_dir():
        raise SystemExit(f"base_path가 디렉터리가 아닙니다: {base_path}")

    rng = random.Random(args.seed)
    selected = collect_one_video_per_videos_dir(base_path, rng, args.all_files)

    print(len(selected))

    if args.show:
        for path in selected:
            print(path)


if __name__ == "__main__":
    main()
