#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


AVI_EXTENSIONS = {".avi"}
TAIL_WINDOW_FRAMES = 300
CLIP_FRAMES = 180


def collect_one_video_per_videos_dir(base_path: Path, rng: random.Random):
    selected = []

    for robot_dir in sorted(base_path.iterdir()):
        if not robot_dir.is_dir():
            continue

        # 예외 하드코딩:
        # base_path/hand_taeyun/right/object_name/index/videos
        if robot_dir.name == "hand_taeyun":
            right_dir = robot_dir / "right"
            if not right_dir.is_dir():
                continue

            for object_dir in sorted(right_dir.iterdir()):
                if not object_dir.is_dir():
                    continue

                for index_dir in sorted(object_dir.iterdir()):
                    if not index_dir.is_dir():
                        continue

                    videos_dir = index_dir / "videos"
                    if not videos_dir.is_dir():
                        continue

                    candidates = [
                        p
                        for p in videos_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in AVI_EXTENSIONS
                    ]
                    if not candidates:
                        continue

                    selected.append(rng.choice(candidates))
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

                candidates = [
                    p
                    for p in videos_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in AVI_EXTENSIONS
                ]
                if not candidates:
                    continue

                selected.append(rng.choice(candidates))

    return selected


def pick_videos_for_grid(paths, cell_count, rng: random.Random):
    if not paths:
        return []

    if len(paths) >= cell_count:
        return rng.sample(paths, cell_count)

    # 비디오 수가 부족하면 중복 허용으로 채운다.
    return paths + rng.choices(paths, k=cell_count - len(paths))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "base_path/robot/object/index/videos/*.avi 에서 videos 폴더당 랜덤 1개를 뽑고 "
            "랜덤 시작 시점으로 그리드 합성 영상을 생성합니다."
        )
    )
    parser.add_argument("base_path", type=Path, help="robot 폴더들이 있는 최상위 경로")
    parser.add_argument(
        "--rows",
        type=int,
        default=25,
        help="그리드 행 수 (기본값: 25)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=40,
        help="그리드 열 수 (기본값: 40)",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=32,
        help="각 타일 가로 픽셀 (기본값: 32)",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=24,
        help="각 타일 세로 픽셀 (기본값: 24)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="출력 영상 길이(초, 기본값: 10)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="출력 FPS (기본값: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("random_grid.mp4"),
        help="출력 파일 경로 (기본값: random_grid.mp4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드",
    )
    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=80.0,
        help=(
            "프레임 평균 픽셀값이 이 값보다 작으면 드롭 프레임으로 간주하고 "
            "이전 프레임으로 대체 (기본값: 80.0)"
        ),
    )
    return parser.parse_args()


def choose_clip_start(frame_count: int, rng: random.Random) -> int:
    if frame_count <= 1:
        return 0

    if frame_count >= CLIP_FRAMES:
        if frame_count >= TAIL_WINDOW_FRAMES:
            start_min = frame_count - TAIL_WINDOW_FRAMES
            start_max = frame_count - CLIP_FRAMES
            if start_max < start_min:
                start_max = start_min
            return rng.randint(start_min, start_max)
        return rng.randint(0, frame_count - CLIP_FRAMES)

    return rng.randint(0, frame_count - 1)


def main():
    args = parse_args()
    base_path = args.base_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not base_path.is_dir():
        raise SystemExit(f"base_path가 디렉터리가 아닙니다: {base_path}")
    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("--rows, --cols 는 1 이상이어야 합니다.")
    if args.tile_width <= 0 or args.tile_height <= 0:
        raise SystemExit("--tile-width, --tile-height 는 1 이상이어야 합니다.")
    if args.seconds <= 0 or args.fps <= 0:
        raise SystemExit("--seconds, --fps 는 0보다 커야 합니다.")

    rng = random.Random(args.seed)
    per_dir_samples = collect_one_video_per_videos_dir(base_path, rng)
    if not per_dir_samples:
        raise SystemExit("유효한 AVI 비디오를 찾지 못했습니다.")

    cell_count = args.rows * args.cols
    grid_videos = pick_videos_for_grid(per_dir_samples, cell_count, rng)

    total_frames = int(round(args.seconds * args.fps))
    out_width = args.cols * args.tile_width
    out_height = args.rows * args.tile_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        raise SystemExit(f"출력 파일을 열 수 없습니다: {output_path}")

    players = []
    try:
        for path in grid_videos:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                players.append(
                    {
                        "path": path,
                        "cap": None,
                        "clip_start": 0,
                        "clip_len": 0,
                        "clip_pos": 0,
                        "prev_frame": None,
                    }
                )
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            clip_start = choose_clip_start(frame_count, rng)
            clip_len = min(CLIP_FRAMES, frame_count) if frame_count > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            players.append(
                {
                    "path": path,
                    "cap": cap,
                    "clip_start": clip_start,
                    "clip_len": clip_len,
                    "clip_pos": 0,
                    "prev_frame": None,
                }
            )

        black = np.zeros((args.tile_height, args.tile_width, 3), dtype=np.uint8)
        canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        for fi in range(total_frames):
            for idx, player in enumerate(players):
                r = idx // args.cols
                c = idx % args.cols
                y0 = r * args.tile_height
                x0 = c * args.tile_width

                frame_small = black
                cap = player["cap"]
                if cap is not None:
                    clip_len = player["clip_len"]
                    clip_start = player["clip_start"]
                    clip_pos = player["clip_pos"]

                    if clip_len > 0 and clip_pos >= clip_len:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
                        clip_pos = 0

                    ok, frame = cap.read()
                    if not ok:
                        # 읽기 실패 시 선택된 클립 시작점으로 되돌려 재시도
                        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
                        ok, frame = cap.read()

                    if ok and frame is not None:
                        resized = cv2.resize(
                            frame,
                            (args.tile_width, args.tile_height),
                            interpolation=cv2.INTER_AREA,
                        )
                        mean_pixel = float(resized.mean())
                        prev_frame = player["prev_frame"]

                        if mean_pixel < args.dark_threshold and prev_frame is not None:
                            frame_small = prev_frame
                        else:
                            frame_small = resized
                            player["prev_frame"] = resized
                        player["clip_pos"] = clip_pos + 1
                    else:
                        prev_frame = player["prev_frame"]
                        if prev_frame is not None:
                            frame_small = prev_frame

                canvas[y0 : y0 + args.tile_height, x0 : x0 + args.tile_width] = frame_small

            writer.write(canvas)

            if (fi + 1) % max(1, int(args.fps)) == 0:
                print(f"[progress] {fi + 1}/{total_frames} frames")

    finally:
        writer.release()
        for player in players:
            cap = player["cap"]
            if cap is not None:
                cap.release()

    print(f"[done] source_videos={len(per_dir_samples)}, grid_cells={cell_count}")
    print(f"[done] output={output_path}")


if __name__ == "__main__":
    main()
