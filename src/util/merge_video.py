import os
import json
from paradex.utils.merge_video import merge_video_synced

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge video files.")
    parser.add_argument("--input_dir", type=str, help="Input directory containing video files.")
    parser.add_argument("--output_file", type=str, help="Output file path.")

    args = parser.parse_args()
    temp_video_path = args.output_file.replace(".mp4", "_temp.mp4")

    merge_video_synced(args.input_dir, temp_video_path)

    output_video_path = args.output_file
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # 기존 파일 덮어쓰기
        "-i", temp_video_path,  # 입력 파일
        "-c:v", "libx264",  # 비디오 코덱: H.264
        "-preset", "slow",  # 압축률과 속도 조절 (slow = 고품질)
        "-crf", "23",  # 품질 설정 (낮을수록 고품질, 18~23 추천)
        "-pix_fmt", "yuv420p",  # 픽셀 포맷 (H.264 표준 호환)
        output_video_path
    ]

    # FFmpeg 실행
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"✅ H.264 encoded video saved: {output_video_path}")
    os.remove(temp_video_path)  # 변환 후 임시 파일 삭제
    
os.remove(temp_video_path)  # 변환 후 임시 파일 삭제