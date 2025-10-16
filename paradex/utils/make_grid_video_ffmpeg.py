import os

# 1. 비디오 폴더 경로
folder_path = "/home/jisoo/shared_data/capture/final_final_simulate/0"

# 2. mp4, mov, avi 파일만 읽기
video_files = sorted([
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith((".mp4", ".mov", ".avi"))
])

# 3. 최대 24개만 (4x6 그리드)
video_files = video_files[:24]
num_videos = len(video_files)

if num_videos == 0:
    raise ValueError("⚠️ 폴더에 비디오 파일이 없습니다.")

# 4. 그리드 크기 설정
cols, rows = 6, 4
tile_w, tile_h = 320, 180  # 한 칸 크기 (6x320, 4x180 → 1920x720)

# 5. 입력 부분 생성
inputs = " ".join([f'-i "{v}"' for v in video_files])

# 6. scale 필터 생성
scales = " ".join([f'[{i}:v]scale={tile_w}:{tile_h}[v{i}];' for i in range(num_videos)])

# 7. layout 계산 (각 영상의 좌표)
layout_parts = []
for i in range(num_videos):
    row = i // cols
    col = i % cols
    x = col * tile_w
    y = row * tile_h
    layout_parts.append(f"{x}_{y}")
layout = "|".join(layout_parts)

# 8. 최종 FFmpeg 명령어 구성
ffmpeg_cmd = (
    f'ffmpeg {inputs} -filter_complex "{scales} ' 
    f'{"".join([f"[v{i}]" for i in range(num_videos)])}'
    f'xstack=inputs={num_videos}:layout={layout}[outv]" '
    f'-map "[outv]" -c:v libx264 -crf 23 -preset fast output_grid_ffmpeg.mp4'
)

# 9. 결과 출력
print("✅ 생성된 FFmpeg 명령어:\n")
print(ffmpeg_cmd)
