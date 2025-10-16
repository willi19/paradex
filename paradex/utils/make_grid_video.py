import cv2
import os
import numpy as np

# 1. 비디오 파일 경로
folder_path ="/home/jisoo/shared_data/capture/final_final_simulate/0"
video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".mp4", ".mov", ".avi"))]

# 2. 캡처 객체 생성
caps = [cv2.VideoCapture(f) for f in video_files]

# 3. FPS와 프레임 수 계산
fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
fps = min(fps_list)
frames_list = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
max_frames = min(frames_list)

# 4. 각 비디오 원본 크기
widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]

max_width = max(widths)
max_height = max(heights)

# 5. 4x6 그리드
cols, rows = 6, 4
out_width = cols * max_width
out_height = rows * max_height

# 6. 최대 출력 해상도 제한
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
scale_w = min(1, MAX_WIDTH / out_width)
scale_h = min(1, MAX_HEIGHT / out_height)
scale = min(scale_w, scale_h)

out_width = int(out_width * scale)
out_height = int(out_height * scale)
max_width = int(max_width * scale)
max_height = int(max_height * scale)

# 7. 비디오 라이터
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_grid_2.mp4", fourcc, fps, (out_width, out_height))

# 8. 프레임 단위로 그리드 생성
for i in range(max_frames):
    grid_frame = np.zeros((rows * max_height, cols * max_width, 3), dtype=np.uint8)
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((heights[idx], widths[idx], 3), dtype=np.uint8)

        frame_resized = cv2.resize(frame, (max_width, max_height))

        row_idx = idx // cols
        col_idx = idx % cols

        y1 = row_idx * max_height
        y2 = y1 + max_height
        x1 = col_idx * max_width
        x2 = x1 + max_width

        grid_frame[y1:y2, x1:x2] = frame_resized

    out.write(grid_frame)
    print(f"Processed frame {i+1}/{max_frames}")
# 9. 릴리즈
for cap in caps:
    cap.release()
out.release()
cv2.destroyAllWindows()
