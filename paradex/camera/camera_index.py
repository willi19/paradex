import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
json_path = os.path.join(current_dir, "../../config/camera_index.json")

# JSON 파일 불러오기
with open(json_path, "r", encoding="utf-8") as f:
    cam_index = json.load(f)
