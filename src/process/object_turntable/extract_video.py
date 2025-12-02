import os
import subprocess
from paradex.utils.path import shared_dir, home_path
from paradex.utils.upload_file import rsync_copy

root_dir = os.path.join(shared_dir, "capture/object_turntable")
obj_list = ['pepper_tuna']

for obj_name in obj_list:
    index_list = os.listdir(os.path.join(root_dir, obj_name))
    for index in index_list:
        demo_path = os.path.join("capture/object_turntable", obj_name, index)
        video_dir = os.path.join(shared_dir, demo_path, "videos")
        if not os.path.exists(video_dir):
            continue
        download_dir = os.path.join(home_path, "paradex_download", demo_path)
        os.makedirs(download_dir, exist_ok=True)
        rsync_copy(video_dir + "/", download_dir + "/", checksum=True, resume=True, verbose=False)
        
        print(f"Downloaded {demo_path}")
        
        video_dir = os.path.join(home_path, "paradex_download", demo_path, "videos")
        
        # Frame 추출
        output_dir = os.path.join(home_path, "paradex_download", demo_path, "images")
        os.makedirs(output_dir, exist_ok=True)
        
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))])
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            cam_name = os.path.splitext(video_file)[0]
            os.makedirs(os.path.join(output_dir, cam_name), exist_ok=True)
            
            # ffmpeg로 20프레임마다 1개씩 추출
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', 'select=not(mod(n\\,20))',  # 20프레임마다 선택
                '-vsync', '0',  # 프레임 번호 유지
                '-qscale:v', '2',
                os.path.join(output_dir, cam_name, 'frame_%06d.jpg')
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  Extracted frames from {cam_name} (every 20 frames)")
        
        print(f"Frame extraction completed for {demo_path}")