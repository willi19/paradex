import cv2
import os
import tqdm
import multiprocessing as mp
from paradex.utils.path import shared_dir, home_path
from paradex.utils.upload_file import rsync_copy

def process_video(video_path):
    root_dir = os.path.dirname(os.path.dirname(video_path))
    cam_name = os.path.basename(video_path).split('.')[0]

    output_dir = os.path.join(root_dir, "images", cam_name)
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in tqdm.tqdm(range(1, length + 1)):
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        if os.path.exists(output_path):
            continue
        # Skip white frames
        if frame.shape[0] >= 30 and frame.shape[1] >= 30:
            test = frame[:30, :30]
            if (test[::2, ::2] == 255).all():
                continue
        
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    print(f"Extracted frames from {video_path} to {output_dir}")

if __name__ == "__main__":
    root_dir = os.path.join(home_path, "paradex_download/capture/object_turntable")

    # Collect tasks
    tasks = []
    for obj_name in sorted(os.listdir(root_dir)):
        obj_path = os.path.join(root_dir, obj_name)
        for index in os.listdir(obj_path):
            video_path = os.path.join(obj_path, index, "videos")
            if os.path.exists(video_path):
                if len(os.listdir(video_path)) != 24:
                    print(f"Warning: No video files in {video_path}")
                    
                for video_file in os.listdir(video_path):
                    tasks.append(os.path.join(video_path, video_file))
            else:
                print(f"Warning: {video_path} does not exist")
    
    print(f"Processing {len(tasks)} sequences with {mp.cpu_count()} workers")
    
    # Parallel
    with mp.Pool(processes=mp.cpu_count()-2) as pool:
        list(tqdm.tqdm(
            pool.imap_unordered(process_video, tasks),
            total=len(tasks)
        ))
    
    print("✓ Done!")