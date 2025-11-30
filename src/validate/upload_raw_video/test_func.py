import os
import glob
from pathlib import Path
import cv2
import time
from multiprocessing import Pool, Manager, cpu_count

from paradex.utils.path import capture_path_list, shared_dir, home_path
from paradex.image.undistort import precomute_undistort_map, apply_undistort_map
from paradex.calibration.utils import load_camparam
from paradex.utils.upload_file import rsync_copy

def undistort_raw_video(video_path, progress_dict, video_id):
    """progress_dict로 진행상황 업데이트"""
    
    serial_num = os.path.basename(video_path).split(".")[0]
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(video_path)))
    
    root_name = root_dir
    for p in capture_path_list:
        root_name = root_name.replace(p, "")

    if root_name.startswith("/"):
        root_name = root_name[1:]
    
    out_path = os.path.join(root_dir, "videos", f"{serial_num}.avi")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    nas_path = os.path.join(shared_dir, root_name, "videos", f"{serial_num}.avi")
    os.makedirs(os.path.dirname(nas_path), exist_ok=True)
    
    # 초기 상태 업데이트
    progress_dict[video_id] = {
        'status': 'starting',
        'video_path': video_path,
        'current_frame': 0,
        'total_frames': 0,
        'progress': 0.0,
        'message': 'Initializing...'
    }
    
    cap = cv2.VideoCapture(video_path)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 총 프레임 수 업데이트
    progress_dict[video_id].update({
        'total_frames': num_frame,
        'message': 'Checking existing file...'
    })

    # if os.path.exists(out_path):
    #     out = cv2.VideoCapture(out_path)
    #     total_frame = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
    #     out.release()
        
    #     if total_frame == num_frame:
    #         cap.release()
    #         progress_dict[video_id].update({
    #             'status': 'uploading',
    #             'message': 'Already processed. Uploading to NAS...'
    #         })
    #         rsync_copy(out_path, nas_path)
    #         os.remove(out_path)
    #         os.remove(video_path)
    #         progress_dict[video_id].update({
    #             'status': 'completed',
    #             'progress': 100.0,
    #             'current_frame': num_frame,
    #             'message': 'Success'
    #         })
    #         return f"{video_path}: already processed"
        
    #     else:
    #         print(f"{video_path}: incomplete file detected. Re-processing...")
    #         os.remove(out_path)
    #         progress_dict[video_id].update({
    #             'message': 'Re-processing incomplete file...'
    #         })

    #     progress_dict[video_id].update({
    #         'status': 'loading_params',
    #         'message': 'Loading camera parameters...'
    #     })
        
    intrinsics, _ = load_camparam(os.path.join(shared_dir, root_name))
    _, mapx, mapy = precomute_undistort_map(intrinsics[serial_num])
        # progress_dict[video_id].update({
        #     'status': 'failed',
        #     'message': f'Cannot load camparam: {str(e)}'
        # })
        # return f"{video_path}: cannot load camparam: {str(e)}"
    
    progress_dict[video_id].update({
        'status': 'processing',
        'message': 'Processing frames...'
    })
      
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    last_frame = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = apply_undistort_map(frame, mapx, mapy)
        out.write(frame)
        last_frame += 1
        
        # 진행상황 업데이트 (매 30프레임마다)
        if last_frame % 30 == 0 or last_frame == num_frame:
            elapsed = time.time() - start_time
            fps_actual = last_frame / elapsed if elapsed > 0 else 0
            eta = (num_frame - last_frame) / fps_actual if fps_actual > 0 else 0
            
            progress_dict[video_id].update({
                'current_frame': last_frame,
                'progress': (last_frame / num_frame * 100) if num_frame > 0 else 0,
                'fps': fps_actual,
                'eta': eta,
                'message': f'Processing... {last_frame}/{num_frame} frames'
            })
            print(f"Processing {video_path}: {last_frame}/{num_frame} frames")
        
    out.release()
    cap.release()
    
    # 완료 상태 업데이트
    progress_dict[video_id].update({
        'status': 'uploading',
        'message': 'Uploading to NAS...'
    })
    
    rsync_copy(out_path, nas_path)
    
    progress_dict[video_id].update({
        'status': 'completed',
        'progress': 100.0,
        'current_frame': num_frame,
        'message': 'Success'
    })
    
    return f"{video_path}: success"

undistort_raw_video("/home/temp_id/captures1/erasethis1201/20251201_033700/raw/videos/22645026.avi", Manager().dict(), 0)