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

def update_progress(progress_dict, video_id, updates):
    """Manager.dict 안전하게 업데이트"""
    temp = dict(progress_dict.get(video_id, {}))
    temp.update(updates)
    progress_dict[video_id] = temp
    
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
    update_progress(progress_dict, video_id, {
        'total_frames': num_frame,
        'message': 'Checking existing file...'
    })

    if os.path.exists(out_path):
        out = cv2.VideoCapture(out_path)
        total_frame = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
        out.release()
        
        if total_frame == num_frame:
            cap.release()
            update_progress(progress_dict, video_id, {
                'status': 'uploading',
                'message': 'Already processed. Uploading to NAS...'
            })
            rsync_copy(out_path, nas_path)
            os.remove(out_path)
            os.remove(video_path)
            update_progress(progress_dict, video_id, {
                'status': 'completed',
                'progress': 100.0,
                'current_frame': num_frame,
                'message': 'Success'
            })
            return f"{video_path}: already processed"
        
        else:
            print(f"{video_path}: incomplete file detected. Re-processing...")
            os.remove(out_path)
            update_progress(progress_dict, video_id, {
                'message': 'Re-processing incomplete file...'
            })

        update_progress(progress_dict, video_id, {
            'status': 'loading_params',
            'message': 'Loading camera parameters...'
        })
        
    try:
        intrinsics, _ = load_camparam(os.path.join(shared_dir, root_name))
        _, mapx, mapy = precomute_undistort_map(intrinsics[serial_num])

    except Exception as e:
        update_progress(progress_dict, video_id, {
            'status': 'failed',
            'message': f'Cannot load camparam: {str(e)}'
        })
        return f"{video_path}: cannot load camparam: {str(e)}"

    update_progress(progress_dict, video_id, {
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

            update_progress(progress_dict, video_id, {
                'current_frame': last_frame,
                'progress': (last_frame / num_frame * 100) if num_frame > 0 else 0,
                'fps': fps_actual,
                'eta': eta,
                'message': f'Processing... {last_frame}/{num_frame} frames'
            })
        
    out.release()
    cap.release()
    
    # 완료 상태 업데이트
    update_progress(progress_dict, video_id, {
        'status': 'uploading',
        'message': 'Uploading to NAS...'
    })
    
    rsync_copy(out_path, nas_path)

    update_progress(progress_dict, video_id, {
        'status': 'completed',
        'progress': 100.0,
        'current_frame': num_frame,
        'message': 'Success'
    })
    
    os.remove(out_path)
    os.remove(video_path)
    
    return f"{video_path}: success"

def get_raw_videopath_list():
    videopath_list = []
    for video_dir in capture_path_list:
        print(f"Searching raw videos in {video_dir}...")
        videopath_list += glob.glob(os.path.join(video_dir, "**/raw/videos/*.avi"), recursive=True)
    return videopath_list


class RawVideoProcessor:
    def __init__(self,):
        self.videopath_list = get_raw_videopath_list()
        self.num_workers = cpu_count()
        
        # Manager로 공유 상태 생성
        manager = Manager()
        self.progress_dict = manager.dict()
        self.log = manager.list()
        
        self.pool = None
        self.process_list = []
        
    def async_callback(self, result):
        self.log.append(result)
        
    def error_callback(self, e):
        self.log.append(f"ERROR in process: {e}")
        
    def finished(self):
        return all(r.ready() for r in self.process_list)
    
    def process(self):
        """비디오 처리 시작"""
        self.pool = Pool(processes=self.num_workers)
        
        # 각 비디오에 고유 ID 부여
        self.process_list = []
        for idx, vid_path in enumerate(self.videopath_list):
            video_id = vid_path.replace(home_path, "")
            if video_id.startswith("/"):
                video_id = video_id[1:]
            
            # 초기 상태 설정
            self.progress_dict[video_id] = {
                'status': 'pending',
                'video_path': vid_path,
                'current_frame': 0,
                'total_frames': 0,
                'progress': 0.0,
                'message': 'Waiting...'
            }
            
            # 프로세스 시작
            result = self.pool.apply_async(
                undistort_raw_video,
                args=(vid_path, self.progress_dict, video_id),
                callback=self.async_callback,
                error_callback=self.error_callback
            )
            self.process_list.append(result)
        
        self.pool.close()
        self.print_progress()
    
    def get_progress(self):
        """전체 진행상황 조회"""
        return dict(self.progress_dict)
    
    def get_overall_progress(self):
        """전체 진행률 계산"""
        if not self.progress_dict:
            return 0.0
        
        total_progress = sum(v['progress'] for v in self.progress_dict.values())
        return total_progress / len(self.progress_dict)
    
    def get_status_summary(self):
        """상태 요약"""
        summary = {
            'pending': 0,
            'starting': 0,
            'loading_params': 0,
            'processing': 0,
            'uploading': 0,
            'completed': 0,
            'failed': 0
        }
        
        for v in self.progress_dict.values():
            status = v.get('status', 'pending')
            summary[status] = summary.get(status, 0) + 1
        
        return summary
    
    def print_progress(self):
        """진행상황 출력"""
        print("\n" + "="*80)
        print(f"Overall Progress: {self.get_overall_progress():.1f}%")
        print("="*80)
        
        summary = self.get_status_summary()
        print(f"Status: Pending={summary['pending']}, Processing={summary['processing']}, "
              f"Completed={summary['completed']}, Failed={summary['failed']}")
        
        print("\nDetailed Progress:")
        for video_id, info in sorted(self.progress_dict.items()):
            status = info.get('status', 'unknown')
            progress = info.get('progress', 0)
            current = info.get('current_frame', 0)
            total = info.get('total_frames', 0)
            fps = info.get('fps', 0)
            eta = info.get('eta', 0)
            msg = info.get('message', '')
            
            video_name = os.path.basename(info.get('video_path', ''))
            
            if status == 'processing':
                print(f"  [{video_id}] {video_name}: "
                      f"{progress:.1f}% ({current}/{total}) | "
                      f"FPS: {fps:.1f} | ETA: {eta:.0f}s | {msg}")
            else:
                print(f"  [{video_id}] {video_name}: {status} - {msg}")
        
        print("="*80 + "\n")
    
    def wait_and_monitor(self, update_interval=2.0):
        """처리 완료까지 대기하면서 진행상황 모니터링"""
        while not self.finished():
            self.print_progress()
            time.sleep(update_interval)
        
        # 최종 결과
        self.print_progress()
        print("\n✅ All videos processed!")
        print(f"\nLogs ({len(self.log)} entries):")
        for log in self.log:
            print(f"  - {log}")
    
    def stop(self):
        """처리 중단"""
        if self.pool:
            self.pool.terminate()
            self.pool.join()
