import time

from paradex.video.raw_video_processor import RawVideoProcessor
from paradex.io.capture_pc.data_sender import DataPublisher

class VideoProgressPublisher:
    """비디오 처리 진행상황을 실시간으로 Main PC에 전송"""
    
    def __init__(self, port=1234):
        self.processor = RawVideoProcessor()
        self.publisher = DataPublisher(port=port, name="video_processor")
        
    def start_processing(self, update_interval=1.0):
        """비디오 처리 시작 및 진행상황 전송"""
        
        # 처리 시작
        self.processor.process()
        
        print(f"Processing {len(self.processor.videopath_list)} videos...")
        print(f"Publishing progress updates every {update_interval}s")
        
        # 진행상황 전송 루프
        while not self.processor.finished():
            # 현재 진행상황 가져오기
            progress_data = self.processor.get_progress()
            
            # 전송할 데이터 준비
            metadata = []
            for video_id, info in progress_data.items():
                metadata.append({
                    'name': video_id,  # 비디오 ID를 name으로
                    'status': info.get('status', 'unknown'),
                    'progress': info.get('progress', 0),
                    'current_frame': info.get('current_frame', 0),
                    'total_frames': info.get('total_frames', 0),
                    'fps': info.get('fps', 0),
                    'eta': info.get('eta', 0),
                    'message': info.get('message', ''),
                    'video_path': info.get('video_path', '')
                })
            
            # 데이터 전송 (이미지 없이 메타데이터만)
            if metadata:
                self.publisher.send_data(metadata=metadata, data=[])
            
            time.sleep(update_interval)
        
        # 최종 결과 전송
        final_progress = self.processor.get_progress()
        final_metadata = []
        for video_id, info in final_progress.items():
            final_metadata.append({
                'name': video_id,
                'status': info.get('status', 'unknown'),
                'progress': info.get('progress', 0),
                'current_frame': info.get('current_frame', 0),
                'total_frames': info.get('total_frames', 0),
                'message': info.get('message', ''),
                'video_path': info.get('video_path', '')
            })
        
        if final_metadata:
            self.publisher.send_data(metadata=final_metadata, data=[])
        
        print("\n✅ All videos processed!")
        print(f"\nLogs ({len(self.processor.log)} entries):")
        for log in self.processor.log:
            print(f"  - {log}")
        
        self.publisher.close()
        
rvp = RawVideoProcessor()
rvp.process()