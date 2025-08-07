import cv2
import json
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt

class ArucoOffsetAnalyzer:
    def __init__(self, video_path, timestamps_json_path, marker_log_json_path):
        self.video_path = video_path
        self.timestamps_json_path = timestamps_json_path
        self.marker_log_json_path = marker_log_json_path
        
        # ArUco 설정 (버전 호환성)
        try:
            # OpenCV 4.7+ 방식
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            try:
                self.aruco_params = cv2.aruco.DetectorParameters()
            except:
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            # 이전 버전 방식
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False
        
        # 데이터 로드
        self.camera_data = self.load_camera_data()
        self.marker_log = self.load_marker_log()
        
        print(f"Camera data loaded: {len(self.camera_data['timestamps'])} frames")
        print(f"Marker log loaded: {len(self.marker_log['marker_changes'])} changes")
    
    def load_camera_data(self):
        """카메라 타임스탬프 데이터 로드"""
        with open(self.timestamps_json_path, 'r') as f:
            data = json.load(f)
        
        # 타임스탬프를 초 단위로 변환 (밀리초에서 초로)
        timestamps_ms = data['timestamps']
        timestamps_sec = [ts / 1000.0 for ts in timestamps_ms]
        
        return {
            'timestamps': timestamps_sec,
            'frameID': data['frameID'],
            'pc_time': data['pc_time']
        }
    
    def load_marker_log(self):
        """마커 변경 로그 데이터 로드"""
        with open(self.marker_log_json_path, 'r') as f:
            return json.load(f)
    
    def detect_markers_in_video(self):
        """비디오에서 ArUco 마커 감지"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        detected_markers = []
        frame_count = 0
        
        print("Analyzing video frames for ArUco markers...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ArUco 마커 감지 (버전 호환성)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.use_new_api:
                try:
                    corners, ids, rejected = self.detector.detectMarkers(gray)
                except:
                    corners, ids, rejected = cv2.aruco.detectMarkers(
                        gray, self.aruco_dict, parameters=self.aruco_params)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None and len(ids) > 0:
                # 가장 큰 마커 선택 (가장 신뢰할 만한)
                areas = [cv2.contourArea(corner[0]) for corner in corners]
                max_area_idx = np.argmax(areas)
                marker_id = ids[max_area_idx][0]
                
                detected_markers.append({
                    'frame_number': frame_count,
                    'marker_id': marker_id,
                    'area': areas[max_area_idx]
                })
                
                if frame_count % 10 == 0:
                    print(f"Frame {frame_count}: Detected marker ID {marker_id}")
            
            frame_count += 1
        
        cap.release()
        print(f"Video analysis complete. Detected markers in {len(detected_markers)} frames.")
        return detected_markers
    
    def match_markers_with_log(self, detected_markers):
        """감지된 마커와 로그의 마커를 매칭"""
        # 마커 로그를 ID별로 인덱싱
        marker_log_by_id = {}
        for change in self.marker_log['marker_changes']:
            marker_id = change['marker_id']
            if marker_id not in marker_log_by_id:
                marker_log_by_id[marker_id] = []
            marker_log_by_id[marker_id].append(change)
        
        matches = []
        
        for detected in detected_markers:
            frame_num = detected['frame_number']
            marker_id = detected['marker_id']
            
            # 해당 프레임의 카메라 타임스탬프 가져오기
            if frame_num < len(self.camera_data['timestamps']):
                camera_timestamp = self.camera_data['timestamps'][frame_num]
                pc_timestamp = self.camera_data['pc_time'][frame_num]
                
                # 로그에서 같은 마커 ID 찾기
                if marker_id in marker_log_by_id:
                    # 가장 가까운 시간의 마커 변경 찾기
                    best_match = None
                    min_time_diff = float('inf')
                    
                    for log_entry in marker_log_by_id[marker_id]:
                        log_timestamp = log_entry['timestamp']
                        time_diff = abs(pc_timestamp - log_timestamp)
                        
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_match = log_entry
                    
                    if best_match and min_time_diff < 1.0:  # 1초 이내 매칭만 허용
                        offset = pc_timestamp - best_match['timestamp']
                        matches.append({
                            'frame_number': frame_num,
                            'marker_id': marker_id,
                            'camera_timestamp': camera_timestamp,
                            'pc_timestamp': pc_timestamp,
                            'log_timestamp': best_match['timestamp'],
                            'offset': offset,
                            'log_frame_number': best_match['frame_number']
                        })
        
        return matches
    
    def analyze_offset(self, matches):
        """오프셋 분석"""
        if not matches:
            print("No matches found for analysis.")
            return
        
        offsets = [match['offset'] for match in matches]
        
        print(f"\n=== Offset Analysis ===")
        print(f"Total matches: {len(matches)}")
        print(f"Average offset: {np.mean(offsets):.6f} seconds")
        print(f"Median offset: {np.median(offsets):.6f} seconds")
        print(f"Standard deviation: {np.std(offsets):.6f} seconds")
        print(f"Min offset: {np.min(offsets):.6f} seconds")
        print(f"Max offset: {np.max(offsets):.6f} seconds")
        
        # 히스토그램 생성
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(offsets, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Offset (seconds)')
        plt.ylabel('Frequency')
        plt.title('Offset Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        frame_numbers = [match['frame_number'] for match in matches]
        plt.plot(frame_numbers, offsets, 'b-', alpha=0.7)
        plt.xlabel('Frame Number')
        plt.ylabel('Offset (seconds)')
        plt.title('Offset vs Frame Number')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        pc_times = [(match['pc_timestamp'] - matches[0]['pc_timestamp']) for match in matches]
        plt.plot(pc_times, offsets, 'r-', alpha=0.7)
        plt.xlabel('Time (seconds from start)')
        plt.ylabel('Offset (seconds)')
        plt.title('Offset vs Time')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        marker_ids = [match['marker_id'] for match in matches]
        unique_ids = sorted(set(marker_ids))
        id_offsets = [np.mean([m['offset'] for m in matches if m['marker_id'] == mid]) for mid in unique_ids]
        
        # 너무 많은 마커가 있으면 처음 20개만 표시
        if len(unique_ids) > 20:
            display_ids = unique_ids[:20]
            display_offsets = id_offsets[:20]
        else:
            display_ids = unique_ids
            display_offsets = id_offsets
        
        x_positions = range(len(display_ids))
        plt.bar(x_positions, display_offsets, alpha=0.7)
        plt.xlabel('Marker ID')
        plt.ylabel('Average Offset (seconds)')
        plt.title('Average Offset by Marker ID')
        plt.xticks(x_positions, [f"{mid}" for mid in display_ids])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'matches': matches,
            'statistics': {
                'count': len(matches),
                'mean': np.mean(offsets),
                'median': np.median(offsets),
                'std': np.std(offsets),
                'min': np.min(offsets),
                'max': np.max(offsets)
            }
        }
    
    def save_results(self, analysis_results, output_path="offset_analysis.json"):
        """분석 결과 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"Analysis results saved to: {output_path}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("Starting ArUco offset analysis...")
        
        # 1. 비디오에서 마커 감지
        detected_markers = self.detect_markers_in_video()
        
        # 2. 로그와 매칭
        matches = self.match_markers_with_log(detected_markers)
        
        # 3. 오프셋 분석
        analysis_results = self.analyze_offset(matches)
        
        # 4. 결과 저장
        if analysis_results:
            self.save_results(analysis_results)
        
        return analysis_results

def main():
    # 파일 경로 설정
    video_dir = "/home/temp_id/captures1/debug_offset/2/video"
    video_file = "22645026-0000.avi"
    timestamps_file = "22645026_timestamps.json"
    
    video_path = os.path.join(video_dir, video_file)
    timestamps_path = os.path.join(video_dir, timestamps_file)
    
    # 마커 로그 파일 경로 (사용자가 지정)
    marker_log_path = "marker_log_1754581138.json"
    if not marker_log_path:
        print("Error: Please provide marker log file path")
        return
    
    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    if not os.path.exists(timestamps_path):
        print(f"Error: Timestamps file not found: {timestamps_path}")
        return
    if not os.path.exists(marker_log_path):
        print(f"Error: Marker log file not found: {marker_log_path}")
        return
    
    # 분석 실행
    analyzer = ArucoOffsetAnalyzer(video_path, timestamps_path, marker_log_path)
    results = analyzer.run_analysis()
    
    if results:
        print(f"\nAnalysis complete! Found {len(results['matches'])} matches.")
        print(f"Average camera lag: {results['statistics']['mean']*1000:.2f} ms")

if __name__ == "__main__":
    main()