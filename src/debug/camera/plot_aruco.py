import cv2
import numpy as np
import time
import math
from collections import deque

class ArucoMarkerGenerator:
    def __init__(self, target_fps=50, window_size=(800, 600)):
        # ArUco 딕셔너리 설정 (OpenCV 버전 호환)
        try:
            # OpenCV 4.7+ 방식
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        except AttributeError:
            # 이전 버전 방식
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        
        # 화면 설정
        self.window_width = window_size[0]
        self.window_height = window_size[1]
        self.window_name = "ArUco Marker Display - 50fps"
        
        # FPS 설정
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.fps_counter = deque(maxlen=30)
        
        # 마커 설정
        self.marker_size = 300  # 기본 크기 증가
        self.num_markers = 4
        self.current_marker_id = 0
        
        # 마커 변경 설정
        self.animation_time = 0
        self.rotation_angle = 0
        self.auto_change = True
        self.change_every_frame = True  # 매 프레임마다 변경
        
        # 마커 변경 로그
        self.marker_change_log = []  # (timestamp, marker_id) 저장
        
        print(f"ArUco Marker Generator initialized - Target FPS: {target_fps}")
        print("Press 'q' to quit, 'r' to rotate, 's' to save, 'n' for next marker")
        print("Keys: q(quit) s(save) f(frame-change) 1-4(modes)")
    
    def generate_marker(self, marker_id, size=200):
        """ArUco 마커 생성"""
        try:
            # OpenCV 4.7+ 방식
            marker = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, size)
        except AttributeError:
            # 이전 버전 방식
            marker = cv2.aruco.drawMarker(self.aruco_dict, marker_id, size)
        
        # 그레이스케일을 BGR로 변환
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        return marker_bgr
    
    def create_single_marker_display(self, marker_id):
        """단일 마커 디스플레이 생성"""
        # 배경 생성
        canvas = np.ones((self.window_height, self.window_width, 3), dtype=np.uint8) * 255
        
        # 마커 생성 (고정 크기, 회전 없음)
        marker = self.generate_marker(marker_id, self.marker_size)
        
        # 중앙에 배치
        x_offset = (self.window_width - self.marker_size) // 2
        y_offset = (self.window_height - self.marker_size) // 2
        
        canvas[y_offset:y_offset + self.marker_size, 
               x_offset:x_offset + self.marker_size] = marker
        
        return canvas
    
    def create_multi_marker_display(self, base_marker_id=0):
        """다중 마커 디스플레이 생성"""
        canvas = np.ones((self.window_height, self.window_width, 3), dtype=np.uint8) * 255
        
        marker_size = self.marker_size // 2
        positions = [
            (self.window_width // 4, self.window_height // 4),      # 좌상
            (3 * self.window_width // 4, self.window_height // 4),  # 우상
            (self.window_width // 4, 3 * self.window_height // 4),  # 좌하
            (3 * self.window_width // 4, 3 * self.window_height // 4)  # 우하
        ]
        
        for i, pos in enumerate(positions):
            marker_id = (base_marker_id + i) % 250  # ArUco 딕셔너리 범위 내에서
            marker = self.generate_marker(marker_id, marker_size)
            
            # 위치 조정 (회전 없음)
            x_start = max(0, pos[0] - marker_size // 2)
            y_start = max(0, pos[1] - marker_size // 2)
            x_end = min(self.window_width, x_start + marker_size)
            y_end = min(self.window_height, y_start + marker_size)
            
            # 실제 크기 조정
            actual_width = x_end - x_start
            actual_height = y_end - y_start
            
            if actual_width > 0 and actual_height > 0:
                marker_fitted = cv2.resize(marker, (actual_width, actual_height))
                canvas[y_start:y_end, x_start:x_end] = marker_fitted
                
                # 마커 ID 텍스트 표시
                cv2.putText(canvas, f"ID: {marker_id}", 
                           (x_start, y_start - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return canvas
    
    def create_grid_display(self, grid_size=3):
        """그리드 형태의 마커 디스플레이"""
        canvas = np.ones((self.window_height, self.window_width, 3), dtype=np.uint8) * 255
        
        marker_size = min(self.window_width, self.window_height) // (grid_size + 1)
        
        for row in range(grid_size):
            for col in range(grid_size):
                marker_id = (self.current_marker_id + row * grid_size + col) % 250
                
                x = (col + 1) * self.window_width // (grid_size + 1) - marker_size // 2
                y = (row + 1) * self.window_height // (grid_size + 1) - marker_size // 2
                
                marker = self.generate_marker(marker_id, marker_size)
                
                # 경계 체크
                if (x >= 0 and y >= 0 and 
                    x + marker_size <= self.window_width and 
                    y + marker_size <= self.window_height):
                    
                    canvas[y:y + marker_size, x:x + marker_size] = marker
        
        return canvas
    
    def create_rotating_circle_display(self):
        """원형으로 회전하는 마커 디스플레이"""
        canvas = np.ones((self.window_height, self.window_width, 3), dtype=np.uint8) * 255
        
        center_x = self.window_width // 2
        center_y = self.window_height // 2
        radius = min(center_x, center_y) - self.marker_size // 2 - 50
        
        num_markers = 6
        marker_size = self.marker_size // 2
        
        for i in range(num_markers):
            angle = (self.animation_time + i * 2 * math.pi / num_markers) % (2 * math.pi)
            
            x = int(center_x + radius * math.cos(angle) - marker_size // 2)
            y = int(center_y + radius * math.sin(angle) - marker_size // 2)
            
            marker_id = (self.current_marker_id + i) % 250
            marker = self.generate_marker(marker_id, marker_size)
            
            # 경계 체크
            if (x >= 0 and y >= 0 and 
                x + marker_size <= self.window_width and 
                y + marker_size <= self.window_height):
                
                canvas[y:y + marker_size, x:x + marker_size] = marker
        
        return canvas
    
    def draw_info(self, canvas, fps, mode_name):
        """정보 텍스트 그리기"""
        # FPS 정보
        cv2.putText(canvas, f"FPS: {fps:.1f}/{self.target_fps}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 모드 정보
        cv2.putText(canvas, f"Mode: {mode_name}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 현재 마커 ID
        cv2.putText(canvas, f"Base Marker ID: {self.current_marker_id}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 조작 가이드
        guide_text = "Keys: q(quit) s(save) f(frame-change) 1-4(modes)"
        cv2.putText(canvas, guide_text, 
                   (10, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    def run(self):
        """메인 실행 루프"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except AttributeError:
            cv2.namedWindow(self.window_name)
        
        frame_count = 0
        start_time = time.time()
        mode = 1  # 1: 단일, 2: 다중, 3: 그리드, 4: 원형 회전
        mode_names = {1: "Single Marker", 2: "Multi Markers", 3: "Grid Display", 4: "Rotating Circle"}
        
        print("Starting ArUco marker display...")
        
        while True:
            loop_start = time.time()
            
            # 애니메이션 업데이트
            self.animation_time = time.time() - start_time
            
            # 매 프레임마다 마커 변경
            if self.change_every_frame:
                new_marker_id = frame_count % 250  # 0-249 순환
                if new_marker_id != self.current_marker_id:
                    self.current_marker_id = new_marker_id
                    # 마커 변경 시점과 ID 로그 저장
                    self.marker_change_log.append((current_time, self.current_marker_id))
                else:
                    self.current_marker_id = new_marker_id
            
            # 모드에 따른 디스플레이 생성
            if mode == 1:
                canvas = self.create_single_marker_display(self.current_marker_id)
            elif mode == 2:
                canvas = self.create_multi_marker_display(self.current_marker_id)
            elif mode == 3:
                canvas = self.create_grid_display()
            elif mode == 4:
                canvas = self.create_rotating_circle_display()
            
            # FPS 계산
            current_time = time.time()
            if len(self.fps_counter) > 0:
                fps = len(self.fps_counter) / (current_time - self.fps_counter[0])
            else:
                fps = 0
            self.fps_counter.append(current_time)
            
            # 정보 표시
            self.draw_info(canvas, fps, mode_names.get(mode, "Unknown"))
            
            # 화면 출력
            cv2.imshow(self.window_name, canvas)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"aruco_display_{frame_count:04d}.png"
                cv2.imwrite(filename, canvas)
                print(f"Display saved as {filename}")
            elif key == ord('f'):
                self.change_every_frame = not self.change_every_frame
                print(f"Frame-by-frame marker change {'enabled' if self.change_every_frame else 'disabled'}")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                mode = int(chr(key))
                print(f"Mode changed to: {mode_names.get(mode, 'Unknown')}")
            
            frame_count += 1
            
            # FPS 제어
            loop_time = time.time() - loop_start
            if loop_time < self.frame_time:
                time.sleep(self.frame_time - loop_time)
            
            # 통계 출력 (매 100프레임마다)
            if frame_count % 100 == 0:
                elapsed = current_time - start_time
                avg_fps = frame_count / elapsed
                print(f"Frame {frame_count}: Average FPS = {avg_fps:.1f}, Current Marker ID = {self.current_marker_id}")
        
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        cv2.destroyAllWindows()
        
        print("\n=== ArUco Marker Change Log ===")
        print(f"Total marker changes: {len(self.marker_change_log)}")
        
        if len(self.marker_change_log) > 0:
            start_time = self.marker_change_log[0][0]
            end_time = self.marker_change_log[-1][0]
            duration = end_time - start_time
            
            print(f"Start time: {start_time:.6f}")
            print(f"End time: {end_time:.6f}")
            print(f"Duration: {duration:.3f} seconds")
            
            # 평균 변경 간격 계산
            if len(self.marker_change_log) > 1:
                avg_interval = duration / (len(self.marker_change_log) - 1)
                actual_fps = 1.0 / avg_interval if avg_interval > 0 else 0
                print(f"Average interval: {avg_interval:.6f} seconds")
                print(f"Actual change rate: {actual_fps:.1f} fps")
            
            # JSON 파일로 저장
            import json
            from datetime import datetime
            
            log_data = {
                "session_info": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration,
                    "total_changes": len(self.marker_change_log),
                    "average_interval": avg_interval if len(self.marker_change_log) > 1 else 0,
                    "actual_fps": actual_fps if len(self.marker_change_log) > 1 else 0,
                    "target_fps": self.target_fps
                },
                "marker_changes": [
                    {
                        "timestamp": timestamp,
                        "marker_id": marker_id,
                        "frame_number": i
                    }
                    for i, (timestamp, marker_id) in enumerate(self.marker_change_log)
                ]
            }
            
            filename = f"marker_log_{int(start_time)}.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
                print(f"\nMarker change log saved to: {filename}")
                print("JSON structure:")
                print("  - session_info: 세션 요약 정보")
                print("  - marker_changes: 각 마커 변경의 상세 정보")
            except Exception as e:
                print(f"Could not save log file: {e}")
        
        print("ArUco marker generator stopped")

if __name__ == "__main__":
    # ArUco 마커 생성기 실행
    generator = ArucoMarkerGenerator(target_fps=90, window_size=(800, 600))
    
    try:
        generator.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        generator.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        generator.cleanup()