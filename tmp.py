import os
import shutil
from pathlib import Path

def download_pick_videos(base_path, output_dir, start_idx=0, end_idx=54):
    """
    0부터 35까지 디렉토리에서 pick.mp4 파일들을 모두 복사/다운로드
    
    Args:
        base_path (str): 기본 경로
        output_dir (str): 다운로드할 목적지 폴더
        start_idx (int): 시작 인덱스
        end_idx (int): 종료 인덱스
    """
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_count = 0
    failed_files = []
    
    for i in range(start_idx, end_idx + 1):
        # 각 디렉토리의 pick.mp4 경로
        source_file = os.path.join(base_path, str(i), "pick.mp4")
        
        # 목적지 파일명 (인덱스 포함)
        dest_file = os.path.join(output_dir, f"pick_{i:03d}.mp4")
        
        try:
            if os.path.exists(source_file):
                # 파일 복사
                shutil.copy2(source_file, dest_file)
                print(f"✓ 다운로드 완료: {i}/pick.mp4 -> pick_{i:03d}.mp4")
                downloaded_count += 1
            else:
                print(f"✗ 파일 없음: {source_file}")
                failed_files.append(i)
                
        except Exception as e:
            print(f"✗ 오류 발생 ({i}): {str(e)}")
            failed_files.append(i)
    
    print(f"\n=== 다운로드 완료 ===")
    print(f"성공: {downloaded_count}개 파일")
    print(f"실패: {len(failed_files)}개 파일")
    
    if failed_files:
        print(f"실패한 인덱스: {failed_files}")

# 사용 예시
base_path = "/home/temp_id/shared_data/capture/lookup/pringles"
output_directory = "/home/temp_id/downloads/pringles_videos"  # 다운로드할 폴더

# 0부터 35까지 모든 pick.mp4 파일 다운로드
download_pick_videos(base_path, output_directory, 0, 54)

# 특정 범위만 다운로드하고 싶다면:
# download_pick_videos(base_path, output_directory, 10, 20)  # 10~20번만