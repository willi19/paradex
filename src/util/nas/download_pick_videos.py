import os
import shutil
from pathlib import Path

def download_pick_videos(base_path, output_dir):
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
    
    for i in os.listdir(base_path):
        # 각 디렉토리의 pick.mp4 경로
        source_file = os.path.join(base_path, str(i), "pick.mp4")
        
        # 목적지 파일명 (인덱스 포함)
        dest_file = os.path.join(output_dir, f"pick_{int(i):03d}.mp4")
        
        try:
            if os.path.exists(source_file):
                # 파일 복사
                shutil.copy2(source_file, dest_file)
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
for obj_name in ["book"]:
    base_path = f"/home/temp_id/shared_data/capture/lookup/{obj_name}"
    output_directory = f"/home/temp_id/downloads/{obj_name}"  # 다운로드할 폴더

    download_pick_videos(base_path, output_directory)