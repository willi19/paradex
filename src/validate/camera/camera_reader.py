#!/usr/bin/env python3
"""
Camera Reader Quick Start Guide
================================

가장 빠르게 시작하는 방법
"""

from paradex.io.camera_system.camera_reader import discover_cameras, CameraReader, MultiCameraReader
import cv2


def quickstart_1():
    """
    퀵스타트 1: 카메라 찾기
    """
    print("=" * 60)
    print("QUICKSTART 1: 카메라 찾기")
    print("=" * 60)
    
    cameras = discover_cameras()
    
    if cameras:
        print(f"\n✓ {len(cameras)}개의 카메라를 찾았습니다:")
        for i, cam in enumerate(cameras, 1):
            print(f"  {i}. {cam}")
    else:
        print("\n✗ 카메라를 찾을 수 없습니다.")
        print("  Camera 클래스를 먼저 실행해주세요.")
    
    return cameras


def quickstart_2():
    """
    퀵스타트 2: 단일 카메라에서 이미지 읽기
    """
    print("\n" + "=" * 60)
    print("QUICKSTART 2: 단일 카메라에서 이미지 읽기")
    print("=" * 60)
    
    cameras = discover_cameras()
    if not cameras:
        print("카메라를 찾을 수 없습니다.")
        return
    
    camera_name = cameras[0]
    print(f"\n카메라 '{camera_name}' 사용")
    
    try:
        with CameraReader(camera_name) as reader:
            print("✓ 연결 성공")
            
            # 10개 프레임 읽기
            print("\n10개 프레임 읽는 중...")
            for i in range(10):
                image, frame_id = reader.get_image()
                print(f"  프레임 {i+1}: ID={frame_id}, 크기={image.shape}")
            
            print("\n✓ 완료!")
    
    except Exception as e:
        print(f"\n✗ 오류: {e}")


def quickstart_3():
    """
    퀵스타트 3: 모든 카메라에서 동시에 읽기
    """
    print("\n" + "=" * 60)
    print("QUICKSTART 3: 모든 카메라에서 동시에 읽기")
    print("=" * 60)
    
    try:
        # camera_names=None이면 자동으로 모든 카메라 탐색
        with MultiCameraReader() as multi_reader:
            print(f"\n✓ {len(multi_reader.camera_names)}개 카메라 연결됨")
            print(f"  카메라 목록: {multi_reader.camera_names}")
            
            # 5개 프레임 읽기
            print("\n5개 프레임 읽는 중...")
            for i in range(5):
                images_dict = multi_reader.get_images()
                
                print(f"  프레임 세트 {i+1}:")
                for cam_name, (image, frame_id) in images_dict.items():
                    print(f"    - {cam_name}: Frame ID={frame_id}")
            
            print("\n✓ 완료!")
    
    except Exception as e:
        print(f"\n✗ 오류: {e}")


def quickstart_4():
    """
    퀵스타트 4: 실시간 영상 표시 (첫 번째 카메라)
    """
    print("\n" + "=" * 60)
    print("QUICKSTART 4: 실시간 영상 표시")
    print("=" * 60)
    print("'q'를 눌러 종료")
    
    cameras = discover_cameras()
    if not cameras:
        print("카메라를 찾을 수 없습니다.")
        return
    
    camera_name = cameras[0]
    print(f"\n카메라 '{camera_name}' 사용")
    
    try:
        with CameraReader(camera_name) as reader:
            print("✓ 연결 성공")
            print("\n실시간 영상을 표시합니다...")
            
            last_frame_id = 0
            
            while True:
                # 새 프레임 대기
                image, frame_id = reader.wait_for_new_frame(
                    last_frame_id=last_frame_id,
                    timeout=0.1
                )
                
                if image is not None:
                    # Frame ID 표시
                    display_image = image.copy()
                    cv2.putText(
                        display_image,
                        f"Frame: {frame_id}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.imshow(f'Camera: {camera_name}', display_image)
                    last_frame_id = frame_id
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
            print("\n✓ 종료")
    
    except Exception as e:
        print(f"\n✗ 오류: {e}")
        cv2.destroyAllWindows()


def main():
    """메인 함수 - 모든 퀵스타트 실행"""
    
    print("\n")
    print("=" * 60)
    print(" Camera Reader - Quick Start Guide")
    print("=" * 60)
    print("\n이 스크립트는 Camera Reader의 기본 사용법을 보여줍니다.\n")
    
    # 1. 카메라 찾기
    cameras = quickstart_1()
    
    if not cameras:
        print("\n" + "=" * 60)
        print("카메라를 찾을 수 없습니다.")
        print("Camera 클래스를 먼저 실행해주세요:")
        print("=" * 60)
        print("""
from paradex.io.camera_system.camera_loader import CameraLoader

loader = CameraLoader()
loader.load_pyspin_camera()
loader.start(mode="stream", syncMode=False)
        """)
        return
    
    # 2. 단일 카메라 읽기
    quickstart_2()
    
    # 3. 다중 카메라 읽기
    quickstart_3()
    
    # 4. 실시간 영상 (선택사항)
    print("\n" + "=" * 60)
    print("실시간 영상을 보시겠습니까? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        quickstart_4()
    
    print("\n" + "=" * 60)
    print("Quick Start Guide 완료!")
    print("=" * 60)
    print("\n자세한 내용은 다음을 참고하세요:")
    print("  - README.md: 전체 문서")
    print("  - camera_reader_examples.py: 더 많은 예제")


if __name__ == "__main__":
    main()