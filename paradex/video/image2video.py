import cv2
import subprocess
import os

def images_to_video(image_files, output_video_path, frame_rate):
    """Convert images to a video using OpenCV."""
    if not image_files:
        print("No images found to create a video.")
        return

    # Read the first image to get size
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    temp_video_path = output_video_path.replace(".mp4", "_temp.mp4")

    video_writer = cv2.VideoWriter(temp_video_path, fourcc, frame_rate, (width, height))

    # Write images to video
    for image_file in sorted(image_files):  # Ensure frames are in correct order
        frame = cv2.imread(image_file)
        video_writer.write(frame)

    video_writer.release()    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # 기존 파일 덮어쓰기
        "-i", temp_video_path,  # 입력 파일
        "-c:v", "libx264",  # 비디오 코덱: H.264
        "-preset", "slow",  # 압축률과 속도 조절 (slow = 고품질)
        "-crf", "23",  # 품질 설정 (낮을수록 고품질, 18~23 추천)
        "-pix_fmt", "yuv420p",  # 픽셀 포맷 (H.264 표준 호환)
        output_video_path
    ]

    # FFmpeg 실행
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"✅ H.264 encoded video saved: {output_video_path}")
        os.remove(temp_video_path)  # 변환 후 임시 파일 삭제
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg encoding failed: {e}")

    print(f"Video saved at {output_video_path}")