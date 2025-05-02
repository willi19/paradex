import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import subprocess
from dex_robot.utils.file_io import load_contact_value
from dex_robot.contact.process import process_contact
from dex_robot.contact.index import sensor_name
from dex_robot.visualization.convert_codec import change_to_h264

def plot_contact_data(demo_path, output_video_path, time_range=30, sampling_rate=30, overwrite=False):
    """Plots contact sensor data and saves as a video using OpenCV (MP4) and converts to H.264 using FFmpeg."""
    try:
        contact_value = load_contact_value(os.path.join(demo_path))
    except:
        print("File not found")
        return

    if not overwrite and os.path.exists(output_video_path):
        print(f"Skipping existing file: {output_video_path}")
        return
    # Determine number of samples corresponding to max_time_seconds
    max_samples = time_range * sampling_rate

    # Generate time axis based on sampling rate
    time_axis = np.linspace(0, time_range, max_samples)

    T = contact_value.shape[0]

    # OpenCV VideoWriter 설정 (mp4v 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    fps = 30
    frame_size = (1920, 1080)  

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    temp_video_path = output_video_path.replace(".mp4", "_temp.mp4")  # 임시 파일 생성
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, frame_size)

    # Process contact data
    contact_value_trimmed = process_contact(contact_value)

    # **Figure 한 번만 생성**
    fig, ax = plt.subplots(figsize=(19.2, 10.8))  
    ax.set_ylim(0, 550)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Contact Sensor Value")
    ax.set_title("Contact Sensor Readings")
    ax.grid()

    # 초기 plot 생성 (빈 그래프)
    lines = []
    for i in range(15):  
        line, = ax.plot(time_axis, np.zeros_like(time_axis), label=sensor_name[i])
        lines.append(line)

    # Legend 추가
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for frame in range(T):
        cv_trimmed = np.zeros((max_samples, 15))
        cv_trimmed[max(0, max_samples - frame):, :] = contact_value_trimmed[max(0, frame - max_samples):frame, :]
        for i, line in enumerate(lines):
            line.set_ydata(cv_trimmed[:, i])

        # Matplotlib 그래프를 OpenCV 이미지로 변환
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # OpenCV 형식으로 변환 및 크기 조정
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, frame_size)

        # OpenCV 비디오에 프레임 추가
        out.write(img)

    # 비디오 저장 완료 후 해제
    out.release()
    change_to_h264(temp_video_path, output_video_path)

