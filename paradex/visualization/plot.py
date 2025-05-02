import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import subprocess
from dex_robot.utils.file_io import load_contact_value, download_path
from dex_robot.io.contact.process import process_contact
from dex_robot.io.contact.index import sensor_name
from dex_robot.visualization.convert_codec import change_to_h264
from dex_robot.utils.file_io import load_camparam, load_robot_traj, load_c2r
import tqdm
from dex_robot.io.contact.index import sensor_name, contact_sensor_idx
from paradex.utils.merge_video import merge_video_synced
from dex_robot.simulator.isaac import simulator as sim
from multiprocessing import Process

def plot_contact_graph_data(demo_path, output_video_path, time_range=3, sampling_rate=30, overwrite=True):
    """Plots contact sensor data and saves as a video using OpenCV (MP4) and converts to H.264 using FFmpeg."""
    try:
        print(demo_path)
        contact_value = load_contact_value(demo_path)
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
    ax.set_ylim(0, 100)
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

    for frame in tqdm.tqdm(range(T)):
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

def plot_projection(demo_path, output_video_path, overwrite=True):
    if not overwrite and os.path.exists(output_video_path):
        print(f"Skipping existing projection file: {output_video_path}")
        return
    
    simulator = sim(
        None,
        view_physics=False,
        view_replay=True,
        headless=True,
        save_video=True,
        save_state=False,
        fixed=True,
        add_plane=False
    )

    try:
        intrinsic, extrinsic = load_camparam(demo_path)
        robot_traj = load_robot_traj(demo_path)
        C2R = load_c2r(demo_path)

    except Exception as e:
        print("File not found {}".format(e))
        return
    
    temp_path = output_video_path + "_temp"
    os.makedirs(temp_path, exist_ok=True)

    cam_dict = {}
    for serial_num, param in intrinsic.items():
        int_mat = np.array(param['Intrinsics']).reshape(3,3)
        ext_mat = np.array(extrinsic[serial_num])
        ext_mat = np.concatenate([ext_mat, np.array([[0,0,0,1]])], axis=0)
        ext_mat = np.linalg.inv(C2R) @ np.linalg.inv(ext_mat)
        ext_mat = ext_mat[:3]
        cam_dict[serial_num] = (int_mat, ext_mat)
    
    simulator.load_camera(cam_dict)
    simulator.set_savepath(temp_path, output_video_path)
    T = robot_traj.shape[0]
    for step in range(T):
        state = robot_traj[step]
        simulator.step(state, state, None)
    simulator.save()

    os.makedirs(output_video_path, exist_ok=True)
    for vid_name in os.listdir(temp_path):
        change_to_h264(os.path.join(temp_path, vid_name), os.path.join(output_video_path, vid_name))
    subprocess.run(["rm", "-r", temp_path])
    print(f"Saved projection video: {output_video_path}")

def plot_contact(demo_path, output_video_path, overwrite=True):
    if not overwrite and os.path.exists(output_video_path):
        print(f"Skipping existing projection file: {output_video_path}")
        return
    
    simulator = sim(
        None,
        view_physics=False,
        view_replay=True,
        headless=True,
        save_video=True,
        save_state=False,
        fixed=False,
        add_plane=False
    )

    try:
        contact_value = load_contact_value(demo_path)
        
    except Exception as e:
        print("File not found {}".format(e))
        return
    
    temp_path = output_video_path.replace(".mp4", "_temp.mp4")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    simulator.load_camera()
    simulator.set_savepath(temp_path, output_video_path)

    T = contact_value.shape[0]
    contact_value = process_contact(contact_value)
    
    for step in range(T):
        target_action = np.zeros(22)
        simulator.step(target_action, target_action, None)
        color_dict = {}
        for ri, ci in contact_sensor_idx.items():
            val = contact_value[step, ci] / 100
            if val < 0:
                val = 0
            if val > 1:
                val = 1
            color_dict[ri] = (0.5+0.5*val, 0.5-0.2*val, 0.8-0.5*val)
        simulator.set_color(color_dict)
    simulator.save()
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    # for vid_name in os.listdir(temp_path):
    #     change_to_h264(os.path.join(temp_path, vid_name), os.path.join(output_video_path, vid_name))
    change_to_h264(temp_path, output_video_path)
    subprocess.run(["rm", "-r", temp_path])
    print(f"Saved contact video: {output_video_path}")

def plot_pose(demo_path, output_video_path, overwrite=True):
    if not overwrite and os.path.exists(output_video_path):
        print(f"Skipping existing projection file: {output_video_path}")
        return
    
    simulator = sim(
        None,
        view_physics=False,
        view_replay=True,
        headless=True,
        save_video=True,
        save_state=False,
        fixed=False,
        add_plane=False
    )

    try:
        robot_traj = load_robot_traj(demo_path)
        
    except Exception as e:
        print("File not found {}".format(e))
        return
    
    temp_path = output_video_path.replace(".mp4", "_temp.mp4")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    simulator.load_camera()
    simulator.set_savepath(temp_path, output_video_path)
    
    T = robot_traj.shape[0]
    for step in range(T):
        state = robot_traj[step]
        state[:6] = 0
        
        simulator.step(state, state, None)
    simulator.save()

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    # for vid_name in os.listdir(temp_path):
    #     change_to_h264(os.path.join(temp_path, vid_name), os.path.join(output_video_path, vid_name))
    change_to_h264(temp_path, output_video_path)
    subprocess.run(["rm", "-r", temp_path])
    print(f"Saved pose video: {output_video_path}")

def plot_overlay(obj_name, index, overwrite=True):
    intrinsic, _ = load_camparam(f"{download_path}/processed/{obj_name}/{index}")
    
    proj_video_list = os.listdir(f"video/projection/{obj_name}/{index}")
    video_list = os.listdir(f"{download_path}/processed/{obj_name}/{index}/video")
    w, h = 2048, 1536

    for proj_video in tqdm.tqdm(proj_video_list):
        serial_num = proj_video.split(".")[0]
        video_name = f"{serial_num}.avi"
        out_path = f"video/overlay/{obj_name}/{index}/{serial_num}.mp4"

        if not overwrite and os.path.exists(out_path):
            continue
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        proj_video = cv2.VideoCapture(f"video/projection/{obj_name}/{index}/{proj_video}")
        video = cv2.VideoCapture(f"{download_path}/processed/{obj_name}/{index}/video/{video_name}")
        temp_path = f"video/overlay/{obj_name}/{index}/{serial_num}_temp.mp4"
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        int_mat = np.array(intrinsic[serial_num]['Intrinsics']).reshape(3,3)
        old_int_mat = np.array([[int_mat[0,0], 0, w//2],[0, int_mat[0,0], h//2],[0, 0, 1]])
        H = int_mat @ np.linalg.inv(old_int_mat)

        len1 = int(proj_video.get(cv2.CAP_PROP_FRAME_COUNT))
        len2 = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        length = min(len1, len2)

        for i in tqdm.tqdm(range(length)):
            ret1, proj_frame = proj_video.read()
            ret2, frame = video.read()

            if not ret1 or not ret2:
                break

            # Warp the image to simulate the new intrinsics
            warped = cv2.warpPerspective(proj_frame, H, (w, h))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gamma=0.8
            alpha = np.clip(gray*gamma, 0, 1)[:, :, np.newaxis]  # (H, W, 1)
            
            alpha = np.repeat(alpha, 3, axis=2)
            warped = (warped * alpha).astype(np.uint8)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

            # ✅ 해당 영역만 밝기 줄이기
            dimmed_frame_masked = (frame_masked * 0.2).astype(np.uint8)

            # ✅ 원본 frame에 다시 덮어쓰기
            frame_dimmed = frame.copy()
            frame_dimmed[mask > 0] = dimmed_frame_masked[mask > 0]

            # ✅ 그 위에 warped를 합성 (그대로 밝기 조정된 배경 위에 덮기)
            fg_region = cv2.bitwise_and(warped, warped, mask=mask)
            # bg_region = cv2.bitwise_and(frame_dimmed, frame_dimmed, mask=cv2.bitwise_not(mask))
            frame = cv2.add(fg_region, frame_dimmed)
            out.write(frame)

        out.release()
        proj_video.release()
        video.release()
        
        change_to_h264(temp_path, out_path)

def plot_final(obj_name, index, overwrite=True):
    # if not overwrite and os.path.exists(f"video/final/{obj_name}/{index}.mp4"):
    #     print(f"Skipping {obj_name}/{index} (video exists)")
    #     return
    
    demo_path = f"{download_path}/processed/{obj_name}/{index}"
    output_video_path = f"video/contact_graph/{obj_name}/{index}.mp4"
    plot_contact_graph_data(demo_path, output_video_path, overwrite=overwrite)

    processes = []

    p1 = Process(target=plot_contact, args=(demo_path, f"video/contact/{obj_name}/{index}.mp4", overwrite))
    processes.append(p1)

    # p2 = Process(target=plot_projection, args=(demo_path, f"video/projection/{obj_name}/{index}", overwrite))
    # processes.append(p2)

    p3 = Process(target=plot_pose, args=(demo_path, f"video/pose/{obj_name}/{index}.mp4", overwrite))
    processes.append(p3)

    for p in processes:
        p.start()
        p.join()


    # plot_overlay(obj_name, index, overwrite=overwrite)

    os.makedirs(f"video/grid/{obj_name}", exist_ok=True)

    if not os.path.exists(f"video/grid/{obj_name}/{index}.mp4"):
        merge_video_synced(f"{demo_path}/video", f"video/grid/{obj_name}/{index}.mp4")

    pose_video = cv2.VideoCapture(f"video/pose/{obj_name}/{index}.mp4")
    contact_video = cv2.VideoCapture(f"video/contact/{obj_name}/{index}.mp4")
    grid_video = cv2.VideoCapture(f"video/grid/{obj_name}/{index}.mp4")
    graph_video = cv2.VideoCapture(f"video/contact_graph/{obj_name}/{index}.mp4")

    os.makedirs(f"video/final/{obj_name}", exist_ok=True)
    output_video_path = f"video/final/{obj_name}/{index}.mp4"
    temp_video_path = f"video/final/{obj_name}/{index}_tmp.mp4"

    merged_video = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 7, (4096, 1536))
    
    cnt = 0
    while True:
        finished = True
        ret, pose_img = pose_video.read()
        if not ret:
            pose_img = np.zeros((768, 1024, 3), dtype=np.uint8)
        else:
            finished = False

        ret, contact_img = contact_video.read()
        if not ret:
            contact_img = np.zeros((768, 1024, 3), dtype=np.uint8)
        else:
            finished = False

        ret, grid_img = grid_video.read()
        if not ret:
            grid_img = np.zeros((1536, 2048, 3), dtype=np.uint8)
        else:
            finished = False

        ret, graph_img = graph_video.read()
        if not ret:
            graph_img = np.zeros((768, 2048, 3), dtype=np.uint8)
        else:
            finished = False

        if finished:
            break
        
        cnt += 1
        merge_img = np.zeros((1536, 4096, 3), dtype=np.uint8)

        pose_img = cv2.resize(pose_img, (1024, 768))
        contact_img = cv2.resize(contact_img, (1024, 768))
        grid_img = cv2.resize(grid_img, (2048, 1536))
        graph_img = cv2.resize(graph_img, (2048, 768))

        merge_img[:768, 2048:] = graph_img
        merge_img[768:, 3072:] = pose_img
        merge_img[768:, 2048:3072] = contact_img
        merge_img[:, :2048] = grid_img

        merged_video.write(merge_img)


    merged_video.release()
    pose_video.release()
    contact_video.release()
    grid_video.release()
    graph_video.release()

    change_to_h264(temp_video_path, output_video_path)

