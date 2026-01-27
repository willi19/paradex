import numpy as np
import cv2
import os
from typing import Tuple

def load_image(image_dir: str, cam_id: str, frame_idx: int, size: Tuple[int, int]) -> np.ndarray:
    # Frames are stored as 1-indexed jpgs (00001.jpg, ...).
    filename = os.path.join(image_dir, cam_id, f"{frame_idx:05d}.jpg")
    if os.path.exists(filename):

        image = cv2.imread(filename)
        image = cv2.resize(image, (size[1], size[0]))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return np.zeros((size[0], size[1], 3), dtype=np.uint8)

def load_series(data_dir: str, candidates: Tuple[str, ...]) -> Tuple[np.ndarray, np.ndarray]:
    # Load first existing npy among candidates and its time.npy if present.
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            data = np.load(path)
            time_path = os.path.join(data_dir, "time.npy")
            if os.path.exists(time_path):
                t = np.load(time_path)
            else:
                t = np.arange(data.shape[0], dtype=float)
            # Align lengths if off-by-one between data and time.
            if len(t) != data.shape[0]:
                n = min(len(t), data.shape[0])
                data = data[:n]
                t = t[:n]
            return data, t
    raise FileNotFoundError(f"No data found in {data_dir} for {candidates}")

# def resample_to(times_src: np.ndarray, data_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
#     # Simple per-joint linear interpolation onto destination timestamps.
#     if data_src.shape[0] == times_dst.shape[0] and np.allclose(times_src, times_dst):
#         return data_src
#     order = np.argsort(times_src)
#     times_src = times_src[order]
#     data_src = data_src[order]
#     out = np.zeros((times_dst.shape[0], data_src.shape[1]), dtype=float)
#     for j in range(data_src.shape[1]):
#         out[:, j] = np.interp(times_dst, times_src, data_src[:, j])
#     return out

def resample_to(data_times, data, pc_times):
    """
    2-pointer 방식으로 pc_times와 가장 가까운 data_times의 데이터를 매칭
    """
    synced_data = []
    n = len(pc_times)
    m = len(data_times)

    i = 0  # pc_times pointer
    j = 0  # data_times pointer

    while i < n:
        # data_times[j]가 pc_time[i]보다 작으면 j를 앞으로
        while j + 1 < m and abs(data_times[j + 1] - pc_times[i]) <= abs(data_times[j] - pc_times[i]):
            j += 1
        synced_data.append(data[j])
        i += 1

    return np.array(synced_data)
