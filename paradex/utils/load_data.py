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
            data = np.load(path, allow_pickle=True)
            time_path = os.path.join(data_dir, "time.npy")
            if os.path.exists(time_path):
                t = np.load(time_path, allow_pickle=True)
            else:
                t = np.arange(data.shape[0], dtype=float)
            # Align lengths if off-by-one between data and time.
            if len(t) != data.shape[0]:
                n = min(len(t), data.shape[0])
                data = data[:n]
                t = t[:n]
            return data, t
    raise FileNotFoundError(f"No data found in {data_dir} for {candidates}")



def resample_to(times_src: np.ndarray, data_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation onto destination timestamps."""
    if data_src.shape[0] == times_dst.shape[0] and np.allclose(times_src, times_dst):
        return data_src
    times_src = np.asarray(times_src, dtype=float)
    times_dst = np.asarray(times_dst, dtype=float)
    data_src = np.asarray(data_src, dtype=float)
    order = np.argsort(times_src)
    times_src = times_src[order]
    data_src = data_src[order]
    flat = data_src.reshape(data_src.shape[0], -1)
    out = np.zeros((times_dst.shape[0], flat.shape[1]), dtype=float)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(times_dst, times_src, flat[:, j])
    return out.reshape((times_dst.shape[0],) + data_src.shape[1:])
