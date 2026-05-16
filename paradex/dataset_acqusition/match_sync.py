import os
import glob
import json
import numpy as np

td = 2 / 30
def fill_framedrop(frame_id, pc_time):
    real_start = 10 # skip first 10 frames to avoid startup issues
    
    frame_id = frame_id[real_start:]
    pc_time = np.array(pc_time)[real_start:]
    
    time_delta = (pc_time[-1] - pc_time[0]) / (frame_id[-1] - frame_id[0])
    offset = np.mean(pc_time - (np.array(frame_id)-1)*time_delta)
    frame_id_nodrop = np.arange(1, frame_id[-1] + 500)
    pc_time_nodrop = (frame_id_nodrop - 1) * time_delta + offset - td
    return pc_time_nodrop, frame_id_nodrop

def get_synced_data(pc_times, data, data_times):
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


SENSOR_DIRS = ["arm", "hand", "teleop"]


def _collect_sensors(raw_dir):
    sensors = []
    for sensor_name in SENSOR_DIRS:
        sensor_path = os.path.join(raw_dir, sensor_name)
        time_path = os.path.join(sensor_path, "time.npy")
        if not os.path.isdir(sensor_path) or not os.path.exists(time_path):
            continue
        sensor_times = np.load(time_path, allow_pickle=True)
        if len(sensor_times) == 0:
            continue
        sensors.append((sensor_name, sensor_path, sensor_times))
    return sensors


def _sync_array(arr, ref_time, sensor_time):
    arr = np.asarray(arr)
    if len(arr) == 0:
        return arr
    return get_synced_data(ref_time, arr, sensor_time)


def _save_synced_dir(in_dir, out_dir, ref_time, sensor_time):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "time.npy"), ref_time)
    for fname in os.listdir(in_dir):
        if fname == "time.npy":
            continue
        src = os.path.join(in_dir, fname)
        if not os.path.isfile(src):
            continue
        data = np.load(src, allow_pickle=True)
        # teleop's left.npy/right.npy are dicts saved as 0-d object arrays
        if data.ndim == 0 and isinstance(data.item(), dict):
            d = data.item()
            synced = {k: _sync_array(v, ref_time, sensor_time) for k, v in d.items()}
            np.save(os.path.join(out_dir, fname), synced)
        else:
            np.save(os.path.join(out_dir, fname), _sync_array(data, ref_time, sensor_time))


def _video_frame_count(videos_dir):
    """Frame count from the first usable video (blank-fill => uniform)."""
    import cv2
    for p in sorted(glob.glob(os.path.join(videos_dir, "*.avi")) +
                    glob.glob(os.path.join(videos_dir, "*.mp4"))):
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    return 0


def synthesize_camera_timeline(session_dir):
    """
    Fake the (frame_id -> pc_time) bridge a TimestampMonitor would provide,
    using camera_meta.json (main-PC wall clock at camera trigger) + the fact
    that dropped frames are blank-filled (so video frame i is uniform: the
    i-th frame is at start_time + i/fps).

    Returns ref_time array of length == video frame count, on the main-PC
    clock, or None if prerequisites are missing.
    """
    meta_path = os.path.join(session_dir, "camera_meta.json")
    videos_dir = os.path.join(session_dir, "videos")
    if not os.path.exists(meta_path) or not os.path.isdir(videos_dir):
        return None
    meta = json.load(open(meta_path))
    fps = float(meta.get("fps", 30))
    start = float(meta["start_time"])
    offset = float(meta.get("offset_ms", 0.0)) / 1000.0
    n = _video_frame_count(videos_dir)
    if n == 0:
        return None
    return start + offset + np.arange(n) / fps


def postprocess_session(session_dir, ref_fps=None):
    """
    Sync arm/hand/teleop data onto a common timeline.

    - With cameras (timestamps/ exists): use camera frame_id timeline (fill_framedrop).
    - Without cameras: use the highest-rate sensor's timeline as reference,
      or a uniform `ref_fps` grid covering the overlap window if given.

    Reads from {session_dir}/raw/{arm,hand,teleop}/ and writes synced data to
    {session_dir}/{arm,hand,teleop}/.
    """
    raw_dir = os.path.join(session_dir, "raw")
    if not os.path.isdir(raw_dir):
        print(f"[postprocess] raw dir not found: {raw_dir}")
        return

    ts_dir = os.path.join(raw_dir, "timestamps")
    sensors = _collect_sensors(raw_dir)
    if not sensors:
        print(f"[postprocess] no sensor data under {raw_dir}")
        return

    synth = synthesize_camera_timeline(session_dir)
    if os.path.exists(os.path.join(ts_dir, "frame_id.npy")) and \
       os.path.exists(os.path.join(ts_dir, "timestamp.npy")):
        frame_id = np.load(os.path.join(ts_dir, "frame_id.npy"))
        pc_time = np.load(os.path.join(ts_dir, "timestamp.npy"))
        ref_time, _ = fill_framedrop(frame_id, pc_time)
        print(f"[postprocess] camera-based timeline (TimestampMonitor): {len(ref_time)} frames")
    elif synth is not None:
        ref_time = synth
        print(f"[postprocess] synthesized camera timeline from camera_meta.json: "
              f"{len(ref_time)} frames (== video frame count, frame i <-> action i)")
    else:
        if ref_fps is not None:
            t_start = max(s[2][0] for s in sensors)
            t_end = min(s[2][-1] for s in sensors)
            if t_end <= t_start:
                print(f"[postprocess] no overlap between sensors: {t_start} > {t_end}")
                return
            n = int((t_end - t_start) * ref_fps)
            ref_time = np.linspace(t_start, t_end, n)
            print(f"[postprocess] uniform timeline @ {ref_fps}Hz: {n} samples")
        else:
            ref_name, _, ref_time = max(sensors, key=lambda s: len(s[2]))
            print(f"[postprocess] using '{ref_name}' as reference timeline: {len(ref_time)} samples")

    for sensor_name, sensor_path, sensor_time in sensors:
        out_dir = os.path.join(session_dir, sensor_name)
        _save_synced_dir(sensor_path, out_dir, ref_time, sensor_time)
        print(f"[postprocess] synced {sensor_name} -> {out_dir}")
