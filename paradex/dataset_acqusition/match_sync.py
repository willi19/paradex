import numpy as np

td = 2 / 30  #: fixed ~66 ms trigger-to-exposure latency offset baked for 30 fps
def fill_framedrop(frame_id, pc_time):
    """Reconstruct a dense, drop-free frame timeline from logged frame ids/times.

    Fits a linear ``frame_id -> time`` model (skipping the first 10 warmup frames)
    and emits a gap-free ``frame_id`` array from 1 to ``last + 500`` with
    reconstructed times, so downstream indexing stays aligned even where the camera
    dropped frames. Robust to drops because the slope divides by the frame-id span,
    not the frame count.

    Parameters
    ----------
    frame_id : numpy.ndarray
        Logged hardware frame ids (may skip values where frames dropped).
    pc_time : array-like
        PC wall-clock timestamp for each logged frame.

    Returns
    -------
    tuple of numpy.ndarray
        ``(pc_time_nodrop, frame_id_nodrop)`` — the dense reconstructed times and
        their frame ids. ``pc_time_nodrop`` has the module-level ``td`` offset
        subtracted.
    """
    real_start = 10 # skip first 10 frames to avoid startup issues
    
    frame_id = frame_id[real_start:]
    pc_time = np.array(pc_time)[real_start:]
    
    time_delta = (pc_time[-1] - pc_time[0]) / (frame_id[-1] - frame_id[0])
    offset = np.mean(pc_time - (np.array(frame_id)-1)*time_delta)
    frame_id_nodrop = np.arange(1, frame_id[-1] + 500)
    pc_time_nodrop = (frame_id_nodrop - 1) * time_delta + offset - td
    return pc_time_nodrop, frame_id_nodrop

def get_synced_data(pc_times, data, data_times):
    """Nearest-time resample of a sensor stream onto the camera frame clock.

    For each entry in ``pc_times`` returns the ``data`` row whose ``data_times`` is
    closest, using a monotone two-pointer scan. Nearest-neighbor, not interpolation.

    Parameters
    ----------
    pc_times : array-like
        Target timeline (typically the dense frame times from :func:`fill_framedrop`).
    data : numpy.ndarray
        Sensor samples, one row per ``data_times`` entry.
    data_times : array-like
        Timestamp of each ``data`` row. Both time arrays must be sorted ascending.

    Returns
    -------
    numpy.ndarray
        One ``data`` row per ``pc_times`` entry (length ``len(pc_times)``).
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
