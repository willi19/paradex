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
