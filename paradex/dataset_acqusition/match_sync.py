
def fill_framedrop(cam_timestamp):
    frameID = cam_timestamp["frameID"]
    real_start = -1
    for i, fi in enumerate(frameID):
        if fi > 5:
            real_start = i
            break
    
    frameID = frameID[real_start:]
    pc_time = np.array(cam_timestamp["pc_time"])[real_start:]
    timestamp = np.array(cam_timestamp["timestamps"])
    time_delta = (pc_time[-1] - pc_time[0]) / (frameID[-1] - frameID[0])
    offset = np.mean(pc_time - (np.array(frameID)-1)*time_delta)
    pc_time_nodrop = []
    frameID_nodrop = []

    time_delta_new = 1 / 30
    
    if time_delta / time_delta_new > 1.01:
        return None, None
    
    for i in range(1, frameID[-1] + 10):
        frameID_nodrop.append(i)
        pc_time_nodrop.append((i-1)*time_delta_new+offset - td)
    
    return pc_time_nodrop, frameID_nodrop

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