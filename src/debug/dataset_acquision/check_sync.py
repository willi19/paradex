import numpy as np
import os

index_list = [13, 14, 15, 16, 17]

for ind in index_list:
    npy_path = f"/home/temp_id/shared_data/debug_/aruco/{ind}/marker2D"
    npy_list = os.listdir(npy_path)
    
    marker_dict = {}
    
    for npy_file in npy_list:
        cnt = 0
        
        npy_file_path = os.path.join(npy_path, npy_file)
        data = np.load(npy_file_path, allow_pickle=True).item()
        serial_num = npy_file.split(".")[0]
        
        for ts, marker in data.items():
            if len(marker) == 0:
                continue
            
            if 176 in marker:
                del marker[176]
            
            cnt += 1    
            if len(marker) > 1:
                print(f"Warning: Multiple markers detected at timestamp {ts} for serial {serial_num}. {marker.keys()}")
                
            for marker_id in marker.keys():
                if ts not in marker_dict:
                    marker_dict[ts] = {}
                if marker_id not in marker_dict[ts]:
                    marker_dict[ts][marker_id] = []
                    
                marker_dict[ts][marker_id].append(serial_num)
        if cnt == 0:
            print(f"Warning: No markers detected in {npy_file_path}")
            
    # 타임스탬프 순으로 정렬하여 출력
    for ts in sorted(marker_dict.keys()):
        markers = marker_dict[ts]
        if len(markers) > 1:
            print(f"Timestamp {ts} has multiple markers: {markers}")