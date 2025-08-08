import numpy as np
import os
from collections import defaultdict

index_list = [19, 20, 21]

for ind in index_list:
    npy_path = f"/home/temp_id/shared_data/debug_/aruco/{ind}/marker2D"
    npy_list = os.listdir(npy_path)
    
    marker_dict = {}
    marker_offset = {}
    
    for npy_file in npy_list:
        cnt = 0
        
        npy_file_path = os.path.join(npy_path, npy_file)
        data = np.load(npy_file_path, allow_pickle=True).item()
        serial_num = npy_file.split(".")[0]
        marker_offset[serial_num] = {"ts": [], "marker": []}
        
        for ts, marker in data.items():
            if len(marker) == 0:
                continue
            
            if 157 in marker:
                del marker[157]
            
            if 195 in marker:
                del marker[195]
                
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
    
    # 시리얼 쌍별 차이 저장
    pair_diffs = defaultdict(list)
    
    # 타임스탬프 순으로 정렬하여 마커 차이 계산
    for ts in sorted(marker_dict.keys()):
        markers = marker_dict[ts]
        if len(markers) >= 1:
            marker_ids = list(markers.keys())
            # print(f"Timestamp {ts} has multiple markers: {marker_ids}")
            
            # 각 마커와 해당 시리얼들의 쌍을 만들기
            marker_serial_pairs = []
            for marker_id in marker_ids:
                for serial in markers[marker_id]:
                    marker_serial_pairs.append((marker_id, serial))
            
            # print(f"  Marker-Serial pairs: {marker_serial_pairs}")
            
            # 모든 시리얼 쌍에 대해 마커 차이 계산
            for i in range(len(marker_serial_pairs)):
                for j in range(i + 1, len(marker_serial_pairs)):
                    marker1, serial1 = marker_serial_pairs[i]
                    marker2, serial2 = marker_serial_pairs[j]
                    
                    # 시리얼 번호 순서로 정렬 (일관성 위해)
                    if serial1 > serial2:
                        serial1, serial2 = serial2, serial1
                        marker1, marker2 = marker2, marker1
                    
                    # 마커 차이 계산
                    diff = marker2 - marker1
                    
                    # 250 주기 보정
                    if diff > 125:
                        diff -= 250
                    elif diff < -125:
                        diff += 250
                    
                    pair_key = (serial1, serial2)
                    pair_diffs[pair_key].append({
                        'timestamp': ts,
                        'marker1': marker1,
                        'marker2': marker2,
                        'diff': diff
                    })
                    
                    if diff > 10:
                        print(marker1, marker2, serial1, serial2, diff, "diff > 10")
                    
                    # print(f"    {serial1} (marker {marker1}) vs {serial2} (marker {marker2}) = diff {diff}")
    
    # print(f"\n=== Summary of Pair Differences ===")
    # print(f"{'Serial1':<12} {'Serial2':<12} {'Count':<6} {'Avg Diff':<10} {'Std Dev':<10} {'Min':<5} {'Max':<5}")
    print("-" * 70)
    
    for (serial1, serial2), diff_list in pair_diffs.items():
        if len(diff_list) >= 1:  # 최소 5개 이상의 데이터가 있는 경우만
            diffs = [d['diff'] for d in diff_list]
            avg_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            min_diff = np.min(diffs)
            max_diff = np.max(diffs)
            
            print(f"{serial1:<12} {serial2:<12} {len(diffs):<6} {avg_diff:<10.2f} {std_diff:<10.2f} {min_diff:<5} {max_diff:<5}")
            
            # # 첫 10개 세부 내용 출력
            # print(f"  First 10 differences: {diffs[:10]}")
    
    # print(f"\n=== Detailed Pair Analysis ===")
    # for (serial1, serial2), diff_list in pair_diffs.items():
    #     if len(diff_list) >= 10:
    #         print(f"\n{serial1} vs {serial2}:")
    #         for i, d in enumerate(diff_list[:20]):  # 처음 20개만
    #             print(f"  TS {d['timestamp']}: {d['marker1']} vs {d['marker2']} = {d['diff']}")
    #         if len(diff_list) > 20:
    #             print(f"  ... and {len(diff_list) - 20} more")