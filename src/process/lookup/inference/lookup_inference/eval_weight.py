import numpy as np
import os
import json
import pandas as pd
from collections import defaultdict
import cv2

from paradex.utils.file_io import shared_dir
from paradex.inference.object_6d import get_current_object_6d

def analyze_weight_test_results():
    """무게에 따른 정확도 분석"""
    
    # 결과 저장용: [grasp_type][captured_object][lookup_object][pose_index] = data
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'total_attempts': 0,
        'successful_attempts': 0,
        'precision_list': []
    }))))
    
    # 각 캡처된 객체별로 분석
    for captured_object in ["pringles", "pringles_light", "pringles_heavy"]:
        root_dir = os.path.join(shared_dir, "inference", "weight_test", captured_object)
        
        if not os.path.exists(root_dir):
            print(f"Directory not found: {root_dir}")
            continue
            
        # 각 grasp_type 폴더 순회
        for grasp_type in os.listdir(root_dir):
            grasp_type_path = os.path.join(root_dir, grasp_type)
            if not os.path.isdir(grasp_type_path):
                continue
                
            # 각 grasp_obj_type 폴더 순회
            for grasp_obj_type in os.listdir(grasp_type_path):
                grasp_obj_type_path = os.path.join(grasp_type_path, grasp_obj_type)
                if not os.path.isdir(grasp_obj_type_path):
                    continue
                
                # lookup object 이름 생성
                lookup_object = "pringles" + ("" if grasp_obj_type == "empty" else ("_" + grasp_obj_type))
                    
                # 각 pose_index 폴더 순회
                for pose_index in os.listdir(grasp_obj_type_path):
                    pose_index_path = os.path.join(grasp_obj_type_path, pose_index)
                    if not os.path.isdir(pose_index_path):
                        continue
                        
                    # 각 시도 폴더 순회 (반복 횟수)
                    for attempt_index in os.listdir(pose_index_path):
                        attempt_path = os.path.join(pose_index_path, attempt_index)
                        if not os.path.isdir(attempt_path):
                            continue
                            
                        # result.json 파일 확인
                        result_file = os.path.join(attempt_path, "result.json")
                        if not os.path.exists(result_file):
                            continue
                            
                        # 결과 로드
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        # 성공 여부 확인
                        success = result_data.get('success', False)
                        
                        # 통계 업데이트 (pose_index별로 분리)
                        results[grasp_type][captured_object][lookup_object][pose_index]['total_attempts'] += 1
                        
                        if success:
                            
                            # precision 계산
                            # place_6d_file = os.path.join(attempt_path, "place_6D.npy")
                            img_dir = os.path.join(attempt_path, "place")
                            img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}
                            
                            if len(img_dict) == 0:
                                continue
                            
                            try:
                                if not os.path.exists(os.path.join(attempt_path, "place_6D.npy")):
                                    cur_6D = get_current_object_6d(captured_object, True, img_dict)
                                    np.save(os.path.join(attempt_path, "place_6D.npy"), cur_6D)
                                else:
                                    cur_6D = np.load(os.path.join(attempt_path, "place_6D.npy"))
                                    if grasp_type == "lay" and captured_object == "pringles" and grasp_obj_type == "pringles":
                                        print(captured_object, grasp_obj_type)
                            except:
                                continue
                                                
                            target_6d_file = os.path.join(attempt_path, "target_6D.npy")
                            
                            if os.path.exists(target_6d_file):
                                try:
                                    place_6D = np.load(target_6d_file)
                                    # Z 방향 확인 (성공 기준)
                                    if cur_6D[2,2] > 0.7:
                                        # 거리 계산 (mm 단위)
                                        distance_mm = np.linalg.norm(
                                            np.array(cur_6D[:2,3]) - np.array(place_6D[:2,3])
                                        ) * 1000
                                        
                                        results[grasp_type][captured_object][lookup_object][pose_index]['precision_list'].append(distance_mm)
                                        results[grasp_type][captured_object][lookup_object][pose_index]['successful_attempts'] += 1
                                    else:
                                        print(cur_6D[2,2])
                                except Exception as e:
                                    print(f"Error processing {attempt_path}: {e}")
    
    return results

def create_pose_tables(results):
    """각 pose별로 별도 표 생성"""
    
    grasp_types = ["up", "tip", "palm", "tripod", "lay"]
    captured_objects = ["pringles", "pringles_light", "pringles_heavy"]
    lookup_objects = ["pringles", "pringles_light", "pringles_heavy"]
    
    tables = {}
    
    for grasp_type in grasp_types:
        if grasp_type not in results:
            continue
            
        # Success Rate 표
        success_data = []
        precision_data = []
        attempts_data = []
        
        for captured_obj in captured_objects:
            success_row = {"Captured Object": captured_obj}
            precision_row = {"Captured Object": captured_obj}
            attempts_row = {"Captured Object": captured_obj}
            
            for lookup_obj in lookup_objects:
                if (captured_obj in results[grasp_type] and 
                    lookup_obj in results[grasp_type][captured_obj]):
                    
                    pose_data = results[grasp_type][captured_obj][lookup_obj]
                    
                    for pose_idx, data in pose_data.items():
                        column_name = f"{lookup_obj}_{pose_idx}"
                        total = data['total_attempts']
                        success = data['successful_attempts']
                        
                        # Success Rate
                        success_rate = success / total if total > 0 else 0
                        success_row[column_name] = f"{success_rate:.2f} ({success}/{total})"
                        
                        # Precision
                        if len(data['precision_list']) > 0:
                            avg_precision = np.mean(data['precision_list'])
                            precision_row[column_name] = f"{avg_precision:.2f} mm"
                        else:
                            precision_row[column_name] = "N/A"
                        
                        # Attempts
                        attempts_row[column_name] = total
            
            success_data.append(success_row)
            precision_data.append(precision_row)
            attempts_data.append(attempts_row)
        
        tables[grasp_type] = {
            'success_rate': pd.DataFrame(success_data),
            'precision': pd.DataFrame(precision_data),
            'attempts': pd.DataFrame(attempts_data)
        }
    
    return tables

def print_pose_tables(tables):
    """각 pose별 표 출력"""
    
    for grasp_type, table_dict in tables.items():
        print(f"\n{'='*60}")
        print(f"GRASP TYPE: {grasp_type.upper()}")
        print(f"{'='*60}")
        
        print(f"\n--- Success Rate ---")
        print(table_dict['success_rate'].to_string(index=False))
        
        print(f"\n--- Precision (mm) ---")
        print(table_dict['precision'].to_string(index=False))
        
        print(f"\n--- Number of Attempts ---")
        print(table_dict['attempts'].to_string(index=False))

def save_pose_tables(tables, output_dir):
    """각 pose별 표를 별도 CSV 파일로 저장"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for grasp_type, table_dict in tables.items():
        # Success Rate 표
        success_file = os.path.join(output_dir, f"{grasp_type}_success_rate.csv")
        table_dict['success_rate'].to_csv(success_file, index=False)
        
        # Precision 표
        precision_file = os.path.join(output_dir, f"{grasp_type}_precision.csv")
        table_dict['precision'].to_csv(precision_file, index=False)
        
        # Attempts 표
        attempts_file = os.path.join(output_dir, f"{grasp_type}_attempts.csv")
        table_dict['attempts'].to_csv(attempts_file, index=False)
    
    print(f"\nTables saved to: {output_dir}")

def main():
    """메인 실행 함수"""
    print("Analyzing weight test results by pose...")
    
    # 결과 분석
    results = analyze_weight_test_results()
    
    # pose별 표 생성
    tables = create_pose_tables(results)
    
    # 결과 출력
    print_pose_tables(tables)
    
    # CSV 파일로 저장
    output_dir = os.path.join(shared_dir, "inference", "weight_test_analysis")
    save_pose_tables(tables, output_dir)
    
    print("Analysis complete. CSV files created for copying.")

if __name__ == "__main__":
    main()