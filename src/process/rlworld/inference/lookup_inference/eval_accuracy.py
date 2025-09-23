import numpy as np
import os
import json
import pandas as pd
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from paradex.utils.file_io import shared_dir
from paradex.inference.object_6d import get_current_object_6d

def analyze_weight_test_results():
    """무게에 따른 정확도 분석"""
    
    # 결과 저장용: [grasp_type][captured_object][lookup_object][pose_index] = data
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'total_attempts': 0,
        'successful_attempts': 0,
        'offset': [],
        'attemp_idx': [],
    })))
    
    # 각 캡처된 객체별로 분석
    for captured_object in ["pringles", "pringles_light"]:
        root_dir = os.path.join(shared_dir, "inference", "accuracy_test", captured_object)
        
        if not os.path.exists(root_dir):
            continue
            
        for grasp_type in os.listdir(root_dir):
            grasp_type_path = os.path.join(root_dir, grasp_type)
            if not os.path.isdir(grasp_type_path):
                continue
                
            for pose_index in os.listdir(grasp_type_path):
                print(f"Analyzing {captured_object} - {grasp_type} - {pose_index}")
                pose_index_path = os.path.join(grasp_type_path, pose_index)
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
                    results[grasp_type][captured_object][pose_index]['total_attempts'] += 1
                    
                    if success:
                        
                        
                        
                        try:
                            if not os.path.exists(os.path.join(attempt_path, "place_6D.npy")):
                                img_dir = os.path.join(attempt_path, "place")
                                img_dict = {img_name.split(".")[0]:cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)}
                                if len(img_dict) == 0:
                                    continue
                                cur_6D = get_current_object_6d(captured_object, True, img_dict)
                                print("saving 6D")
                                np.save(os.path.join(attempt_path, "place_6D.npy"), cur_6D)
                            else:
                                cur_6D = np.load(os.path.join(attempt_path, "place_6D.npy"))
                        except:
                            continue
                                            
                        target_6d_file = os.path.join(attempt_path, "target_6D.npy")
                        
                        if os.path.exists(target_6d_file):
                            try:
                                place_6D = np.load(target_6d_file)
                                # Z 방향 확인 (성공 기준)
                                if cur_6D[2,2] > 0.7:
                                    # 거리 계산 (mm 단위)
                                    distance_mm = np.array(cur_6D[:2,3]) - np.array(place_6D[:2,3])
                                    
                                    results[grasp_type][captured_object][pose_index]['offset'].append(distance_mm)
                                    results[grasp_type][captured_object][pose_index]['attemp_idx'].append(attempt_index)
                                    results[grasp_type][captured_object][pose_index]['successful_attempts'] += 1
                                else:
                                    print(cur_6D[2,2])
                            except Exception as e:
                                print(f"Error processing {attempt_path}: {e}")

    return results

def draw_confidence_ellipse(ax, points, confidence=0.95, **kwargs):
    """신뢰구간 타원 그리기"""
    
    # 평균과 공분산 계산
    mean_x, mean_y = np.mean(points, axis=0)
    cov = np.cov(points.T)
    
    # 고유값과 고유벡터
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # 신뢰구간에 따른 chi-square 값
    chi2_values = {0.50: 1.386, 0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
    chi2_val = chi2_values.get(confidence, 5.991)
    
    # 타원의 반축 길이
    a = np.sqrt(chi2_val * eigenvals[1])  # 장축
    b = np.sqrt(chi2_val * eigenvals[0])  # 단축
    
    # 회전 각도
    angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
    
    # 타원 생성
    ellipse = Ellipse((mean_x, mean_y), 2*a, 2*b, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    
    return ellipse, (a, b, angle)

def plot_single_pose_simple(offset, offset_index, capture_object, grasp_type, pose_index, save_dir="accuracy_result"):
    """단순하게 offset 점들과 95% 타원을 그리는 함수 (짝홀 분리)"""
    
    if len(offset) == 0:
        print(f"No offset data for {capture_object}/{grasp_type}/{pose_index}")
        return
    
    # 저장 폴더 생성
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    # 짝홀로 데이터 분리
    even_mask = np.array([int(idx) % 2 == 0 for idx in offset_index])
    odd_mask = ~even_mask
    
    even_offset = offset[even_mask]
    odd_offset = offset[odd_mask]
    even_indices = np.array(offset_index)[even_mask]
    odd_indices = np.array(offset_index)[odd_mask]
    
    # 짝수 시도 (right→left) - 파란색
    if len(even_offset) > 0:
        plt.scatter(even_offset[:, 0], even_offset[:, 1], 
                   c='blue', s=100, alpha=0.7, edgecolors='black', 
                   linewidth=1, marker='o', label='Even (Right→Left)')
        
        # 짝수 평균점
        mean_even = np.mean(even_offset, axis=0)
        plt.scatter(mean_even[0], mean_even[1], c='darkblue', marker='D', 
                   s=200, edgecolors='white', linewidth=2, label='Even Mean')
        
        # 짝수 attempt index 라벨
        for x, y, idx in zip(even_offset[:, 0], even_offset[:, 1], even_indices):
            plt.annotate(str(idx), (x, y), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, weight='bold', color='blue')
        
        # 짝수 95% 신뢰구간 타원
        if len(even_offset) > 2:
            draw_confidence_ellipse(plt.gca(), even_offset, confidence=0.95, 
                                   alpha=0.2, color='blue', linestyle='--', linewidth=2)
    
    # 홀수 시도 (left→right) - 빨간색
    if len(odd_offset) > 0:
        plt.scatter(odd_offset[:, 0], odd_offset[:, 1], 
                   c='red', s=100, alpha=0.7, edgecolors='black', 
                   linewidth=1, marker='s', label='Odd (Left→Right)')
        
        # 홀수 평균점
        mean_odd = np.mean(odd_offset, axis=0)
        plt.scatter(mean_odd[0], mean_odd[1], c='darkred', marker='D', 
                   s=200, edgecolors='white', linewidth=2, label='Odd Mean')
        
        # 홀수 attempt index 라벨
        for x, y, idx in zip(odd_offset[:, 0], odd_offset[:, 1], odd_indices):
            plt.annotate(str(idx), (x, y), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, weight='bold', color='red')
        
        # 홀수 95% 신뢰구간 타원
        if len(odd_offset) > 2:
            draw_confidence_ellipse(plt.gca(), odd_offset, confidence=0.95, 
                                   alpha=0.2, color='red', linestyle=':', linewidth=2)
    
    # 전체 평균점
    mean_all = np.mean(offset, axis=0)
    plt.scatter(mean_all[0], mean_all[1], c='green', marker='*', 
               s=300, edgecolors='black', linewidth=2, label='Overall Mean')
    
    # 목표 위치 (원점) 표시
    plt.scatter(0, 0, c='black', marker='x', s=400, linewidth=4, label='Target')
    
    # 축 설정
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('X Offset (mm)')
    plt.ylabel('Y Offset (mm)')
    plt.title(f'{capture_object} - {grasp_type} - Pose {pose_index}\n(Even: Right→Left, Odd: Left→Right)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 컬러바 (전체 거리 기준)
    distances = np.linalg.norm(offset, axis=1)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Distance (mm)')
    
    # 통계 정보 텍스트로 추가
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # 방향별 통계 계산
    stats_text = f'Total Attempts: {len(offset)}\nMean Distance: {mean_dist:.2f} mm\nStd Distance: {std_dist:.2f} mm'
    
    if len(even_offset) > 0:
        even_mean_dist = np.mean(np.linalg.norm(even_offset, axis=1))
        stats_text += f'\nEven (R→L): {len(even_offset)} attempts, {even_mean_dist:.2f} mm'
    
    if len(odd_offset) > 0:
        odd_mean_dist = np.mean(np.linalg.norm(odd_offset, axis=1))
        stats_text += f'\nOdd (L→R): {len(odd_offset)} attempts, {odd_mean_dist:.2f} mm'
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 저장 (show() 전에!)
    os.makedirs(f"accuracy_result/{capture_object}/{grasp_type}", exist_ok=True)
    save_path = f"accuracy_result/{capture_object}/{grasp_type}/{pose_index}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
# 사용 예시:
if __name__ == "__main__":
    # 결과 로드 (analyze_weight_test_results() 함수 실행 필요)
    results = analyze_weight_test_results()  # 이 함수는 별도로 정의되어야 함
    
    for capture_object in results.keys():
        precision_dict = {}
        for grasp_type in results[capture_object].keys():
            precision_dict[grasp_type] = []
            for index in results[capture_object][grasp_type].keys():
                offset = np.array(results[capture_object][grasp_type][index]['offset'])
                mid = np.mean(offset, axis=0)

                precision = np.linalg.norm(np.std(offset, axis=0))
                # print(f"{capture_object} - {grasp_type} - {index} | Mean Offset: {mid}, Precision (STD): {precision}, Attempts: {results[capture_object][grasp_type][index]['total_attempts']}, Successes: {results[capture_object][grasp_type][index]['successful_attempts']}")
                precision_dict[grasp_type].append(precision)
                offset_index = results[capture_object][grasp_type][index]['attemp_idx']
                plot_single_pose_simple(offset, offset_index, capture_object, grasp_type, index)
                # plot offset with dots
            print(f"{capture_object} - {grasp_type} | Average Precision (STD): {np.mean(precision_dict[grasp_type])}, Poses: {len(precision_dict[grasp_type])}")
    print("\n✅ 분석 완료! CSV 파일로 저장되었습니다.")