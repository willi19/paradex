import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

from paradex.image.aruco import detect_aruco, triangulate_marker, draw_aruco
from paradex.image.undistort import undistort_img
from paradex.geometry.Tsai_Lenz import solve
from paradex.geometry.conversion import project, to_homo
from paradex.utils.file_io import shared_dir, load_camparam
from paradex.image.projection import get_cammtx
from paradex.geometry.math import rigid_transform_3D

marker_id = [261, 262, 263, 264, 265, 266]
root_path = os.path.join(shared_dir, "debug_", "inference")

def calculate_correlation_offset(cam_indices, cam_y_values, robot_indices, robot_y_values, max_offset=50):
    """두 시퀀스 간의 최적 오프셋을 상관관계로 찾기"""
    
    # 공통 인덱스 범위 찾기
    min_cam_idx = min(cam_indices)
    max_cam_idx = max(cam_indices)
    min_robot_idx = min(robot_indices)
    max_robot_idx = max(robot_indices)
    
    # 겹치는 범위 계산
    common_start = max(min_cam_idx, min_robot_idx)
    common_end = min(max_cam_idx, max_robot_idx)
    
    if common_start >= common_end:
        print("Warning: No overlapping range between cam and robot data")
        return 0, 0.0
    
    best_offset = 0
    best_correlation = -1
    
    # 오프셋을 변화시키며 상관관계 계산
    for offset in range(-max_offset, max_offset + 1):
        cam_values = []
        robot_values = []
        
        # 오프셋을 적용하여 매칭되는 인덱스 쌍 찾기
        for cam_idx in cam_indices:
            robot_idx = cam_idx + offset
            
            if robot_idx in robot_indices:
                cam_values.append(cam_y_values[cam_indices.index(cam_idx)])
                robot_values.append(robot_y_values[robot_indices.index(robot_idx)])
        
        # 최소 10개 이상의 매칭점이 있어야 신뢰할 만함
        if len(cam_values) >= 10:
            correlation = np.corrcoef(cam_values, robot_values)[0, 1]
            
            if not np.isnan(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_offset = offset
    
    return best_offset, best_correlation

def interpolate_and_align(cam_indices, cam_y_values, robot_indices, robot_y_values, offset):
    """오프셋을 적용하여 데이터를 정렬하고 보간"""
    
    # 로봇 인덱스에 오프셋 적용
    adjusted_robot_indices = [idx - offset for idx in robot_indices]
    
    # 공통 범위 찾기
    all_indices = sorted(set(cam_indices + adjusted_robot_indices))
    min_idx = max(min(cam_indices), min(adjusted_robot_indices))
    max_idx = min(max(cam_indices), max(adjusted_robot_indices))
    
    common_indices = [idx for idx in all_indices if min_idx <= idx <= max_idx]
    
    # 보간 함수 생성
    if len(cam_indices) > 1:
        cam_interp = interp1d(cam_indices, cam_y_values, kind='linear', 
                             bounds_error=False, fill_value='extrapolate')
    else:
        cam_interp = lambda x: cam_y_values[0] if len(cam_y_values) > 0 else 0
    
    if len(adjusted_robot_indices) > 1:
        robot_interp = interp1d(adjusted_robot_indices, robot_y_values, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
    else:
        robot_interp = lambda x: robot_y_values[0] if len(robot_y_values) > 0 else 0
    
    # 공통 인덱스에서 값 계산
    aligned_cam_y = cam_interp(common_indices)
    aligned_robot_y = robot_interp(common_indices)
    
    return common_indices, aligned_cam_y, aligned_robot_y

def plot_cam_robot_comparison(cam_indices, cam_y_values, robot_indices, robot_y_values, 
                            offset, correlation, index, save_path):
    """카메라와 로봇 Y 위치 비교 플롯"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Camera vs Robot Y Position Analysis - Index {index}', fontsize=16)
    
    # 1. 원본 데이터 오버레이
    axes[0, 0].plot(cam_indices, cam_y_values, 'b.-', label='Camera Y', alpha=0.7, markersize=3)
    axes[0, 0].plot(robot_indices, robot_y_values, 'r.-', label='Robot Y', alpha=0.7, markersize=3)
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('Y Position (m)')
    axes[0, 0].set_title('Original Data (Raw)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 오프셋 적용 후 정렬된 데이터
    try:
        common_indices, aligned_cam_y, aligned_robot_y = interpolate_and_align(
            cam_indices, cam_y_values, robot_indices, robot_y_values, offset)
        
        axes[0, 1].plot(common_indices, aligned_cam_y, 'b.-', label='Camera Y (aligned)', alpha=0.7, markersize=3)
        axes[0, 1].plot(common_indices, aligned_robot_y, 'r.-', label='Robot Y (aligned)', alpha=0.7, markersize=3)
        axes[0, 1].set_xlabel('Frame Index')
        axes[0, 1].set_ylabel('Y Position (m)')
        axes[0, 1].set_title(f'Aligned Data (Offset: {offset:+d}, Corr: {correlation:.3f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 차이 분석
        if len(aligned_cam_y) > 0 and len(aligned_robot_y) > 0:
            y_diff = aligned_cam_y - aligned_robot_y
            axes[1, 0].plot(common_indices, y_diff, 'g.-', alpha=0.7, markersize=3)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Frame Index')
            axes[1, 0].set_ylabel('Y Difference (m)')
            axes[1, 0].set_title(f'Camera - Robot Y Difference')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 통계 정보 표시
            mean_diff = np.mean(y_diff)
            std_diff = np.std(y_diff)
            axes[1, 0].text(0.05, 0.95, f'Mean: {mean_diff:.4f}m\nStd: {std_diff:.4f}m', 
                           transform=axes[1, 0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 4. 상관관계 산점도
        if len(aligned_cam_y) > 0 and len(aligned_robot_y) > 0:
            axes[1, 1].scatter(aligned_robot_y, aligned_cam_y, alpha=0.6, s=20)
            
            # 완벽한 상관관계 라인
            min_val = min(np.min(aligned_robot_y), np.min(aligned_cam_y))
            max_val = max(np.max(aligned_robot_y), np.max(aligned_cam_y))
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect correlation')
            
            axes[1, 1].set_xlabel('Robot Y Position (m)')
            axes[1, 1].set_ylabel('Camera Y Position (m)')
            axes[1, 1].set_title(f'Correlation Scatter Plot (r={correlation:.3f})')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axis('equal')
    
    except Exception as e:
        print(f"Error in plotting aligned data: {e}")
        # 빈 플롯들에 에러 메시지
        for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # 플롯 저장
    plot_save_path = os.path.join(save_path, f"cam_robot_y_analysis_{index}.png")
    plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_save_path}")
    
    plt.show()
    
    return correlation

def analyze_single_index(index_path, index):
    """단일 인덱스에 대한 분석"""
    print(f"\n=== Analyzing Index {index} ===")
    
    try:
        # 데이터 로딩
        serial_list = [vid_name.split('.')[0] for vid_name in os.listdir(os.path.join(index_path, "videos"))]
        
        intrinsic, extrinsic = load_camparam(index_path)
        cammtx = get_cammtx(intrinsic, extrinsic)
        
        cor_3d = np.load(os.path.join(index_path, "cor_3d.npy"), allow_pickle=True).item()
        cor_2d = {serial: np.load(os.path.join(index_path, "marker2D", f"{serial}.npy"), allow_pickle=True).item() 
                 for serial in serial_list}
        
        c2r = np.load(os.path.join(index_path, "C2R.npy"))
        marker_offset = np.load(os.path.join(index_path, "marker_pos.npy"), allow_pickle=True).item()
        robot_pose = np.load(os.path.join(index_path, "xarm", "position.npy"))
        
        print(f"Robot poses: {len(robot_pose)}, Camera observations: {len(cor_3d)}")
        
        # 로봇 Y 위치 계산
        robot_y = {}
        for i, pose in enumerate(robot_pose):
            robot_y[i+1] = pose[1, 3]
        
        # 카메라 Y 위치 계산
        cam_y = {}
        for i, data in cor_3d.items():
            A = []
            B = []
            for id, cor in data.items():
                if id in marker_offset:
                    A.append(marker_offset[id][:, :3])
                    B.append(cor)
            
            if len(A) > 0:
                A = np.concatenate(A, axis=0)
                B = np.concatenate(B, axis=0)
                
                T = rigid_transform_3D(A, B)
                T = np.linalg.inv(c2r) @ T
                
                cam_y[i] = T[1, 3]
        
        # 데이터 정리
        cam_indices = sorted(cam_y.keys())
        cam_y_values = [cam_y[idx] for idx in cam_indices]
        robot_indices = sorted(robot_y.keys())
        robot_y_values = [robot_y[idx] for idx in robot_indices]
        
        print(f"Camera Y data points: {len(cam_indices)}")
        print(f"Robot Y data points: {len(robot_indices)}")
        print(f"Camera Y range: {np.min(cam_y_values):.4f} ~ {np.max(cam_y_values):.4f}")
        print(f"Robot Y range: {np.min(robot_y_values):.4f} ~ {np.max(robot_y_values):.4f}")
        
        # 오프셋 계산
        offset, correlation = calculate_correlation_offset(
            cam_indices, cam_y_values, robot_indices, robot_y_values)
        
        print(f"Best offset: {offset:+d} frames")
        print(f"Best correlation: {correlation:.4f}")
        
        # 플롯 생성
        final_correlation = plot_cam_robot_comparison(
            cam_indices, cam_y_values, robot_indices, robot_y_values,
            offset, correlation, index, index_path)
        
        return {
            'index': index,
            'offset': offset,
            'correlation': correlation,
            'cam_data_points': len(cam_indices),
            'robot_data_points': len(robot_indices),
            'cam_y_range': (np.min(cam_y_values), np.max(cam_y_values)),
            'robot_y_range': (np.min(robot_y_values), np.max(robot_y_values))
        }
        
    except Exception as e:
        print(f"Error analyzing index {index}: {e}")
        return None

def main():
    """메인 분석 함수"""
    results = []
    
    for index in sorted(os.listdir(root_path)):
        index_path = os.path.join(root_path, index)
        
        # 필요한 파일들이 있는지 확인
        required_files = ["cor_3d.npy", "C2R.npy", "marker_pos.npy"]
        required_dirs = ["videos", "marker2D", "xarm"]
        
        if not all(os.path.exists(os.path.join(index_path, f)) for f in required_files):
            print(f"Skipping {index}: Missing required files")
            continue
        
        if not all(os.path.exists(os.path.join(index_path, d)) for d in required_dirs):
            print(f"Skipping {index}: Missing required directories")
            continue
        
        result = analyze_single_index(index_path, index)
        if result:
            results.append(result)
    
    # 전체 결과 요약
    print(f"\n{'='*60}")
    print(f"SUMMARY OF ALL ANALYSES")
    print(f"{'='*60}")
    print(f"{'Index':<8} {'Offset':<8} {'Correlation':<12} {'Cam Points':<12} {'Robot Points':<12}")
    print(f"{'-'*60}")
    
    for result in results:
        print(f"{result['index']:<8} {result['offset']:+d}      {result['correlation']:<12.4f} "
              f"{result['cam_data_points']:<12} {result['robot_data_points']:<12}")
    
    if results:
        offsets = [r['offset'] for r in results]
        correlations = [r['correlation'] for r in results]
        
        print(f"\nOverall Statistics:")
        print(f"  Average offset: {np.mean(offsets):+.1f} frames")
        print(f"  Offset std dev: {np.std(offsets):.1f} frames")
        print(f"  Average correlation: {np.mean(correlations):.4f}")
        print(f"  Min correlation: {np.min(correlations):.4f}")
        print(f"  Max correlation: {np.max(correlations):.4f}")

if __name__ == "__main__":
    main()