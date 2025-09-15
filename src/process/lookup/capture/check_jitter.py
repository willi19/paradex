import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation
import os
from collections import defaultdict

# 기존 코드에서 가져온 부분
from paradex.utils.file_io import shared_dir, download_dir, eef_calib_path, load_latest_eef
from paradex.simulator import IsaacSimulator
from paradex.robot.mimic_joint import parse_inspire
from paradex.geometry.coordinate import DEVICE2WRIST



def refine_trajectory(wrist_pos, qpos, hand_qpos, tolerance=1e-6, max_acc=40.0, max_vel=0.2, max_ang_vel=2.0, dt=0.01):
    """
    wrist_pos에서 이전 상태와 동일한 중복 프레임들을 제거하고,
    가속도와 속도 제한을 적용하여 궤적을 정제
    
    Args:
        wrist_pos: (N, 4, 4) SE3 변환 행렬
        qpos: (N, DOF) 관절 위치
        hand_qpos: (N, HAND_DOF) 핸드 관절 위치
        tolerance: 동일성 판단을 위한 허용 오차
        max_acc: 최대 허용 가속도 (m/s²)
        max_vel: 최대 허용 속도 (m/s)
        max_ang_vel: 최대 허용 각속도 (rad/s)
        dt: 시간 간격
    
    Returns:
        refined_wrist_pos: 정제된 SE3 변환 행렬
        refined_qpos: 정제된 관절 위치
        refined_hand_qpos: 정제된 핸드 관절 위치
    """
    if len(wrist_pos) <= 1:
        return wrist_pos, qpos, hand_qpos
    
    # Step 1: 중복 프레임 제거
    valid_indices = [0]
    
    for i in range(1, len(wrist_pos)):
        # 현재 프레임과 이전 프레임 비교
        current_frame = wrist_pos[i]
        previous_frame = wrist_pos[valid_indices[-1]]
        
        # SE3 행렬 차이 계산
        diff = np.abs(current_frame - previous_frame)
        max_diff = np.max(diff)
        
        # 허용 오차보다 크면 유지
        if max_diff > tolerance:
            valid_indices.append(i)
    
    # 중복 제거된 데이터
    dedup_wrist_pos = wrist_pos[valid_indices]
    dedup_qpos = qpos[valid_indices]
    dedup_hand_qpos = hand_qpos[valid_indices]
    
    print(f"Removed {len(wrist_pos) - len(dedup_wrist_pos)} duplicate frames")
    
    if len(dedup_wrist_pos) <= 2:
        return dedup_wrist_pos, dedup_qpos, dedup_hand_qpos
    
    # Step 2: 가속도 40 이상인 구간 제거
    positions = dedup_wrist_pos[:, :3, 3]
    velocities = np.diff(positions, axis=0) / dt
    accelerations = np.diff(velocities, axis=0) / dt
    acc_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    # 가속도가 임계값 이하인 구간만 유지
    valid_acc_indices = [0, 1]  # 처음 두 점은 항상 유지 (가속도 계산 불가)
    
    for i in range(len(acc_magnitudes)):
        if acc_magnitudes[i] <= max_acc:
            valid_acc_indices.append(i + 2)  # acceleration index + 2 = position index
    
    # 중복 제거 및 정렬
    valid_acc_indices = sorted(list(set(valid_acc_indices)))
    
    acc_filtered_wrist_pos = dedup_wrist_pos[valid_acc_indices]
    acc_filtered_qpos = dedup_qpos[valid_acc_indices]
    acc_filtered_hand_qpos = dedup_hand_qpos[valid_acc_indices]
    
    print(f"Removed {len(dedup_wrist_pos) - len(acc_filtered_wrist_pos)} high acceleration frames")
    
    if len(acc_filtered_wrist_pos) <= 2:
        return acc_filtered_wrist_pos, acc_filtered_qpos, acc_filtered_hand_qpos
    
    # Step 3: 속도 및 각속도 제한을 위한 linear interpolation
    refined_wrist_pos = [acc_filtered_wrist_pos[0]]
    refined_qpos = [acc_filtered_qpos[0]]
    refined_hand_qpos = [acc_filtered_hand_qpos[0]]
    
    for i in range(1, len(acc_filtered_wrist_pos)):
        current_pos = acc_filtered_wrist_pos[i, :3, 3]
        previous_pos = refined_wrist_pos[-1][:3, 3]
        
        current_rot = acc_filtered_wrist_pos[i, :3, :3]
        previous_rot = refined_wrist_pos[-1][:3, :3]
        
        # 선형 속도 계산
        distance = np.linalg.norm(current_pos - previous_pos)
        linear_velocity = distance / dt
        
        # 각속도 계산
        from scipy.spatial.transform import Rotation as R
        prev_r = R.from_matrix(previous_rot)
        curr_r = R.from_matrix(current_rot)
        
        # 회전 차이를 axis-angle로 변환하여 각속도 계산
        relative_rot = curr_r * prev_r.inv()
        axis_angle = relative_rot.as_rotvec()
        angular_velocity = np.linalg.norm(axis_angle) / dt
        
        # 필요한 세그먼트 수 계산 (선형 속도와 각속도 중 더 큰 것 기준)
        n_segments_linear = int(np.ceil(linear_velocity / max_vel)) if linear_velocity > max_vel else 1
        n_segments_angular = int(np.ceil(angular_velocity / max_ang_vel)) if angular_velocity > max_ang_vel else 1
        n_segments = max(n_segments_linear, n_segments_angular, 1)
        
        if n_segments == 1:
            # 속도가 제한 내면 그대로 추가
            refined_wrist_pos.append(acc_filtered_wrist_pos[i])
            refined_qpos.append(acc_filtered_qpos[i])
            refined_hand_qpos.append(acc_filtered_hand_qpos[i])
        else:
            # 속도가 초과하면 linear interpolation으로 중간 점들 추가
            for j in range(1, n_segments + 1):
                alpha = j / n_segments
                
                # SE3 행렬 interpolation
                interp_wrist_pos = np.eye(4)
                
                # Position interpolation
                interp_pos = previous_pos + alpha * (current_pos - previous_pos)
                interp_wrist_pos[:3, 3] = interp_pos
                
                # Rotation interpolation (SLERP)
                key_rots = R.concatenate([prev_r, curr_r])
                key_times = [0, 1]
                from scipy.spatial.transform import Slerp
                slerp = Slerp(key_times, key_rots)
                interp_r = slerp(alpha)
                interp_wrist_pos[:3, :3] = interp_r.as_matrix()
                
                # Joint position interpolation
                interp_qpos = refined_qpos[-1] + alpha * (acc_filtered_qpos[i] - refined_qpos[-1])
                interp_hand_qpos = refined_hand_qpos[-1] + alpha * (acc_filtered_hand_qpos[i] - refined_hand_qpos[-1])
                
                refined_wrist_pos.append(interp_wrist_pos)
                refined_qpos.append(interp_qpos)
                refined_hand_qpos.append(interp_hand_qpos)
    
    # List를 numpy array로 변환
    refined_wrist_pos = np.array(refined_wrist_pos)
    refined_qpos = np.array(refined_qpos)
    refined_hand_qpos = np.array(refined_hand_qpos)
    
    print(f"Final trajectory: {len(wrist_pos)} → {len(refined_wrist_pos)} frames")
    print(f"Added {len(refined_wrist_pos) - len(acc_filtered_wrist_pos)} interpolated frames for velocity smoothing")
    
    return refined_wrist_pos, refined_qpos, refined_hand_qpos

def se3_to_pose_velocity(se3_matrices, dt=1.0):
    """
    SE3 변환 행렬 시퀀스에서 linear/angular velocity 계산
    
    Args:
        se3_matrices: (N, 4, 4) SE3 변환 행렬들
        dt: 시간 간격
    
    Returns:
        linear_vel: (N-1, 3) 선형 속도
        angular_vel: (N-1, 3) 각속도
    """
    positions = se3_matrices[:, :3, 3]  # translation 부분
    rotations = se3_matrices[:, :3, :3]  # rotation 부분
    
    # Linear velocity
    linear_vel = np.diff(positions, axis=0) / dt
    
    # Angular velocity 계산
    angular_vel = []
    for i in range(len(rotations) - 1):
        R1 = rotations[i]
        R2 = rotations[i + 1]
        
        # Rotation matrix difference
        dR = R2 @ R1.T
        
        # Convert to axis-angle representation for angular velocity
        r = Rotation.from_matrix(dR)
        axis_angle = r.as_rotvec()
        angular_vel.append(axis_angle / dt)
    
    return linear_vel, np.array(angular_vel)

def compute_derivatives(data, dt=1.0):
    """
    데이터의 1차, 2차, 3차 미분 계산 (velocity, acceleration, jerk)
    
    Args:
        data: (N, D) 시계열 데이터
        dt: 시간 간격
    
    Returns:
        velocity, acceleration, jerk
    """
    velocity = np.diff(data, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    
    return velocity, acceleration, jerk

def analyze_motion_data(wrist_pos, dt=1/30):
    """
    손목 위치와 관절 위치 데이터의 운동학적 특성 분석
    
    Args:
        wrist_pos: (N, 4, 4) SE3 변환 행렬
        arm_qpos: (N, DOF) 관절 위치
        dt: 시간 간격
    
    Returns:
        analysis_results: 분석 결과 딕셔너리
    """
    results = {}
    
    # SE3 wrist position 분석
    linear_vel, angular_vel = se3_to_pose_velocity(wrist_pos, dt)
    
    # Linear motion derivatives
    linear_acc = np.diff(linear_vel, axis=0) / dt
    linear_jerk = np.diff(linear_acc, axis=0) / dt
    
    # Angular motion derivatives  
    angular_acc = np.diff(angular_vel, axis=0) / dt
    angular_jerk = np.diff(angular_acc, axis=0) / dt
    
    results['wrist'] = {
        'linear_velocity': linear_vel,
        'linear_acceleration': linear_acc,
        'linear_jerk': linear_jerk,
        'angular_velocity': angular_vel,
        'angular_acceleration': angular_acc,
        'angular_jerk': angular_jerk
    }

    return results

def plot_motion_distributions(all_results, save_path=None):
    """
    모든 데모의 운동학적 특성 분포 시각화
    """
    plt.style.use('default')
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Motion Characteristics Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 색상 팔레트 설정
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # Wrist linear motion
    for i, metric in enumerate(['linear_velocity', 'linear_acceleration', 'linear_jerk']):
        ax = axes[i, 0]
        combined_data = []
        for results in all_results:
            data = results['wrist'][metric]
            # L2 norm for each timestep
            norms = np.linalg.norm(data, axis=1)
            combined_data.extend(norms)
        
        ax.hist(combined_data, bins=50, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.5)
        ax.set_title(f'Wrist {metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_val = np.mean(combined_data)
        std_val = np.std(combined_data)
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    # Wrist angular motion
    for i, metric in enumerate(['angular_velocity', 'angular_acceleration', 'angular_jerk']):
        ax = axes[i, 1]
        combined_data = []
        for results in all_results:
            data = results['wrist'][metric]
            # L2 norm for each timestep
            norms = np.linalg.norm(data, axis=1)
            combined_data.extend(norms)
        
        ax.hist(combined_data, bins=50, alpha=0.7, color=colors[i+3], edgecolor='black', linewidth=0.5)
        ax.set_title(f'Wrist {metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Magnitude (rad or rad/s)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_val = np.mean(combined_data)
        std_val = np.std(combined_data)
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(all_results):
    """
    운동학적 특성의 통계 정보 출력
    """
    print("="*80)
    print("MOTION STATISTICS SUMMARY")
    print("="*80)
    
    # Wrist statistics
    print("\n🤖 WRIST MOTION STATISTICS")
    print("-"*50)
    
    wrist_metrics = ['linear_velocity', 'linear_acceleration', 'linear_jerk', 
                    'angular_velocity', 'angular_acceleration', 'angular_jerk']
    
    for metric in wrist_metrics:
        combined_data = []
        for results in all_results:
            data = results['wrist'][metric]
            norms = np.linalg.norm(data, axis=1)
            combined_data.extend(norms)
            
        
        
        combined_data = np.array(combined_data)
        print(f"{metric.replace('_', ' ').title():25s} | "
              f"Mean: {np.mean(combined_data):8.4f} | "
              f"Std: {np.std(combined_data):8.4f} | "
              f"Max: {np.max(combined_data):8.4f}")

# 메인 실행 코드
def main():
    hand_name = "allegro"
    arm_name = "xarm"
    
    obj_list = os.listdir(os.path.join(shared_dir, "capture", "lookup"))
    LINK2WRIST = load_latest_eef()
    
    all_results = []
    
    print("Loading and analyzing data...")
    for obj_name in obj_list:
        demo_list = os.listdir(os.path.join(shared_dir, "capture", "lookup", obj_name))
        for demo_name in demo_list:
            demo_path = os.path.join(shared_dir, "capture", "lookup", obj_name, demo_name)
            
            try:
                wrist_pos = np.load(os.path.join(demo_path, "arm", "action.npy"))
                arm_qpos = np.load(os.path.join(demo_path, "arm", "qpos.npy"))
                hand_qpos = np.load(os.path.join(demo_path, "hand", "qpos.npy"))
                
                wrist_pos, arm_qpos, hand_qpos = refine_trajectory(wrist_pos, arm_qpos, hand_qpos, dt=1/30)
                # 데이터 검증
                if wrist_pos.shape[-2:] == (4, 4):
                    results = analyze_motion_data(wrist_pos, dt=1/30)
                    angvel = np.linalg.norm(results['wrist']['angular_velocity'],axis=1)
                    if np.sum(angvel > 2.0) > 0:  # 예외 처리
                        print(obj_name, demo_name, " - Skipped due to high velocity")
                        print(np.where(angvel > 2.0))
                        print(angvel[np.where(angvel > 2.0)])
                        # print(np.linalg.norm(results['wrist']['linear_acceleration'][np.where(vel > 0.5)], axis=1))
                        # print(np.linalg.norm(results['wrist']['linear_jerk'][np.where(vel > 0.5)], axis=1))
                        continue
                    # if obj_name == "book" and demo_name == "1":
                    #     print(obj_name, demo_name)
                    #     import pdb; pdb.set_trace()
                    all_results.append(results)
                    # print(f"Processed: {obj_name}/{demo_name}")
                else:
                    print(f"Skipped invalid data: {obj_name}/{demo_name}")
                    
            except Exception as e:
                print(f"Error processing {obj_name}/{demo_name}: {e}")
    
    print(f"\nSuccessfully processed {len(all_results)} demonstrations")
    
    # 통계 출력 및 시각화
    print_statistics(all_results)
    plot_motion_distributions(all_results, save_path="motion_analysis.png")

if __name__ == "__main__":
    main()