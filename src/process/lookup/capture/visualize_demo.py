import numpy as np
import os
import time
from scipy.spatial.transform import Rotation

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

hand_name = "allegro"
arm_name = "xarm"
obj_name = "pringles_heavy"

# LINK2WRIST = np.linalg.inv(DEVICE2WRIST["xarm"]) @ DEVICE2WRIST[hand_name]
LINK2WRIST = load_latest_eef()
demo_name = "21"
demo_path = os.path.join(shared_dir, "capture", "lookup", obj_name, demo_name)

sim = IsaacSimulator(headless=False, plane=True)

sim.load_robot_asset(None, hand_name)
sim.load_robot_asset(arm_name, hand_name)

sim.load_object_asset("bottle")

env_list = []
action_dict = {}
object_dict = {}

# hand_qpos = np.load(os.path.join(demo_path, "raw", hand_name, f"qpos.npy"))
wrist_pos = np.load(os.path.join(demo_path, "arm", "action.npy"))
arm_qpos = np.load(os.path.join(demo_path, "arm", "qpos.npy"))
hand_qpos = np.load(os.path.join(demo_path, "hand", "qpos.npy"))

wrist_pos, arm_qpos, hand_qpos = refine_trajectory(wrist_pos, arm_qpos, hand_qpos, dt=1/30)

T = wrist_pos.shape[0]
state = np.concatenate([arm_qpos, hand_qpos], axis=1)

object_T = np.load(os.path.join(demo_path, f"obj_T.npy"), allow_pickle=True).item()[obj_name]
env_name = "fuck"

sim.add_env(env_name,env_info = {"robot":{},
                                "robot_vis":{"arm":(arm_name, hand_name)},
                                "object":{},
                                "object_vis":{}#"bottle":"bottle"}
                                })
mid = 84# put as it is
start = max(0,mid - 30)
end = min(T-1, mid + 30)

while True:
    sim.reset(env_name, {"robot":{},
            "robot_vis":{"arm":state[start]},
            "object":{},
            "object_vis":{}#"bottle":object_T[0].copy()}
            })
    for idx in range(start, end, 1):
        sim.step(env_name, {"robot":{},
                "robot_vis":{"arm":state[idx].copy()},
                "object_vis":{}#"bottle":object_T[idx].copy()}
                })

        sim.tick()
        time.sleep(1/30)
    
    for idx in range(end-1, start-1, -1):
        sim.step(env_name, {"robot":{},
                "robot_vis":{"arm":state[idx].copy()},
                "object_vis":{}#"bottle":object_T[idx].copy()}
                })

        sim.tick()
        time.sleep(1/30)
    
sim.terminate()
