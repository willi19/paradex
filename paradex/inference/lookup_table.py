import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R, Slerp
import os
import random
import json

from paradex.utils.file_io import shared_dir

lookup_table_path = os.path.join(shared_dir, "capture", "lookup")

class LookupTraj():
    def __init__(self, root_path):
        self.root_path = root_path
        self.info = json.load(open(os.path.join(self.root_path, "meta.json"), 'r'))
        
        self.traj = {"eef_se3": {}, "hand_qpos":{}, "obj_T":{}} 
        for state in ["pick", "place"]:
            self.traj["eef_se3"][state] = np.load(os.path.join(self.root_path, f"refined_{state}_action.npy"))
            self.traj["obj_T"][state] = np.load(os.path.join(self.root_path, f"{state}_objT.npy"))
            self.traj["hand_qpos"][state] = np.load(os.path.join(self.root_path, f"refined_{state}_hand.npy"))
    
    def get_traj(self, pick_6D, place_6D):
        ret = {}
        for state in ["pick", "place"]:
            ret[state] = {}
            ret[state]["eef_se3"] = pick_6D @ self.traj["eef_se3"][state] if state == "pick" else place_6D @ self.traj["eef_se3"][state]
            ret[state]["obj_T"] = pick_6D @ self.traj["obj_T"][state] if state == "pick" else place_6D @ self.traj["obj_T"][state]
            ret[state]["hand_qpos"] = self.traj["hand_qpos"][state]
        return ret

class LookupTable():
    def __init__(self, obj_name, hand, grasp_type_list=None, index_list=None):
        self.root_path = os.path.join(lookup_table_path, obj_name)
        
        self.lookup_traj = {}
        index_list = os.listdir(self.root_path) if index_list is None else index_list
        
        for index in index_list:
            demo_path = os.path.join(self.root_path, index)
            element = LookupTraj(demo_path)
            if element.info["hand_name"] == hand and (grasp_type_list is None or element.info["grasp_type"] in grasp_type_list):
                self.lookup_traj[index] = element
        self.index_list = list(self.lookup_traj.keys())
        
    def get_trajs(self, pick_6D, place_6D):
        ret = []
        for index in self.index_list:
            traj = self.lookup_traj[index].get_traj(pick_6D, place_6D)
            traj["index"] = index
            ret.append(traj)
        return ret


def get_traj(obj, hand, start6D, pick6D, place6D, index):
    index_path = os.path.join(lookup_table_path, obj, index)
    
    pick_traj = np.load(f"{index_path}/refined_pick_action.npy")
    place_traj = np.load(f"{index_path}/refined_place_action.npy")
    
    pick_hand_traj = np.load(f"{index_path}/refined_pick_hand.npy")
    place_hand_traj = np.load(f"{index_path}/refined_place_hand.npy")
    start_hand = np.zeros((pick_hand_traj.shape[1]))
    pick_traj = pick6D @ pick_traj
    place_traj = place6D @ place_traj 
    
    approach_traj, approach_hand_traj = get_linear_path(start6D, pick_traj[0], start_hand, pick_hand_traj[0])
    return_traj, return_hand_traj = get_linear_path(place_traj[-1], start6D, place_hand_traj[-1], start_hand)
    move_traj, move_hand = get_linear_path(pick_traj[-1], place_traj[0], pick_hand_traj[-1], place_hand_traj[0])
    
    
    traj = np.concatenate([approach_traj, pick_traj, move_traj, place_traj, return_traj])
    hand_traj = np.concatenate([approach_hand_traj, pick_hand_traj, move_hand, place_hand_traj, return_hand_traj])
    
    state = np.zeros((len(traj)))
    state[len(approach_traj):len(approach_traj)+len(pick_traj)] = 1
    state[len(approach_traj)+len(pick_traj):len(approach_traj)+len(pick_traj)+len(move_traj)] = 2
    state[len(approach_traj)+len(pick_traj)+len(move_traj):len(approach_traj)+len(pick_traj)+len(move_traj)+len(place_traj)] = 3
    state[len(approach_traj)+len(pick_traj)+len(move_traj)+len(place_traj):] = 4

    
    return index, traj, hand_traj, state

def refine_trajectory(wrist_pos, qpos, hand_qpos, tolerance=1e-6, max_acc=40.0, max_vel=0.15, max_ang_vel=2.0, dt=1 / 30):
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
    
    
    # print(f"Removed {len(wrist_pos) - len(dedup_wrist_pos)} duplicate frames")
    
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
    
    # print(f"Removed {len(dedup_wrist_pos) - len(acc_filtered_wrist_pos)} high acceleration frames")
    
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
    
    # print(f"Final trajectory: {len(wrist_pos)} → {len(refined_wrist_pos)} frames")
    # print(f"Added {len(refined_wrist_pos) - len(acc_filtered_wrist_pos)} interpolated frames for velocity smoothing")
    
    return refined_wrist_pos, refined_qpos, refined_hand_qpos

def get_linear_path(start_6D, end_6D, start_hand, end_hand, max_vel=0.2, max_ang_vel=2.0, dt=1/30):
    """
    시작점에서 끝점까지의 선형 경로를 속도 제한에 따라 생성
    
    Args:
        start_6D: (4, 4) 시작 SE3 변환 행렬
        end_6D: (4, 4) 끝 SE3 변환 행렬
        start_hand: 시작 핸드 관절 위치
        end_hand: 끝 핸드 관절 위치
        max_vel: 최대 선형 속도 (m/s)
        max_ang_vel: 최대 각속도 (rad/s)
        dt: 시간 간격 (s)
    
    Returns:
        move_traj: (N, 4, 4) 생성된 궤적
        move_hand: (N, DOF) 핸드 관절 궤적
    """
    from scipy.spatial.transform import Rotation as R, Slerp
    
    # 시작점과 끝점 추출
    start_pos = start_6D[:3, 3]
    end_pos = end_6D[:3, 3]
    
    start_rot = R.from_matrix(start_6D[:3, :3])
    end_rot = R.from_matrix(end_6D[:3, :3])
    
    # 필요한 거리와 회전각 계산
    linear_distance = np.linalg.norm(end_pos - start_pos)
    
    # 회전 차이 계산
    relative_rot = end_rot * start_rot.inv()
    angular_distance = np.linalg.norm(relative_rot.as_rotvec())
    
    # 각각에 필요한 시간 계산
    time_for_linear = linear_distance / max_vel if linear_distance > 0 else 0
    time_for_angular = angular_distance / max_ang_vel if angular_distance > 0 else 0
    
    # 더 오래 걸리는 쪽에 맞춰 총 시간 결정
    total_time = max(time_for_linear, time_for_angular, dt)  # 최소 dt는 보장
    
    # 필요한 스텝 수 계산
    length = int(np.ceil(total_time / dt))
    
    # 궤적 배열 초기화
    move_traj = np.zeros((length, 4, 4))
    move_hand = np.zeros((length, start_hand.shape[0]))
    
    # SLERP 설정
    key_times = [0, 1]
    key_rots = R.concatenate([start_rot, end_rot])
    slerp = Slerp(key_times, key_rots)
    
    # 궤적 생성
    for i in range(length):
        alpha = (i + 1) / length
        
        # 위치 보간
        pos = (1 - alpha) * start_pos + alpha * end_pos
        
        # 회전 보간
        rot = slerp(alpha).as_matrix()
        
        # SE3 행렬 구성
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        
        move_traj[i] = T
        
        # 핸드 관절 보간
        move_hand[i] = start_hand * (1 - alpha) + end_hand * alpha
     
    return move_traj, move_hand