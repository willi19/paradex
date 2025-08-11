import json
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import shutil

from paradex.io.contact.process import process_contact
from paradex.robot import RobotWrapper
from paradex.utils.file_io import rsc_path, shared_dir
from paradex.video.raw_video import fill_framedrop, get_synced_data
from paradex.geometry.coordinate import DEVICE2WRIST
def plot_robot_data(pc_time, fid, arm_action_sync, arm_qpos_sync, arm_position_orig, arm_pc_time, index, save_dir):
    """로봇 데이터 플롯 생성"""
    
    # 전체 플롯 (2x2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle(f'Robot Data Analysis - Index {index} (Full View)', fontsize=16)
    
    # 1. PC Time vs Frame ID
    axes1[0, 0].plot(pc_time, fid, 'b.-', markersize=2, linewidth=0.5)
    axes1[0, 0].set_xlabel('PC Time (s)')
    axes1[0, 0].set_ylabel('Frame ID')
    axes1[0, 0].set_title('PC Time vs Frame ID')
    axes1[0, 0].grid(True, alpha=0.3)
    
    # 2. Frame ID difference (frame drops 확인)
    fid_diff = np.diff(fid)
    axes1[0, 1].plot(pc_time[1:], fid_diff, 'r.-', markersize=2, linewidth=0.5)
    axes1[0, 1].set_xlabel('PC Time (s)')
    axes1[0, 1].set_ylabel('Frame ID Difference')
    axes1[0, 1].set_title('Frame Drops Detection')
    axes1[0, 1].axhline(y=1, color='g', linestyle='--', alpha=0.7, label='Normal (diff=1)')
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)
    
    # 3. End-effector Y coordinate: Action vs State (synced + raw)
    if arm_action_sync is not None and arm_position_orig is not None:
        # Action Y coordinate (synced to camera time)
        action_y = arm_action_sync[:, 1, 3]  # Y coordinate from action
        
        # Position을 카메라 시간에 동기화
        position_sync = get_synced_data(pc_time, arm_position_orig, arm_pc_time)
        
        if position_sync is not None:
            position_y = position_sync[:, 1, 3]  # Y coordinate from actual position
            
            # Synced data (lines)
            axes1[1, 0].plot(pc_time, action_y, 'r-', linewidth=2, label='Action Y (synced)', alpha=0.8)
            axes1[1, 0].plot(pc_time, position_y, 'b-', linewidth=2, label='State Y (synced)', alpha=0.8)
            
            # Raw data (scatter)
            position_y_raw = arm_position_orig[:, 1, 3]
            axes1[1, 0].scatter(arm_pc_time, position_y_raw, c='cyan', s=8, alpha=0.6, label='State Y (raw)', zorder=5)
            
            axes1[1, 0].set_xlabel('PC Time (s)')
            axes1[1, 0].set_ylabel('Y Position (m)')
            axes1[1, 0].set_title('End-Effector Y Coordinate: Action vs State')
            axes1[1, 0].legend()
            axes1[1, 0].grid(True, alpha=0.3)
            
            # 차이 계산
            y_diff = action_y - position_y
            print(f"Action-State Y difference: mean={np.mean(y_diff):.4f}, std={np.std(y_diff):.4f}")
    
    # 4. End-effector Z coordinate: Action vs State (synced + raw)
    if arm_action_sync is not None and arm_position_orig is not None:
        # Action Z coordinate (synced to camera time)
        action_z = arm_action_sync[:, 2, 3]  # Z coordinate from action
        
        # Position을 카메라 시간에 동기화
        position_sync = get_synced_data(pc_time, arm_position_orig, arm_pc_time)
        
        if position_sync is not None:
            position_z = position_sync[:, 2, 3]  # Z coordinate from actual position
            
            # Synced data (lines)
            axes1[1, 1].plot(pc_time, action_z, 'r-', linewidth=2, label='Action Z (synced)', alpha=0.8)
            axes1[1, 1].plot(pc_time, position_z, 'b-', linewidth=2, label='State Z (synced)', alpha=0.8)
            
            # Raw data (scatter)
            position_z_raw = arm_position_orig[:, 2, 3]
            axes1[1, 1].scatter(arm_pc_time, position_z_raw, c='orange', s=8, alpha=0.6, label='State Z (raw)', zorder=5)
            
            axes1[1, 1].set_xlabel('PC Time (s)')
            axes1[1, 1].set_ylabel('Z Position (m)')
            axes1[1, 1].set_title('End-Effector Z Coordinate: Action vs State')
            axes1[1, 1].legend()
            axes1[1, 1].grid(True, alpha=0.3)
            
            # 차이 계산
            z_diff = action_z - position_z
            print(f"Action-State Z difference: mean={np.mean(z_diff):.4f}, std={np.std(z_diff):.4f}")
    
    plt.tight_layout()
    
    # 전체 플롯 저장
    plot_save_path1 = os.path.join(save_dir, f"robot_analysis_full_{index}.png")
    plt.savefig(plot_save_path1, dpi=150, bbox_inches='tight')
    print(f"Full plot saved to: {plot_save_path1}")
    
    
    # 확대 플롯 (중간 부분) - Action 데이터 추가
    if arm_action_sync is not None and arm_position_orig is not None and position_sync is not None:
        # 중간 40% 구간 선택 (30%~70%)
        total_len = len(pc_time)
        start_idx = int(total_len * 0.3)
        end_idx = int(total_len * 0.7)
        
        # 확대할 시간 범위
        zoom_pc_time = pc_time[start_idx:end_idx]
        zoom_action_y = action_y[start_idx:end_idx]
        zoom_action_z = action_z[start_idx:end_idx]
        zoom_position_y = position_y[start_idx:end_idx]
        zoom_position_z = position_z[start_idx:end_idx]
        
        # Raw 데이터에서 해당 시간 범위 필터링
        time_mask = (arm_pc_time >= zoom_pc_time[0]) & (arm_pc_time <= zoom_pc_time[-1])
        zoom_arm_pc_time = arm_pc_time[time_mask]
        zoom_position_y_raw = position_y_raw[time_mask]
        zoom_position_z_raw = position_z_raw[time_mask]
        
        # Action raw 데이터도 추가 (원본 action 데이터)
        if hasattr(arm_action_orig, '__len__') and len(arm_action_orig) > 0:
            action_y_raw = arm_action_orig[:, 1, 3]
            action_z_raw = arm_action_orig[:, 2, 3]
            zoom_action_y_raw = action_y_raw[time_mask]
            zoom_action_z_raw = action_z_raw[time_mask]
        else:
            zoom_action_y_raw = None
            zoom_action_z_raw = None
        
        # 확대 플롯 생성 (2x1)
        fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10))
        fig2.suptitle(f'Robot Data Analysis - Index {index} (Zoomed: {zoom_pc_time[0]:.2f}s ~ {zoom_pc_time[-1]:.2f}s)', fontsize=16)
        
        # Y coordinate 확대 - Action과 State 모두 표시
        axes2[0].plot(zoom_pc_time, zoom_action_y, 'r-', linewidth=3, label='Action Y (synced)', alpha=0.9)
        axes2[0].plot(zoom_pc_time, zoom_position_y, 'b-', linewidth=3, label='State Y (synced)', alpha=0.9)
        axes2[0].scatter(zoom_arm_pc_time, zoom_position_y_raw, c='cyan', s=15, alpha=0.8, label='State Y (raw)', zorder=5)
        
        # Action raw 데이터도 추가 (있다면)
        if zoom_action_y_raw is not None:
            axes2[0].scatter(zoom_arm_pc_time, zoom_action_y_raw, c='orange', s=15, alpha=0.8, 
                           label='Action Y (raw)', zorder=5, marker='^')
        
        axes2[0].set_xlabel('PC Time (s)')
        axes2[0].set_ylabel('Y Position (m)')
        axes2[0].set_title('End-Effector Y Coordinate (Zoomed) - Action & State')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        
        # Z coordinate 확대 - Action과 State 모두 표시
        axes2[1].plot(zoom_pc_time, zoom_action_z, 'r-', linewidth=3, label='Action Z (synced)', alpha=0.9)
        axes2[1].plot(zoom_pc_time, zoom_position_z, 'b-', linewidth=3, label='State Z (synced)', alpha=0.9)
        axes2[1].scatter(zoom_arm_pc_time, zoom_position_z_raw, c='cyan', s=15, alpha=0.8, label='State Z (raw)', zorder=5)
        
        # Action raw 데이터도 추가 (있다면)
        if zoom_action_z_raw is not None:
            axes2[1].scatter(zoom_arm_pc_time, zoom_action_z_raw, c='orange', s=15, alpha=0.8, 
                           label='Action Z (raw)', zorder=5, marker='^')
        
        axes2[1].set_xlabel('PC Time (s)')
        axes2[1].set_ylabel('Z Position (m)')
        axes2[1].set_title('End-Effector Z Coordinate (Zoomed) - Action & State')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 확대 플롯 저장
        plot_save_path2 = os.path.join(save_dir, f"robot_analysis_zoom_{index}.png")
        plt.savefig(plot_save_path2, dpi=150, bbox_inches='tight')
        print(f"Zoomed plot saved to: {plot_save_path2}")
        
        # 확대 구간 통계
        print(f"\n=== Zoomed Section Statistics ===")
        print(f"Zoom range: {zoom_pc_time[0]:.3f}s ~ {zoom_pc_time[-1]:.3f}s ({len(zoom_pc_time)} samples)")
        
        zoom_y_diff = zoom_action_y - zoom_position_y
        zoom_z_diff = zoom_action_z - zoom_position_z
        print(f"Zoomed Y difference: mean={np.mean(zoom_y_diff):.4f}, std={np.std(zoom_y_diff):.4f}")
        print(f"Zoomed Z difference: mean={np.mean(zoom_z_diff):.4f}, std={np.std(zoom_z_diff):.4f}")

    # plt.show()
def print_data_statistics(pc_time, fid, arm_action_sync, arm_qpos_sync, arm_position_orig, arm_pc_time, index):
    """데이터 통계 출력"""
    print(f"\n=== Data Statistics for Index {index} ===")
    print(f"Camera data:")
    print(f"  PC Time range: {pc_time[0]:.3f} ~ {pc_time[-1]:.3f} seconds")
    print(f"  Total frames: {len(pc_time)}")
    print(f"  Frame ID range: {fid[0]} ~ {fid[-1]}")
    print(f"  Average frame interval: {np.mean(np.diff(pc_time)):.4f} seconds")
    print(f"  Frame drops: {np.sum(np.diff(fid) > 1)} times")
    
    if arm_pc_time is not None:
        print(f"\nArm data:")
        print(f"  Arm PC Time range: {arm_pc_time[0]:.3f} ~ {arm_pc_time[-1]:.3f} seconds")
        print(f"  Total arm samples: {len(arm_pc_time)}")
        print(f"  Average arm interval: {np.mean(np.diff(arm_pc_time)):.4f} seconds")
    
    if arm_action_sync is not None:
        print(f"\nEnd-effector Action data:")
        print(f"  Shape: {arm_action_sync.shape}")
        action_y = arm_action_sync[:, 1, 3]
        action_z = arm_action_sync[:, 2, 3]
        print(f"  Action Y range: {np.min(action_y):.4f} ~ {np.max(action_y):.4f}")
        print(f"  Action Z range: {np.min(action_z):.4f} ~ {np.max(action_z):.4f}")
    
    if arm_position_orig is not None:
        print(f"\nEnd-effector State data:")
        print(f"  Shape: {arm_position_orig.shape}")
        position_y = arm_position_orig[:, 1, 3]
        position_z = arm_position_orig[:, 2, 3]
        print(f"  State Y range: {np.min(position_y):.4f} ~ {np.max(position_y):.4f}")
        print(f"  State Z range: {np.min(position_z):.4f} ~ {np.max(position_z):.4f}")

if __name__ == '__main__':
    save_path = os.path.join(shared_dir, "debug_", "inference")
    
    for index in os.listdir(save_path):
        index_dir = os.path.join(os.path.join(save_path, str(index)))
        raw_dir = os.path.join(os.path.join(index_dir, "raw"))
        
        if not os.path.exists(raw_dir):
            continue
        
        valid = True
        
        arm_name = None
        for tmp_name in ['xarm', 'franka']:
            if tmp_name in os.listdir(raw_dir):
                arm_name = tmp_name
                break
        
        if arm_name == None:
            valid = False
        
        for data_name in ["timestamp"]:#, "C2R.npy"]:
            if data_name not in os.listdir(raw_dir):
                valid = False
        
        if "cam_param" not in os.listdir(index_dir):
            valid = False
        
        if not valid:
            print(f"{index} invalid")
            continue
        
        timestamp = json.load(open(os.path.join(raw_dir, "timestamp", "camera_timestamp.json")))
        pc_time_orig = np.array(timestamp["pc_time"])
        fid_orig = np.array(timestamp["frameID"])
        
        pc_time, fid = fill_framedrop(timestamp)
        os.makedirs(os.path.join(index_dir), exist_ok=True)
        
        robot = RobotWrapper(
            os.path.join(rsc_path, f"robot/{arm_name}/{arm_name}.urdf")
        )
        end_link = robot.get_link_index("link6")
        
        # process arm
        arm_action_orig = np.load(os.path.join(raw_dir, arm_name, "action.npy"))  # T X 4 X 4
        arm_qpos_orig = np.load(os.path.join(raw_dir, arm_name, "position.npy"))  # T X dof
        
        arm_position_orig = []
        for i in range(len(arm_qpos_orig)):
            robot.compute_forward_kinematics(arm_qpos_orig[i])
            arm_position_orig.append(robot.get_link_pose(end_link))
        arm_position_orig = np.array(arm_position_orig)  # T X 4 X 4
        
        arm_pc_time = np.load(os.path.join(raw_dir, arm_name, "time.npy"))
        
        arm_action_sync = get_synced_data(pc_time, arm_action_orig, arm_pc_time)
        arm_qpos_sync = get_synced_data(pc_time, arm_qpos_orig, arm_pc_time)
        arm_position_sync = get_synced_data(pc_time, arm_position_orig, arm_pc_time)
        
        os.makedirs(os.path.join(index_dir, arm_name), exist_ok=True)
        np.save(os.path.join(index_dir, arm_name, "action.npy"), arm_action_sync)
        # np.save(os.path.join(index_dir, arm_name, "qpos.npy"), arm_qpos_sync)
        np.save(os.path.join(index_dir, arm_name, "position.npy"), arm_position_sync)
        
        # 데이터 통계 출력
        print_data_statistics(pc_time, fid, arm_action_sync, arm_qpos_sync, arm_position_orig, arm_pc_time, index)
        
        # 플롯 생성 (전체 + 확대)
        plot_robot_data(pc_time, fid, arm_action_sync, arm_qpos_sync, arm_position_orig, arm_pc_time, index, index_dir)
        
        print(f"Completed processing for index {index}")