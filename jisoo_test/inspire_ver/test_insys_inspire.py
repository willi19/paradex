import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from jisoo_test.gui_controller import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
import pickle


def load_trajectory_data(trajectory_path: str):
    """Load trajectory data and planning info from trajectory path.

    Args:
        trajectory_path: Path to trajectory directory containing .npy files
                        Format: .../generated_trajectory_two_steps_optimized/traj/[prefix]_0/

    Returns:
        tuple: (traj_dict, planning_info)
    """
    trajectory_path = Path(trajectory_path)

    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory path does not exist: {trajectory_path}")

    # Derive info path from trajectory path
    # trajectory_path: .../generated_trajectory_two_steps_optimized/traj/[prefix]_0/
    # info_path:       .../generated_trajectory_two_steps_optimized/info/planning_info.pickle
    traj_root = trajectory_path.parent.parent  # Go up two levels: traj/ -> generated_trajectory_two_steps_optimized/
    info_path = traj_root / "info" / "planning_info.pickle"

    print(f">>> Trajectory path: {trajectory_path}")
    print(f">>> Info path: {info_path}")

    # Load planning metadata if available
    planning_info = {}
    if info_path.exists():
        with open(info_path, 'rb') as f:
            planning_info = pickle.load(f)
        print(f">>> Loaded planning info with keys: {list(planning_info.keys())}")
        if 'hand_joints' in planning_info:
            print(f">>> Hand joints: {planning_info['hand_joints']}")
    else:
        print(f">>> Warning: Planning info not found at {info_path}")

    # Load trajectory files
    traj_dict = {}
    for traj_type in ['approach', 'grasp_pose', 'squeeze_pose', 'aftergrasp_pose']:
        traj_file = trajectory_path / f"{traj_type}.npy"
        if traj_file.exists():
            traj_dict[traj_type] = np.load(str(traj_file))
            print(f">>> Loaded {traj_type}: shape {traj_dict[traj_type].shape}")
        else:
            print(f">>> Warning: {traj_type}.npy not found")

    # Check required files
    if 'approach' not in traj_dict:
        raise FileNotFoundError(f"Required file 'approach.npy' not found in {trajectory_path}")
    if 'grasp_pose' not in traj_dict:
        raise FileNotFoundError(f"Required file 'grasp_pose.npy' not found in {trajectory_path}")

    # Optionally load full trajectory as fallback
    full_traj_path = trajectory_path / "full_traj.npy"
    if full_traj_path.exists():
        full_traj = np.load(str(full_traj_path))
        print(f">>> Loaded full_traj: shape {full_traj.shape}")
        traj_dict['full_traj'] = full_traj

    return traj_dict, planning_info


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test Inspire hand with planned trajectories")
parser.add_argument(
    "-t", "--trajectory_path",
    type=str,
    required=True,
    help="Path to trajectory directory (e.g., .../traj/[prefix]_0/)"
)
parser.add_argument(
    "--robot_type",
    type=str,
    default="inspire",
    choices=['inspire', 'allegro'],
    help="Hand type (default: inspire)"
)
args = parser.parse_args()

# Load trajectory data
traj_dict, planning_info = load_trajectory_data(args.trajectory_path)


def qpos_to_inspire_action(qpos):
    """Convert qpos (radians) to inspire integer action (0-1000).

    qpos order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
    inspire action order: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]

    Joint limits from URDF (xarm_inspire.urdf):
        right_thumb_1_joint (yaw):    [0, 1.15]
        right_thumb_2_joint (pitch):  [0, 0.55]
        right_index_1_joint:          [0, 1.6]
        right_middle_1_joint:         [0, 1.6]
        right_ring_1_joint:           [0, 1.6]
        right_little_1_joint:         [0, 1.6]
    """
    limits = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])  # thumb_yaw, thumb_pitch, index, middle, ring, pinky (CORRECT URDF limits)

    if qpos.ndim == 1:
        q = qpos[:6]
        action = np.zeros(6, dtype=np.int32)
        normalized = np.clip(q / limits, 0.0, 1.0)
        # Map: qpos=0 -> action=1000 (open), qpos=limit -> action=0 (closed)
        action_float = (1.0 - normalized) * 1000.0
        # Reorder: qpos [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
        #       -> action [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        action[0] = int(np.clip(action_float[5], 0, 1000))  # pinky
        action[1] = int(np.clip(action_float[4], 0, 1000))  # ring
        action[2] = int(np.clip(action_float[3], 0, 1000))  # middle
        action[3] = int(np.clip(action_float[2], 0, 1000))  # index
        action[4] = int(np.clip(action_float[1], 0, 1000))  # thumb_pitch
        action[5] = int(np.clip(action_float[0], 0, 1000))  # thumb_yaw
    else:
        q = qpos[:, :6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros((qpos.shape[0], 6), dtype=np.int32)
        action[:, 0] = np.clip(action_float[:, 5], 0, 1000).astype(int)  # pinky
        action[:, 1] = np.clip(action_float[:, 4], 0, 1000).astype(int)  # ring
        action[:, 2] = np.clip(action_float[:, 3], 0, 1000).astype(int)  # middle
        action[:, 3] = np.clip(action_float[:, 2], 0, 1000).astype(int)  # index
        action[:, 4] = np.clip(action_float[:, 1], 0, 1000).astype(int)  # thumb_pitch
        action[:, 5] = np.clip(action_float[:, 0], 0, 1000).astype(int)  # thumb_yaw
    return action

# ===== Trajectory Information =====
# From plan_arm_trajectory_twosteps.py:
# - approach.npy: collision-free trajectory from init to pre-grasp [N_approach, 12] (6 arm + 6 hand)
# - grasp_pose.npy: final grasp configuration [1, 12]
# - squeeze_pose.npy: firmer grasp with fingers closed more [1, 12]
# - aftergrasp_pose.npy: post-grasp motion trajectory [N_aftergrasp, 12]
# - planning_info.pickle: metadata (hand_joints, pregrasp_hand_qpos, etc.)

approach = traj_dict['approach']  # NX12: full DOF trajectory
grasp_pose = traj_dict['grasp_pose']  # 1X12: final grasp config
squeeze_pose = traj_dict.get('squeeze_pose', grasp_pose)  # 1X12: squeeze config

# Use full pose (12 DOF: 6 arm + 6 hand)
approach_full = approach.copy()  # NX12
grasp_full = grasp_pose.copy()  # 1X12
squeeze_full = squeeze_pose.copy()  # 1X12

# Extract aftergrasp trajectory if available
aftergrasp_traj = None
if 'aftergrasp_pose' in traj_dict:
    aftergrasp_traj = traj_dict['aftergrasp_pose'].copy()  # NX12 (6 arm + 6 hand DOF)
    print(f">>> Loaded aftergrasp_pose with shape: {aftergrasp_traj.shape}")
    print(f">>> Using aftergrasp trajectory (full DOF)")

    # Validate with planning info if available
    if 'aftergrasp_ik_success' in planning_info:
        print(f">>> Aftergrasp IK success: {planning_info['aftergrasp_ik_success']}")
        if 'aftergrasp_distance' in planning_info:
            print(f">>> Aftergrasp distance: {planning_info['aftergrasp_distance']:.4f}m")

# Create grasp stage poses from trajectories
# Each stage is replicated 6 times for GUI controller compatibility
start = np.tile(approach_full[-1:], (6,1))  # 6X12: start pose (open hand)
pregrasp = np.tile(approach_full[-1,:], (6,1))  # 6X12: pre-grasp pose (still open)
grasp = np.tile(grasp_full, (6,1)) if grasp_full.shape[0] == 1 else grasp_full # 6X12: grasp pose (closed on object)
squeeze = np.tile(squeeze_full, (6,1))  # 6X12: squeeze pose (firmer grasp)

print(f">>> Created grasp stage poses:")
print(f"    start: {start.shape}")
print(f"    pregrasp: {pregrasp.shape}")
print(f"    grasp: {grasp.shape}")
print(f"    squeeze: {squeeze.shape}")

base_pose = np.array([[ 0.07842263, -0.16270738,  0.98355285,  0.4116784 ],
 [ 0.19462326, -0.96510918, -0.17517437, -0.42133483],
 [ 0.97773804,  0.20515989, -0.04401976,  0.41725355],
 [ 0.,          0.,          0.,          1.        ]])

xarm_init_pose = np.array([-0.8048279285430908,
                0.2773207128047943,
                -1.4464116096496582,
                2.0092501640319824,
                0.7059974074363708,
                -2.361839532852173])

predefined_poses = {
    'base': base_pose
}
grasp_pose_dict = {
    'start': start,
    'pregrasp': pregrasp,
    'grasp': grasp,
    'squeezed': squeeze
}
# ===== Initialize Robot Controllers =====
arm = get_arm("xarm")
hand = get_hand("inspire", ip=True)
print(">>> Controllers created.")

# ===== Convert Hand Joint Values to Inspire Actions =====
# Inspire hand expects integer actions [0-1000] instead of radians
# Convert hand portion (indices 6:12) while keeping arm portion (0:6) unchanged

print(">>> Converting hand qpos to inspire actions...")

# Convert approach trajectory
approach[:,6:] = qpos_to_inspire_action(approach[:,6:])
print(f"    Approach hand range: [{approach[:,6:].min():.0f}, {approach[:,6:].max():.0f}]")

# Convert aftergrasp trajectory if available
if aftergrasp_traj is not None:
    aftergrasp_traj[:,6:] = qpos_to_inspire_action(aftergrasp_traj[:,6:])
    print(f"    Aftergrasp hand range: [{aftergrasp_traj[:,6:].min():.0f}, {aftergrasp_traj[:,6:].max():.0f}]")

# Convert grasp stage poses (NX12 -> keep arm [:,:6], convert hand [:,6:])
for grasp_type in grasp_pose_dict:
    grasp_pose_dict[grasp_type][:,6:] = qpos_to_inspire_action(grasp_pose_dict[grasp_type][:,6:])
    print(f"    {grasp_type} hand: {grasp_pose_dict[grasp_type][0,6:]}")

print(">>> All hand poses converted to inspire actions.")


# ===== Display Planning Summary =====
print("\n" + "="*60)
print("PLANNING SUMMARY")
print("="*60)
print(f"Trajectory path: {args.trajectory_path}")
print(f"Approach trajectory: {approach.shape[0]} waypoints")
print(f"Aftergrasp trajectory: {aftergrasp_traj.shape[0] if aftergrasp_traj is not None else 'None'} waypoints")
if 'pregrasp_step' in planning_info:
    print(f"Pre-grasp step used: {planning_info['pregrasp_step']}/5")
if 'grasp_ik_success' in planning_info:
    print(f"Grasp IK success: {planning_info['grasp_ik_success']}")
if 'planning_time' in planning_info:
    print(f"Planning time: {planning_info['planning_time']:.2f}s")
print("="*60 + "\n")

# ===== Initialize GUI Controller =====
rgc = RobotGUIController(
    robot_controller=arm,
    hand_controller=hand,
    predefined_poses=predefined_poses,
    grasp_pose=grasp_pose_dict,
    approach_traj=approach,
    aftergrasp_traj=aftergrasp_traj,  # Use aftergrasp trajectory instead of lift_distance
    lift_distance=100.0 if aftergrasp_traj is None else None,  # Fallback to lift if no trajectory
    place_distance=40.0
)

print(">>> Starting GUI controller...")
rgc.run()