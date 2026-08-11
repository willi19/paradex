import numpy as np
import os
import transforms3d
import json
import trimesh
import torch
import time
from planner import CuroboPlanner
from planner import load_world_config
from curobo.types.base import TensorDeviceType
from curobo.util_file import load_yaml
from curobo.types.math import Pose
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from paradex.utils.path import shared_dir, home_path
from paradex.utils.file_io import find_latest_directory
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.utils import get_robot_urdf_path
from paradex.robot.robot_wrapper import RobotWrapper
mesh_path = os.path.join(shared_dir, "RSS2026_Mingi", "object", "paradex")
OBSTACLE = {
            'cuboid':
                {
                 'table': {'dims': [2, 3, 0.2], 'pose': [1.1,0,-0.1+0.037,0,0,0,1]},
                }
            }
def check_valid(q0, q1):
    def sample_line(q0, q1, steps=20):
        t = torch.linspace(0, 1, steps, device=q0.device, dtype=q0.dtype).unsqueeze(-1)
        return q0.unsqueeze(0) * (1 - t) + q1.unsqueeze(0) * t  # [steps, dof]
    robot_cfg = load_yaml(os.path.join("src/curobo/content/configs/robot/xarm_allegro.yml"))["robot_cfg"]
    world_config = load_world_config(OBSTACLE, {})
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(
        robot_cfg, world_config, collision_activation_distance=0.0, tensor_args=tensor_args
    )
    rw = RobotWorld(config)
    q0 = torch.tensor(q0, dtype=torch.float32).to(tensor_args.device)
    q1 = torch.tensor(q1, dtype=torch.float32).to(tensor_args.device)
    traj = sample_line(q0, q1, steps=20)
    d_world, d_self = rw.get_world_self_collision_distance_from_joints(traj)
    collided = (d_world > 0).any() or (d_self > 0).any()
    return not bool(collided)
def cart2se3(cart):
    se3 = np.eye(4)
    se3[:3, 3] = cart[:3]
    quat = cart[3:7]
    se3[:3, :3] = transforms3d.quaternions.quat2mat(quat)
    return se3
def load_planner(obj_name, obj_pose):
    obj_dict = {
        obj_name: {
            'pose': obj_pose,
            'file_path': os.path.join(mesh_path, obj_name, "raw_mesh", f"{obj_name}.obj"),
            'urdf_path': os.path.join(mesh_path, obj_name, "processed_data", "urdf", "coacd.urdf"),
        }
    }
    robot_cfg = load_yaml(os.path.join("src/curobo/content/configs/robot/xarm_allegro.yml"))["robot_cfg"]
    tensor_args = TensorDeviceType()
    return CuroboPlanner(OBSTACLE, obj_dict, robot_cfg, tensor_args)
root_dir = os.path.join(shared_dir, "RSS2026_Mingi/candidate/after_sim")
rbwrp = RobotWrapper(os.path.join("src/curobo/content/assets/robot/allegro_description/allegro_hand_description_right.urdf"))
rbwrp_tot = RobotWrapper(os.path.join("src/curobo/content/assets/robot/allegro_description/xarm_allegro.urdf"))
xarm_init_pose = np.array([
    -0.8048279285430908,
    0.2773207128047943,
    -1.4464116096496582,
    2.0092501640319824,
    0.7059974074363708,
    -2.361839532852173
])
init_pose = np.concatenate([xarm_init_pose, np.zeros(16)])
init_pose[-4] = 0.5
obj_name = "pringles"
obj_pose_idx = find_latest_directory(os.path.join(shared_dir, "inference", "bodex"))
obj_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "obj_T.npy"))
idx_list = os.listdir(os.path.join(root_dir, obj_name))
obj_mesh_path = os.path.join(mesh_path, obj_name, "raw_mesh", f"{obj_name}.obj")
mesh = trimesh.load(obj_mesh_path)
sst = time.time()
planner = load_planner(obj_name, obj_pose)
wrist_se3 = []
link_poses_dict = {
    "base_link": [],
    # # 손가락 0
    "link_2.0": [],
    "link_3.0": [],
    "link_3.0_tip": [],
    # 손가락 1
    "link_6.0": [],
    "link_7.0": [],
    "link_7.0_tip": [],
    # 손가락 2
    "link_10.0": [],
    "link_11.0": [],
    "link_11.0_tip": [],
    # 엄지
    "link_14.0": [],
    "link_15.0": [],
    "link_15.0_tip": [],
}
for idx in idx_list:
    robot_pose = np.load(os.path.join(root_dir, obj_name, idx, "robot_pose.npy"))
    pregrasp_pose = robot_pose[0]
    grasp_se3 = cart2se3(pregrasp_pose[:7])
    wrist_se3.append(obj_pose @ grasp_se3)
    link_poses_np = rbwrp.compute_forward_kinematics(
        pregrasp_pose[7:],
        link_list=list(link_poses_dict.keys())
    )
    # 각 링크별로 SE3를 저장
    for name in link_poses_np.keys():
        link_se3 = obj_pose @ grasp_se3 @ link_poses_np[name]
        link_poses_dict[name].append(link_se3)
wrist_se3 = np.array(wrist_se3)
for name in link_poses_dict.keys():
    link_poses_dict[name] = np.array(link_poses_dict[name])  # [batch, 4, 4]
succ, traj_list, result = planner.plan_batch(init_pose, wrist_se3, link_poses_dict=link_poses_dict)
succ_idx = np.where(succ)[0]
print(succ_idx, "succeed indices")
succ_traj_list = []
for idx in succ_idx:
    file_name = idx_list[idx]
    robot_pose = np.load(os.path.join(root_dir, obj_name, file_name, "robot_pose.npy"))
    pregrasp_pose = robot_pose[0][7:]
    grasp_pose = robot_pose[1][7:]
    traj = traj_list[idx]
    traj_tmp = traj[-1].copy()
    traj_tmp[6:] = pregrasp_pose  # set finger joints to pregr
    os.makedirs(os.path.join(shared_dir, "experiment", "squeeze_pose", obj_name, file_name), exist_ok=True)
    squeeze_idx = len(os.listdir(os.path.join(shared_dir, "experiment", "squeeze_pose", obj_name, file_name))) + 1
    if squeeze_idx > 10:
        print(f"Already have 10 squeeze poses for {file_name}, skip.")
        continue
    squeeze_pose = grasp_pose * (squeeze_idx + 1) - pregrasp_pose * squeeze_idx
    if check_valid(traj[-1], traj_tmp):
        print(squeeze_idx, file_name)
        info_dict = {
            "obj_name": obj_name,
            "file_name": file_name,
            "squeeze_idx": squeeze_idx,
        }
        np.save(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "approach.npy"), traj)
        np.save(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "pregrasp.npy"), pregrasp_pose)
        np.save(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "grasp_pose.npy"), grasp_pose)
        np.save(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, f"squeeze_pose.npy"), squeeze_pose)
        json.dump(info_dict, open(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, f"input_info.json"), 'w'))
        break
if idx == succ_idx[-1]:
    print("No valid squeeze pose found.")
# first_succ_idx = np.where(succ)[0][0]
# succ_hand_pose = np.load(os.path.join(root_dir, obj_name, idx_list[first_succ_idx], "robot_pose.npy"))
# succ_traj = traj[first_succ_idx]
# succ_traj[:, 6:] = np.load(os.path.join(root_dir, obj_name, idx_list[first_succ_idx], "robot_pose.npy"))[0][7:]  # set finger joints to pregrasp pose
# np.save("traj", succ_traj_list[0])
traj_info = {
}
# traj_merged = np.concatenate([traj[i] for i in range(traj.shape[0])], axis=0)
vis = ViserViewer()
vis.add_object(f"object", mesh, obj_pose)
vis.add_robot("robot", os.path.join("src/curobo/content/assets/robot/allegro_description/xarm_allegro.urdf"))
# vis.add_robot("debug", os.path.join("/home/robot/paradex/rsc/robot/xarm_allegro.urdf"))
vis.add_traj("traj", {"robot": traj})
vis.add_floor(height=0.037)
vis.start_viewer()
# [-0.5898058   0.23250508 -0.7627127   1.5369673   1.2416404   0.28760543
#   0.02896557  0.09037924  0.21543327  0.29215217 -0.02722958  0.2643849
#   0.12115008  0.24228528 -0.07622969 -0.02339935  0.0050975   0.2658659
#   0.8732907  -0.01357394 -0.09024085 -0.05836986]
4:14
import os
import numpy as np
import glob
import json
from paradex.utils.path import shared_dir
from paradex.utils.file_io import find_latest_directory
from paradex.io.robot_controller.gui_controller import RobotGUIController
from paradex.io.robot_controller import get_arm, get_hand
def convert(hand_pose):
    if hand_pose.shape == (16,):
        pose = hand_pose.copy()
        pose[:4] = hand_pose[12:]
        pose[4:] = hand_pose[:12]
    else:
        pose = hand_pose.copy()
        pose[:, :4] = hand_pose[:, 12:]
        pose[:, 4:] = hand_pose[:, :12]
    return pose
base_pose = np.array([[ 0.07842263, -0.16270738,  0.98355285,  0.4116784 ],
 [ 0.19462326, -0.96510918, -0.17517437, -0.42133483],
 [ 0.97773804,  0.20515989, -0.04401976,  0.41725355],
 [ 0.,          0.,          0.,          1.        ]])
obj_pose_idx = find_latest_directory(os.path.join(shared_dir, "inference", "bodex"))
approach_traj = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "approach.npy"))
approach_traj[:, 6:] = convert(approach_traj[:, 6:])
pregrasp_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "pregrasp.npy"))
grasp_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "grasp_pose.npy"))
squeeze_pose = np.load(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "squeeze_pose.npy"))
predefined_poses = {
    'base': base_pose
}
grasp_pose_dict = {
    'start': approach_traj[-1, 6:],
    'pregrasp': convert(pregrasp_pose),
    'grasp': convert(grasp_pose),
    'squeezed': convert(squeeze_pose)
}
arm = get_arm("xarm")
hand = get_hand("allegro")
print(">>> Controllers created.")
rgc = RobotGUIController(
    robot_controller=arm,
    hand_controller=hand,
    predefined_poses=predefined_poses,
    grasp_pose=grasp_pose_dict,
    approach_traj=approach_traj,
    lift_distance=100.0,
    place_distance=40.0
)
rgc.run()
input_info = json.load(open(os.path.join(shared_dir, "inference", "bodex", obj_pose_idx, "input_info.json"), 'r'))
result_file = os.path.join(shared_dir, "experiment", "squeeze_pose", input_info["obj_name"], input_info["file_name"], f"result_{input_info['squeeze_idx']}.json")
result = {}
while True:
    end = input("Success? (y/n): ")
    if end.lower() == 'y':
        result['success'] = True
        break
    elif end.lower() == 'n':
        result['success'] = False
        break
json.dump(result, open(result_file, 'w'))