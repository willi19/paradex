import time
import numpy as np
import pickle
import os
from paradex.inference.metric import get_pickplace_timing, compute_mesh_to_ground_distance
import open3d as o3d
from paradex.utils.file_io import rsc_path
from paradex.robot import RobotWrapper
from paradex.io.robot_controller import XArmController, AllegroController, InspireController
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R
import chime
from paradex.simulator.isaac import simulator

LINK62PALM = np.array(
    [
        [0, -1, 0, 0.0],
        [1, 0, 0, 0.0],#0.035],
        [0, 0, 1, -0.18],
        [0, 0, 0, 1],
    ]
)

demo_path = "data_Icra/teleoperation/bottle"
demo_path_list = os.listdir(demo_path)
demo_path_list.sort()

arm_name = "xarm"
hand_name = "allegro"

obj_mesh = o3d.io.read_triangle_mesh(os.path.join(rsc_path, "bottle", "bottle.obj"))

robot = RobotWrapper(
    os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf")
)
link_index = robot.get_link_index("palm_link")
lift_T = 100

def homo2cart(h):
    def project_to_so3(R):
        U, _, Vt = np.linalg.svd(R)
        R_proj = U @ Vt
        if np.linalg.det(R_proj) < 0:
            U[:, -1] *= -1
            R_proj = U @ Vt
        return R_proj


    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]
        
        R_proj = project_to_so3(R)

        axis, angle = t3d.axangles.mat2axangle(R_proj)
        axis_angle = axis * angle
    else:
        raise ValueError("Invalid input shape.")
    return np.concatenate([t, axis_angle])

# def initialize_teleoperation(save_path):
#     controller = {}
#     if arm_name == "xarm":
#         controller["arm"] = XArmController(save_path)

#     if hand_name == "allegro":
#         controller["hand"] = AllegroController(save_path)
        
#     elif hand_name == "inspire":
#         controller["hand"] = InspireController(save_path)
    
#     return controller

array([[-0.12324549,  0.08564689,  0.98867343,  0.3965003 ],
       [ 0.18111348, -0.97759515,  0.10726434, -0.03921698],
       [ 0.97570921,  0.19228193,  0.10497239,  0.06110497],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
def get_object_pose():
    f = open("data_Icra/teleoperation/bottle/1/obj_traj.pickle", "rb")
    obj_pose = pickle.load(f)["bottle"][0]
    
    obj_pose[:2, 3] = np.array([0.57176, 0.047841])  # Adjusted position
    f.close()
    return obj_pose

def determine_theta():
    return 0

def determine_traj_idx():
    return 1

def load_demo(demo_name):
    robot_prev = RobotWrapper(
        os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate_prev.urdf")
    )
    
    obj_T = pickle.load(open(os.path.join(demo_path, demo_name, "obj_traj.pickle"), "rb"))['bottle']
    robot_traj = np.load(os.path.join(demo_path, demo_name, "robot_qpos.npy"))
    target_traj = np.load(os.path.join(demo_path, demo_name, "target_qpos.npy"))

    T = obj_T.shape[0]

    height_list = []

    for step in range(T):
        h = compute_mesh_to_ground_distance(obj_T[step], obj_mesh)    
        height_list.append(h)

    end_T, _ = get_pickplace_timing(height_list)
    
    dist = 0.07
    for step in range(T):
        q_pose = robot_traj[step]
        robot_prev.compute_forward_kinematics(q_pose)
        wrist_pose = robot_prev.get_link_pose(link_index)
    
        obj_wrist_pose = np.linalg.inv(obj_T[step]) @ wrist_pose
        obj_pos = obj_wrist_pose[:2, 3]

        d = np.linalg.norm(obj_pos[:2])
        if d < dist:
            start_T = step
            break

    robot_pose = []
    hand_action = []
    obj_pose = obj_T[start_T].copy()
    tx, ty = obj_pose[:2, 3]

    for i in range(start_T, end_T + 1 + lift_T):
        if i > end_T:
            robot_pose.append(robot_pose[-1].copy())
            robot_pose[-1][2, 3] += 0.001
            hand_action.append(target_traj[end_T, 6:])
            continue

        robot_prev.compute_forward_kinematics(robot_traj[i])
        wrist_pose = robot_prev.get_link_pose(link_index)
        
        # wrist_axangle = target_traj[i, 3:6]
        # angle = np.linalg.norm(wrist_axangle)
        # if angle > 1e-6:
        #     wrist_axis = wrist_axangle / angle
        # else:
        #     wrist_axis = np.zeros(3)
            
        # wrist_rotmat = t3d.axangles.axangle2mat(wrist_axis, angle)

        # wrist_pose = np.zeros((4, 4))
        # wrist_pose[:3, :3] = wrist_rotmat
        # wrist_pose[:3, 3] = robot_traj[i][:3]
        # wrist_pose = np.linalg.inv(obj_T[0]) @ wrist_pose

        hand_action.append(target_traj[i, 6:])
        wrist_pose[:2, 3] -= np.array([tx, ty])  # Adjust position relative to object
        robot_pose.append(wrist_pose.copy())
    
    robot_pose = np.array(robot_pose)
    hand_action = np.array(hand_action)
    
    
    desired_x = np.array([0, 1, 0])
    current_x = robot_pose[-1, :3, 0][:2]  # x축 방향 (2D)
    theta = np.arctan2(desired_x[1], desired_x[0]) - np.arctan2(current_x[1], current_x[0])

    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ])

    obj_pose[:2, 3] -= np.array([tx, ty])
    # robot_pose[:, :2, 3] -= np.array([tx, ty])
    robot_pose[:, 2, 3] += (0.08 - robot_pose[0, 2, 3])

    robot_pose = np.einsum('ij, kjl -> kil', rot_mat, robot_pose)
    # import pdb; pdb.set_trace()
    return obj_pose, robot_pose, hand_action

def get_traj(tx, ty, theta, wrist_pose_demo):
    rot_mat = R.from_euler('z', theta, degrees=True).as_matrix()
    rot_T = np.eye(4)
    rot_T[:3, :3] = rot_mat

    wrist_pose = wrist_pose_demo.copy()
    wrist_pose = np.einsum('ij, kjl -> kil', rot_T, wrist_pose)
    wrist_pose[:, :2, 3] += np.array([tx, ty]) 

    return wrist_pose

if __name__ == "__main__":
    obj_name = "bottle"
    save_video = False
    save_state = False
    view_physics = False
    view_replay = True
    headless = False


    sim = simulator(
        obj_name,
        view_physics,
        view_replay,
        headless,
        save_video,
        save_state,
        fixed=True
    )
    
    
    object_pose = get_object_pose()
    tx, ty = object_pose[:2, 3]
    
    theta = determine_theta()
    traj_idx = determine_traj_idx()
    traj_idx = str(traj_idx)
    
    obj_pose_demo, wrist_pose_demo, hand_pose = load_demo(traj_idx)

    transformed_traj = get_traj(tx, ty, theta, wrist_pose_demo)
    # sensors = initialize_teleoperation(None)
    
    # if hand_name is not None:
    #     sensors["hand"].set_homepose(hand_pose[0])
    #     sensors["hand"].home_robot()

    # if arm_name is not None:
    #     sensors["arm"].set_homepose(homo2cart(transformed_traj[0]))
    #     sensors["arm"].home_robot()

    #     home_start_time = time.time()
    #     while sensors["arm"].ready_array[0] != 1:
    #         if time.time() - home_start_time > 0.3:
    #             chime.warning()
    #             home_start_time = time.time()
    #         time.sleep(0.0008)
    #     chime.success()
    robot_pose = np.zeros(22)
    
    step = 0
    #while True:
    for step in range(len(transformed_traj)*5):
        i = step
        l6_pose =  transformed_traj[i] @ LINK62PALM
        l6_idx = robot.get_link_index("link6")
        
        #robot.compute_forward_kinematics(robot_pose)
        #wrist_pose = robot.get_link_pose(link_index)
        # l6_pose = robot.get_link_pose(l6_idx)
        
        
        
        # arm_action = homo2cart(l6_pose)
        hand_action = hand_pose[i]
        import pdb; pdb.set_trace()
        q_ik, success = robot.solve_ik(
            l6_pose,
            "link6",
            q_init=robot_pose,
            
        )
        # q_ik, success = robot.solve_ik(
        #     transformed_traj[i],
        #     "palm_link",
        #     q_init=robot_pose,
            
        # )
        print(f"Step {i}, IK success: {success}")
        if not success:
            robot.compute_forward_kinematics(q_ik)
            cur_wrist_pose = robot.get_link_pose(link_index)
            print(cur_wrist_pose[:3, 3]-transformed_traj[i][:3, 3])
            
        robot_pose = q_ik.copy()
        robot_pose[6:] = hand_action
        # action = np.concatenate([robot_pose, hand_action])
        
        sim.step(robot_pose, robot_pose, object_pose)
        # if arm_name is not None:                
        #     sensors["arm"].set_target_action(
        #                     arm_action
        #             )
        # if hand_name is not None:
        #     sensors["hand"].set_target_action(
        #                     hand_action
        #                 )
                    
        # time.sleep(0.1)  # Simulate time taken for each action
        print(i, len(transformed_traj), "Robot action executed.")
