from dex_robot.simulate.simulator import simulator
import numpy as np
from scipy.spatial.transform import Rotation
from dex_robot.xsens.receiver import XSensReceiver
from dex_robot.io.xsens import hand_index

home_wrist_pose = np.load("data/home_pose/allegro_eef_frame.npy")
home_hand_pose = np.load("data/home_pose/allegro_hand_joint_angle.npy")
init_obj_pose = np.array(
        [
            [0.53174365, 0.8465994, 0.02276282, 100.49357095],
            [-0.84589686, 0.5331071, 0.01597875, -20.19108522],
            [0.00139258, -0.0277516, 0.99961209, 20.03901474],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ]
)

LINK62PALM = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

XSENS2ISAAC = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

home_wrist_pose = home_wrist_pose @ LINK62PALM
def homo2cart(h):
    if h.shape == (4, 4):
        t = h[:3, 3]
        R = h[:3, :3]
        q = Rotation.from_matrix(R).as_euler("XYZ")
    else:
        t = h[:3]
        q = h[3:]
        q = Rotation.from_quat(q).as_euler("XYZ")

    return np.concatenate([t, q])

def home_robot(sim):
    target_action = np.concatenate([homo2cart(home_wrist_pose), home_hand_pose])
    sim.step(target_action, target_action, init_obj_pose)
    print("Robot homed.")

def main():
    obj_name = "bottle"
    save_video = False
    save_state = False
    view_physics = False
    view_replay = True
    num_sphere = 0
    headless = False
    fixed = False

    sim = simulator(
        obj_name,
        view_physics,
        view_replay,
        num_sphere,
        headless,
        save_video,
        save_state,
    )
    
    

    target_action = np.concatenate([homo2cart(home_wrist_pose), home_hand_pose])

    

    host = "192.168.0.2"
    port = 9763

    xsens_updater = XSensReceiver()
    xsens_updater.init_server(host, port)

    traj_cnt = 5

    init_wrist_pose = None
    init_robot_pose = None
    cur_robot_pose = None
    start = True

    timestamp = []
    robot_action = []
    robot_state = []
    contact_sensor_value = []

    state_2_cnt = 0

    for _ in range(traj_cnt):
        home_robot(sim)
        init_robot_pose = home_wrist_pose.copy()
        cur_robot_pose = init_robot_pose.copy()
        start = True
        target_action = np.concatenate([homo2cart(home_wrist_pose), home_hand_pose])

        while True:
            data = xsens_updater.get_data()
            state = data["state"]

            if state == -1:
                continue
            
            if state == 0:
                #print(data["hand_pose"], init_wrist_pose)
                if start:
                    init_wrist_pose = data["hand_pose"][0].copy()
                    start = False
                try:
                    delta_wrists_R = XSENS2ISAAC[:3,:3].T @ np.linalg.inv(init_wrist_pose[:3,:3]) @ data["hand_pose"][0][:3,:3] @ XSENS2ISAAC[:3,:3]
                except:
                    print(init_wrist_pose, "init")
                    print(data["hand_pose"][0], "cur")
                
                delta_wrists_t = (data["hand_pose"][0][:3,3] - init_wrist_pose[:3,3])
                cur_robot_pose = np.zeros((4,4))
                cur_robot_pose[:3,:3] = init_robot_pose[:3,:3] @ delta_wrists_R
                # print(delta_wrists_t, init_robot_pose[:3,3])
                cur_robot_pose[:3,3] = delta_wrists_t + init_robot_pose[:3,3]
                cur_robot_pose[3,3] = 1

                qpos_sim = np.zeros(16)
                hand_joint_angle = np.zeros((20,3))# data["hand_joint_angle"].copy() 
                hand_pose_frame = data["hand_pose"].copy()  
                # hand_pose_frame = hand_pose_frame @ XSENS2ISAAC.T
                allegro_angles = np.zeros(16)

                for finger_id in range(4):
                    for joint_id in range(4):
                        if joint_id == 0:
                            rot_mat = np.linalg.inv(hand_pose_frame[0,:3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
                        else:
                            rot_mat = np.linalg.inv(hand_pose_frame[hand_index.hand_index_parent[finger_id * 4 + joint_id+1], :3,:3]) @ hand_pose_frame[finger_id * 4 + joint_id + 1, :3,:3]
                        hand_joint_angle[finger_id * 4 + joint_id + 1] = Rotation.from_matrix(rot_mat).as_euler("zyx")
                # zyx euler angle in hand frame = zxy axis angle in robot frame
                allegro_angles[0] = hand_joint_angle[5][0]  # z in robot, y in hand
                allegro_angles[1] = hand_joint_angle[5][2] * 1.2  # y in robot, z in hand
                allegro_angles[2] = hand_joint_angle[6][2] * 0.8
                allegro_angles[3] = hand_joint_angle[7][2] * 0.8

                thumb_meta = np.dot(hand_pose_frame[0,:3,:3].T, hand_pose_frame[1,:3,:3])
                thumb_meta_angle = Rotation.from_matrix(thumb_meta).as_euler("xyz")

                # for drum
                allegro_angles[12] = thumb_meta_angle[0]  # -x in robot, y in hand
                allegro_angles[13] = thumb_meta_angle[1] - 0.5  # y in robot, z in hand
                allegro_angles[14] = hand_joint_angle[2][2] * 1.2
                allegro_angles[15] = hand_joint_angle[3][2] * 1.2

                # for others
                # allegro_angles[4] = thumb_meta_angle[0]  # -x in robot, y in hand
                # allegro_angles[5] = thumb_meta_angle[1] * 0.1  # y in robot, z in hand
                # allegro_angles[6] = hand_joint_angle[2][2] + 1.0
                # allegro_angles[7] = hand_joint_angle[3][2] * 1.2

                allegro_angles[4] = hand_joint_angle[9][0]  # z in robot, y in hand
                allegro_angles[5] = hand_joint_angle[9][2] * 1.2  # y in robot, z in hand
                allegro_angles[6] = hand_joint_angle[10][2] * 0.8
                allegro_angles[7] = hand_joint_angle[11][2] * 0.8

                allegro_angles[8] = hand_joint_angle[13][0]  # z in robot, y in hand
                allegro_angles[9] = hand_joint_angle[13][2] * 1.2  # y in robot, z in hand
                allegro_angles[10] = hand_joint_angle[14][2] * 0.8
                allegro_angles[11] = hand_joint_angle[15][2] * 0.8

                #qpos_sim[6:] = allegro_angles

                target_action = np.concatenate([homo2cart(cur_robot_pose), allegro_angles])

            if state == 1:
                init_wrist_pose = None
                if not start:
                    init_robot_pose = cur_robot_pose.copy()
                cur_robot_pose = None
                start = True
                print("init")

            if state == 2:
                state_2_cnt += 1
                if state_2_cnt > 30:
                    state_2_cnt = 0
                    break
            else:
                state_2_cnt = 0

            sim.step(target_action, target_action, init_obj_pose)


    xsens_updater.quit()
    print("Program terminated.")
    exit(0)
if __name__ == "__main__":
    main()
# 1 0 0
# 0 1 0
# 0 0 1

# Z X Y

# 0 1 0
# 0 0 1
# 1 0 0

# Y Z X

# 0 0 1
# 1 0 0
# 0 1 0
