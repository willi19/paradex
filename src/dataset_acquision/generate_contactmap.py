import argparse
import os
import numpy as np
from dex_robot.utils.file_io import shared_path, rsc_path
from dex_robot.visualization.grid_image import grid_image
from dex_robot.utils.robot_wrapper import RobotWrapper
import json
import cv2
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate contact map from a given camera path.")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    robot = RobotWrapper(os.path.join(rsc_path, "xarm6", "xarm6_allegro_wrist_mounted_rotate.urdf"))

    name_list = [args.name] if args.name else os.listdir(os.path.join(shared_path, 'processed'))
    for name in name_list:
        root_path = os.path.join(shared_path, 'processed', name)
        index_list = os.listdir(root_path)
        for index in index_list:
            grasp_info_path = os.path.join(root_path, index, 'grasp_info.json')
            if not os.path.exists(grasp_info_path):
                continue
            
            grasp_info = json.load(open(grasp_info_path, 'r'))

            start = grasp_info['start']
            end = grasp_info['end']
            lift = grasp_info['lift']

            if "success" not in grasp_info:
                print(f"Grasp info for {name}/{index} does not contain success key.")
                grasp_img_dir = os.path.join(root_path, index, 'last_frame')
                
                if not os.path.exists(grasp_img_dir):
                    print(f"Grasp image directory {grasp_img_dir} does not exist.")
                    continue

                img_list = os.listdir(grasp_img_dir)
                
                image_dict = {}

                for img_name in img_list:
                    cam_id = img_name.split('.')[0]
                    
                    img_path = os.path.join(grasp_img_dir, img_name)
                    image_dict[cam_id] = cv2.imread(img_path)

                grid_img = grid_image(image_dict)
                print("Press '1' for success, '0' for fail, or any other key for -1 (unsure)")
                cv2.imshow("Grasp Image", grid_img)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()

                if key == ord('1'):
                    grasp_info['success'] = 1
                elif key == ord('0'):
                    grasp_info['success'] = 0
                else:
                    grasp_info['success'] = -1
                
                with open(grasp_info_path, 'w') as f:
                    json.dump(grasp_info, f, indent=4)

            if grasp_info['success'] == 0 or grasp_info['success'] == -1:
                continue
                
            os.makedirs(os.path.join(shared_path, 'contact_map', name, index), exist_ok=True)
            
            contact_value = np.load(os.path.join(root_path, index, 'contact', 'data.npy'))
            contact_value = np.mean(contact_value[start:end+1], axis=0) - contact_value[0]

            pose_idx = (start+lift) // 2

            robot_pose = np.load(os.path.join(root_path, index, 'hand', 'state.npy'))
            robot_action = np.load(os.path.join(root_path, index, 'hand', 'action.npy'))

            np.save(os.path.join(shared_path, 'contact_map', name, index, 'contact.npy'), contact_value)
            np.save(os.path.join(shared_path, 'contact_map', name, index, 'robot_pose.npy'), robot_pose[pose_idx])
            np.save(os.path.join(shared_path, 'contact_map', name, index, 'robot_action.npy'), robot_action[pose_idx])

            # Todo add ContactMap N x (3 + segment * 2)
            
            arm_pose = np.load(os.path.join(root_path, index, 'arm', 'state.npy'))
            qpos = np.zeros(22)
            qpos[:6] = arm_pose[pose_idx]
            robot.compute_forward_kinematics(qpos)

            wrist_T = robot.get_link_pose(robot.get_link_index('palm_link'))

