import os
import argparse
import numpy as np
import copy
import viser
import time
import trimesh

from pymodbus.client import ModbusTcpClient
from paradex.io.robot_controller.under_test.visualize_contact import read_modbus_data
from paradex.io.robot_controller.under_test.inspire_contact_info import contact_tg, sensororder
from paradex.robot.robot_module import robot_info
from paradex.robot.robot_wrapper_deprecated import RobotWrapper




from hand_rh56dftp import InspireHandRH56DFTP


from paradex.utils.path import shared_dir
from paradex.io.robot_controller import  get_hand

URDF_PATH = "/home/temp_id/paradex/rsc/robot/inspire_left.urdf"
BASE_T = np.eye(4)

def forward_kinematic(robot_wrapper: RobotWrapper, state: np.ndarray = None):
    robot_wrapper.compute_forward_kinematics(state)
        

def get_mesh(robot_wrapper, state, base_T=np.eye(4), mesh_tg = 'all'):

    forward_kinematic(robot_wrapper, state)
    robot_obj = robot_info(URDF_PATH, down_sample=True)
    
    link_list = []
    for link_nm, mesh_items in robot_obj.mesh_dict.items():
        if mesh_items!=[]:
            link_list.append(link_nm)

    vis_list = []
    
    for link_nm in link_list:
        link_pose = base_T@robot_wrapper.get_link_pose(robot_wrapper.get_link_index(link_nm))
        # print(f'{link_nm}: {link_pose}')
        for o3d_mesh in robot_obj.mesh_dict[link_nm]:
            mesh = copy.deepcopy(o3d_mesh)
            mesh.transform(link_pose)

            tm = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                process=False,
            )
            
            
            vis_list.append((link_nm, tm))

    return vis_list


# def get_T_dict(self, robot_wrapper, state, base_T=np.eye(4)):
#     forward_kinematic(robot_wrapper, state)
    
#     T_dict = {}
#     for link_nm in self.link_list:
#         link_pose = base_T@robot_wrapper.get_link_pose(robot_wrapper.get_link_index(link_nm))
#         T_dict[link_nm] = link_pose
#     return T_dict
    


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     MODBUS_IP = "192.168.11.210"
#     MODBUS_PORT = 6000
#     TOUCH_SENSOR_BASE_ADDR = 3000
#     TOTAL_REG_COUNT = 2124  # 5123 - 3000 + 1
    
#     robot_wrapper = RobotWrapper('/home/temp_id/paradex/rsc/robot/inspire_left.urdf')

#     # parser.add_argument('--hand', type=str, default=None)
#     # parser.add_argument('--index', type=str, required=True)

#     # args = parser.parse_args()
    
#     # hand = get_hand("inspire")
#     # hand.start(os.path.join(shared_dir, "capture", "contact_test", args.index, "raw", "hand"))
#     # client = ModbusTcpClient(MODBUS_IP, port=MODBUS_PORT)
#     # client.connect()
#     # while True:
#     #     all_register = read_modbus_data(client)
#     #     if len(all_register) != TOTAL_REG_COUNT:
#     #         print(":느낌표: Unexpected data length")
#     #         continue
#     #     print(all_register)
    
    
    
#     # Setup the mock to return predictable data based on address
#     def side_effect(address, count):
#         # Return values equal to the address
#         # This allows us to verify that we are reading from the correct location
#         return list(range(address, address + count))
    
#     hand = InspireHandRH56DFTP()
#     hand.open() # Ensure connected
    
#     print("Testing read_tactile_data()...")
#     data = hand.read_pose()
    
#     print(data)




# -------------------------
# Utility
# -------------------------
def visualize_robot_once(
    server: viser.ViserServer,
    robot_wrapper: RobotWrapper,
    qpos: np.ndarray,
    base_T: np.ndarray = np.eye(4),
):
    """
    qpos: (ndof,) numpy array
    """

    # FK
    robot_wrapper.compute_forward_kinematics(qpos)

    # clear previous meshes
    server.scene.reset()

    # Iterate links
    # for link_name in robot_wrapper.link_names:
    #     link_idx = robot_wrapper.get_link_index(link_name)
    #     link_T = base_T @ robot_wrapper.get_link_pose(link_idx)


    #     for mi, mesh in enumerate(robot_wrapper.robot_obj.mesh_dict[link_name]):
    #         mesh_vis = copy.deepcopy(mesh)
    #         mesh_vis.apply_transform(link_T)

    mesh_vis = get_mesh(robot_wrapper, qpos)


    for i, (link_name, tm) in enumerate(mesh_vis):
        print(link_name)
        server.scene.add_mesh_trimesh(
            name=f"inspire_hand/{link_name}_{i}",
            mesh=tm,
            scale=1.0,
            wxyz=(1.0, 0.0, 0.0, 0.0),   # 이미 pose bake됨
            position=(0.0, 0.0, 0.0),
            visible=True,
        )

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # 1. Start viser
    server = viser.ViserServer()

    # 2. Load robot
    robot_wrapper = RobotWrapper(URDF_PATH)

    # 3. Assume this comes from hand.read_pose()
    # 예시: 6 DoF
    # data = np.array([0.0, 0.3, -0.2, 0.1, 0.0, 0.0], dtype=np.float32)

    # ⚠️ URDF joint 수와 다르면 여기서 맞춰야 함
    # data = data[:robot_wrapper.ndof]
    
    hand = InspireHandRH56DFTP()
    hand.open() # Ensure connected
    
    print("Testing read_tactile_data()...")
    data = hand.read_angles()
    data = np.array(data, dtype=np.float32)

    print(data)
    # 4. Visualize once
    visualize_robot_once(
        server=server,
        robot_wrapper=robot_wrapper,
        qpos=data,
    )

    # 5. Keep server alive
    while True:
        time.sleep(0.1)