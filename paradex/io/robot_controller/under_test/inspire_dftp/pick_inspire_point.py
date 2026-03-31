import copy
import numpy as np
import open3d as o3d
import trimesh


from paradex.robot.robot_module import robot_info
from paradex.robot.robot_wrapper_deprecated import RobotWrapper

# =========================
# Config
# =========================
URDF_PATH = "/home/temp_id/paradex/rsc/robot/KISTAR_URDF/KISTAR.urdf"
FINGER_TIP_MESH_PATH = "/home/temp_id/paradex/rsc/robot/KISTAR_URDF/meshes/kistar/finger_tip.STL"
BASE_T = np.eye(4)


def forward_kinematic(robot_wrapper: RobotWrapper, state: np.ndarray = None):
    robot_wrapper.compute_forward_kinematics(state)
        

def get_mesh(robot_wrapper, state, base_T=np.eye(4), mesh_tg = 'all'):

    forward_kinematic(robot_wrapper, state)
    robot_obj = robot_info(URDF_PATH, down_sample=False)
    
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


def get_target_mesh_and_pcd_meanpose(
    robot_wrapper: RobotWrapper,
    target_link: str,
    base_T=np.eye(4),
):
    # finger_tip.STL 로컬 vertex index를 그대로 쓰기 위해 FK pose를 적용하지 않음
    target_tm = trimesh.load(
        FINGER_TIP_MESH_PATH,
        force="mesh",
        process=False,
    )
    target_mesh = trimesh_to_o3d_mesh(target_tm)
    target_mesh.compute_vertex_normals()

    target_points = np.asarray(target_tm.vertices).copy()
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    target_pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 0.0]]), (len(target_points), 1))
    )

    return target_mesh, target_pcd


def trimesh_to_o3d_mesh(tm: trimesh.Trimesh):
    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tm.vertices),
        triangles=o3d.utility.Vector3iVector(tm.faces),
    )


# =========================
# Pick vertices
# =========================
def pick_vertices(
    target_pcd: o3d.geometry.PointCloud,
    link_name: str,
):
    print("=" * 70)
    print(f"[PICK MODE - POINTCLOUD ONLY] link = {link_name}")
    print(" Left Click : pick point (on pointcloud)")
    print(" Q          : quit")
    print("=" * 70)

    # FK 이후 target_link 기준 point cloud
    target_count = len(target_pcd.points)
    # -------------------------
    # VisualizerWithEditing
    # -------------------------
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"Pick {link_name}")

    # target pointcloud (pick 대상)
    vis.add_geometry(target_pcd)

    # render option
    opt = vis.get_render_option()
    opt.point_size = 10.0
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()

    raw_picked = vis.get_picked_points()
    picked = []
    for idx in raw_picked:
        # target pcd 기준 인덱스만 사용
        if 0 <= idx < target_count:
            picked.append(idx)

    # 중복 제거 + 정렬
    picked = sorted(set(picked))
    print(f"\n[PICKED TARGET-LINK VERTEX INDICES] {picked}\n")

    return picked


# =========================
# (Optional) Arrow anchor check
# =========================
def compute_arrow_anchor(mesh, vertex_indices):
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    v_idx = np.array(vertex_indices, dtype=np.int64)

    start_point = vertices[v_idx].mean(axis=0)
    normal = normals[v_idx].mean(axis=0)
    normal /= np.linalg.norm(normal)

    return start_point, normal

# =========================
# Main
# =========================
if __name__ == "__main__":

    # -------------------------
    # Robot init
    # -------------------------
    robot_wrapper = RobotWrapper(URDF_PATH)

    # -------------------------
    # 🔧 여기서 link 이름만 바꿔가며 pick
    # -------------------------
    TARGET_LINK = "index_tip"
    # 예시:
    # "thumb_tip"
    # "index_intermediate"
    # "middle_proximal"
    # "hand_base_link"

    # -------------------------
    # Mesh load (mean pose)
    # -------------------------
    print(robot_wrapper.link_names)
    target_mesh, target_pcd = get_target_mesh_and_pcd_meanpose(
        robot_wrapper=robot_wrapper,
        target_link=TARGET_LINK,
    )

    # -------------------------
    # Pick 실행
    # -------------------------
    #['base', 'hand_base_link', 'index_proximal', 'index_intermediate', 
    # 'index_tip', 'middle_proximal', 'middle_intermediate', 'middle_tip', 
    # 'pinky_proximal', 'pinky_intermediate', 'pinky_tip', 'ring_proximal', 
    # 'ring_intermediate', 'ring_tip', 'thumb_proximal_base', 'thumb_proximal', 
    # 'thumb_intermediate', 'thumb_distal', 'thumb_tip']
    
    
    picked_indices = pick_vertices(
        target_pcd,
        TARGET_LINK,
    )

    # -------------------------
    # (선택) arrow anchor 계산 테스트
    # -------------------------
    # Open3D 출력값으로 교체해서 확인 가능
    example_vertex_indices = [0, 1, 2]  # ← 나중에 교체

    if len(picked_indices) >= 3:
        example_vertex_indices = picked_indices[:3]

    start_pt, normal = compute_arrow_anchor(target_mesh, example_vertex_indices)

    print("====================================")
    print("[EXAMPLE ARROW ANCHOR]")
    print(" start_point :", start_pt)
    print(" normal      :", normal)
    print("====================================")

    print("\n[DONE] vertex index picking only. qpos has no effect.")
