import copy
import numpy as np
import open3d as o3d
import trimesh


from paradex.robot.robot_module import robot_info
from paradex.robot.robot_wrapper_deprecated import RobotWrapper

# =========================
# Config
# =========================
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

# =========================
# FK + mesh extraction
# =========================
def get_link_o3d_mesh_meanpose(
    robot_wrapper: RobotWrapper,
    target_link: str,
    base_T=np.eye(4),
):
    """
    get_mesh()를 이용해서
    mean pose(qpos=0)에서 target_link의 Open3D mesh를 world 좌표로 bake
    """

    # -------------------------
    # mean pose
    # -------------------------
    qpos = np.zeros(12, dtype=np.float32)  # Inspire hand DOF

    # -------------------------
    # get all meshes (이미 FK + bake 완료)
    # -------------------------
    mesh_list = get_mesh(
        robot_wrapper=robot_wrapper,
        state=qpos,
        base_T=base_T,
    )
    # mesh_list: List[(link_name, trimesh.Trimesh)]

    # -------------------------
    # target link mesh만 추출
    # -------------------------
    target_meshes = []
    for link_nm, tm in mesh_list:
        print(link_nm)
        if link_nm == target_link:
            # trimesh → open3d 변환
            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(tm.vertices),
                triangles=o3d.utility.Vector3iVector(tm.faces),
            )
            target_meshes.append(o3d_mesh)

    if len(target_meshes) == 0:
        raise RuntimeError(f"[ERROR] No mesh found for link {target_link}")

    # -------------------------
    # 여러 mesh면 merge
    # -------------------------
    mesh = target_meshes[0]
    for m in target_meshes[1:]:
        mesh += m

    mesh.compute_vertex_normals()
    return mesh


# =========================
# Pick vertices
# =========================
def pick_vertices(mesh: o3d.geometry.TriangleMesh, link_name: str):
    print("=" * 70)
    print(f"[PICK MODE - MESH + POINTCLOUD] link = {link_name}")
    print(" Left Click : pick point (on pointcloud)")
    print(" Q          : quit")
    print("=" * 70)

    # -------------------------
    # PointCloud (vertex 그대로 사용)
    # -------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    # point 색 (눈에 잘 띄게)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 0.0]]), (len(mesh.vertices), 1))
    )

    # -------------------------
    # VisualizerWithEditing
    # -------------------------
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"Pick {link_name}")

    # mesh는 보기용
    # vis.add_geometry(mesh)

    # pointcloud는 pick용
    vis.add_geometry(pcd)

    # render option
    opt = vis.get_render_option()
    opt.point_size = 10.0
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True  # 내부도 보이게

    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    print(f"\n[PICKED VERTEX INDICES] {picked}\n")

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
    TARGET_LINK = "base_link"
    # 예시:
    # "thumb_tip"
    # "index_intermediate"
    # "middle_proximal"
    # "hand_base_link"

    # -------------------------
    # Mesh load (mean pose)
    # -------------------------
    print(robot_wrapper.link_names)
    mesh = get_link_o3d_mesh_meanpose(
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
    
    
    pick_vertices(mesh, TARGET_LINK)

    # -------------------------
    # (선택) arrow anchor 계산 테스트
    # -------------------------
    # Open3D 출력값으로 교체해서 확인 가능
    example_vertex_indices = [0, 1, 2]  # ← 나중에 교체

    start_pt, normal = compute_arrow_anchor(mesh, example_vertex_indices)

    print("====================================")
    print("[EXAMPLE ARROW ANCHOR]")
    print(" start_point :", start_pt)
    print(" normal      :", normal)
    print("====================================")

    print("\n[DONE] vertex index picking only. qpos has no effect.")
