import numpy as np
import cv2

from paradex.geometry.conversion import project

def get_cammtx(intrinsic, extrinsic):
    cammat = {}
    for serial_num in list(intrinsic.keys()):
        int_mat = intrinsic[serial_num]["intrinsics_undistort"]
        ext_mat = extrinsic[serial_num]
        cammat[serial_num] = int_mat @ ext_mat
    return cammat

def project_point(verts, cammtx, image, color=(255, 0, 0)):
    vert_2d = project(cammtx, verts)
    h, w, _ = image.shape
    for v in vert_2d:
        if v[0] > w or v[0] < 0 or v[1] > h or v[1] < 0:
            continue
        image = cv2.circle(image, (int(v[0]), int(v[1])), 5, color, -1)
    return image

def project_mesh(image, mesh, intrinsic, extrinsic, obj_T=None, renderer=None):
    import trimesh
    import pyrender
    import open3d as o3d
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 0.2, 0.2, 0.4],  
        metallicFactor=0.2,
        roughnessFactor=0.5
    )
    
    if isinstance(mesh, trimesh.Trimesh):
        # 이미 trimesh일 경우 바로 사용
        pyr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        pyr_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)

    # ✅ Scene 준비
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.6, 0.6, 0.6])

    # ✅ obj_T: 없으면 기본 Z+ 방향 0.5m 앞
    if obj_T is None:
        obj_T = np.eye(4)
        obj_T[2, 3] = 0.5  # 카메라 기준 +Z 방향 0.5m 전방

    # ✅ Mesh 위치 설정
    # print(obj_T)
    scene.add(pyr_mesh, pose=obj_T)
    # v = pyrender.Viewer(scene, shadows=True)
    
    # ✅ 카메라 설정
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

    # ✅ extrinsic = [R | t], world_T_camera
    cam_pose = np.eye(4)
    cam_pose[:3, :] = extrinsic
    cam_pose = np.linalg.inv(cam_pose)  # camera_T_world → pyrender pose
    cam_pose = cam_pose @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

    scene.add(camera, pose=cam_pose)

    # ✅ Light도 카메라 위치에
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    # ✅ Render
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=image.shape[1],
            viewport_height=image.shape[0]
        )
    color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    
    color_rgb = color_rgba[..., :3].astype(np.float32)
    alpha = color_rgba[..., 3:] / 255.0  # 0~1로 정규화

    # 원본 image를 float32로 변환
    image_f = image.astype(np.float32)

    # 알파 블렌딩: 렌더된 부분과 원본 이미지 합성
    blended = (color_rgb * alpha + image_f * (1 - alpha)).astype(np.uint8)
    return blended

def project_mesh_nvdiff(object, renderer):
    # from paradex.visualization_.renderer import BatchRenderer
    
    img, mask = renderer.render(object)
    return img, mask