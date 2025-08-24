import os
import time
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import trimesh
import viser
import viser.transforms as tf

try:
    # Provided by your project
    from paradex.pose_utils.vis_utils import get_initial_mesh
except Exception:
    get_initial_mesh = None  # Fallback if running standalone

# ---- MEDIA PIPELINE (21 pts) TOPOLOGY ----
MEDIA_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # little
    (0, 17),                                   # wrist to little base
]


def _homogenize(T: np.ndarray) -> np.ndarray:
    """Ensure a 4x4 homogeneous transform from (3x4) or (4x4)."""
    T = np.asarray(T)
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        out = np.eye(4, dtype=T.dtype)
        out[:3, :4] = T
        return out
    raise ValueError(f"Unsupported transform shape: {T.shape}")


def _world_from_extrinsic(extrinsic_wc: np.ndarray) -> np.ndarray:
    """Given world->camera extrinsic, return camera->world 4x4."""
    E = _homogenize(extrinsic_wc)
    return tf.SE3.from_matrix(extrinsic_wc).inverse()


def _edge_segments_from_keypoints(kp_xyz: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Build (E, 2, 3) line segments from (N, 3) keypoints and edges."""
    segs = []
    for i, j in edges:
        if i < len(kp_xyz) and j < len(kp_xyz):
            segs.append(np.stack([kp_xyz[i], kp_xyz[j]], axis=0))
    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack(segs, axis=0).astype(np.float32)


def make_checkerboard_plane_trimesh(
    center: np.ndarray,
    normal: np.ndarray,
    size: float = 10.0,
    step: float = 0.25,
    color_a=(255, 255, 255, 255),
    color_b=(170, 170, 170, 255),
) -> trimesh.Trimesh:
    """
    Build a colored checkerboard plane centered at `center` with normal `normal`.
    Coordinates are already in world space; no extra transform needed on add.
    """
    n = normal / (np.linalg.norm(normal) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)

    half = size * 0.5
    nx = int(np.round(size / step))
    ny = int(np.round(size / step))

    verts, faces, colors = [], [], []
    for i in range(nx):
        for j in range(ny):
            x0, y0 = -half + i * step, -half + j * step
            x1, y1 = x0 + step, y0 + step

            p0 = center + u * x0 + v * y0
            p1 = center + u * x1 + v * y0
            p2 = center + u * x1 + v * y1
            p3 = center + u * x0 + v * y1

            base = len(verts)
            verts.extend([p0, p1, p2, p3])
            faces.append([base + 0, base + 1, base + 2])
            faces.append([base + 0, base + 2, base + 3])

            c = color_a if ((i + j) % 2 == 0) else color_b
            colors.extend([c, c, c, c])

    mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces, dtype=np.int32), process=False)
    mesh.visual.vertex_colors = np.array(colors, dtype=np.uint8)
    return mesh


class KeypointObjectCameraVisualizer:
    """
    Viser-based visualizer that shows:
      - Multi-view cameras (frames + frustums)
      - An object mesh following a per-frame SE(3) trajectory
      - 3D hand keypoints + bone segments per frame

    Expects:
      cam_params: {cam_id: {"extrinsic": (3x4|4x4 world->cam), "intrinsic": (3x3)}}
      hand_keypoint_dict: {frame_idx: (21, 3)} in WORLD coord
      obj_trajectory_dict: {frame_idx: (4x4) or {"T": (4x4)}} in WORLD coord
    """

    def __init__(
        self,
        port: int = 11375,
        up_direction: str = "+y",
        render_path: str = "./render_keypoint_obj",
        dark_mode: bool = False,
    ) -> None:
        self.server = viser.ViserServer(port=port)
        self.server.gui.configure_theme(dark_mode=dark_mode)
        self.server.scene.set_up_direction(up_direction)
        self.render_path = render_path
        os.makedirs(self.render_path, exist_ok=True)

        # Playback state
        self.gui_timestep = None
        self.gui_playing = None
        self.framerate = 10
        self.num_frames = 0

        # Handles
        self.frame_nodes: Dict[int, viser.FrameHandle] = {}
        self.frustum_handles: Dict[str, viser.CameraFrustumHandle] = {}

        # Optional per-camera images for frustums
        self._cam_imgs: Optional[Dict[str, List[np.ndarray]]] = None
        self._cam_img_down = 4

        # Root frames
        self.server.scene.add_frame("/global", show_axes=False)
        self.server.scene.add_frame("/frames", show_axes=False)

        # Default lights
        self.server.scene.enable_default_lights(True)

        @self.server.on_client_connect
        def _on_connect(client: viser.ClientHandle) -> None:
            client.camera.look_speed = 0.2
            client.camera.move_speed = 0.05
            client.camera.far = 20.0

            near_slider = client.gui.add_slider("Near", min=0.01, max=10.0, step=0.001, initial_value=client.camera.near)
            far_slider = client.gui.add_slider("Far", min=1, max=50.0, step=0.001, initial_value=client.camera.far)

            @near_slider.on_update
            def _(_):
                client.camera.near = near_slider.value

            @far_slider.on_update
            def _(_):
                client.camera.far = far_slider.value

    # ------------------------------ Cameras ------------------------------
    def draw_cameras(
        self,
        cam_params: Dict[str, Dict[str, np.ndarray]],
        width: int,
        height: int,
        cam_imgs: Optional[Dict[str, List[np.ndarray]]] = None,
        downsample_factor: int = 4,
    ) -> None:
        self._cam_imgs = cam_imgs
        self._cam_img_down = max(1, int(downsample_factor))

        self.server.scene.add_frame("/frames/cameras", show_axes=False)

        # Place each camera frame + frustum
        for cam_id, cp in cam_params.items():
            E_wc = np.asarray(cp["extrinsic"])  # world->camera
            K = np.asarray(cp["intrinsic"])     # 3x3
            T_c2w = _world_from_extrinsic(E_wc)

            frame = self.server.scene.add_frame(
                f"/frames/cameras/{cam_id}",
                wxyz=T_c2w.rotation().wxyz,
                position=T_c2w.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )

            # Prepare image (optional)
            if cam_imgs is not None and cam_id in cam_imgs and len(cam_imgs[cam_id]) > 0:
                idx0 = 0
                img = cam_imgs[cam_id][idx0]
                img_small = img[::self._cam_img_down, ::self._cam_img_down]
            else:
                img_small = np.ones((height // self._cam_img_down, width // self._cam_img_down, 3), dtype=np.uint8) * 255

            fy = float(K[1, 1])
            fov_y = 2.0 * np.arctan2(height / 2.0, fy)

            frustum = self.server.scene.add_camera_frustum(
                f"/frames/cameras/{cam_id}/frustum",
                fov=fov_y,
                aspect=width / float(height),
                scale=0.05,
                image=img_small,
            )
            self.frustum_handles[cam_id] = frustum

            @frustum.on_click
            def _(_, _frame=frame) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = _frame.wxyz
                    client.camera.position = _frame.position

        # Small UI: toggle all camera frustums
        with self.server.gui.add_folder("Cameras", expand_by_default=False):
            toggle = self.server.gui.add_checkbox("Show Frustums", True)

            @toggle.on_update
            def _(_):
                for fr in self.frustum_handles.values():
                    fr.visible = toggle.value

    def add_camera_plane_from_ids(
        self,
        cam_params: dict,
        target_ids: List[str],
        drop: float = 0.18,
        size: float = 10.0,
        step: float = 0.25,
        name: str = "/global/camera_fit_plane",
    ):
        """
        target_ids 카메라 센터들을 평면에 맞추고, 법선 방향으로 drop만큼 내린 위치에
        체커보드 평면을 생성하여 씬에 추가합니다.
        """
        centers = []
        for cam_id in target_ids:
            if cam_id in cam_params:
                extr = cam_params[cam_id]["extrinsic"]
                # extr shape (3x4) 또는 (4x4) 모두 지원
                if extr.shape == (4, 4):
                    R = extr[:3, :3]
                    t = extr[:3, 3]
                else:
                    R = extr[:, :3]
                    t = extr[:, 3]
                # world 좌표계의 카메라 센터
                C = -R.T @ t
                centers.append(C)

        if len(centers) < 3:
            print("⚠️ Not enough cameras to fit a plane.")
            return None

        centers = np.asarray(centers)
        # SVD로 평면 피팅
        centroid = centers.mean(axis=0)
        U, S, Vt = np.linalg.svd(centers - centroid)
        normal = Vt[-1]  # 최소분산 축

        # 카메라를 내려다보는 바닥쪽으로 drop 만큼 이동
        offset_center = centroid - drop * normal

        # 체커보드 메시(trimesh) 만들고 Viser에 추가
        plane_mesh = make_checkerboard_plane_trimesh(
            center=offset_center, normal=normal, size=size, step=step
        )

        # 기존 핸들이 있으면 교체
        if hasattr(self, "camera_plane_handle") and self.camera_plane_handle is not None:
            try:
                self.camera_plane_handle.remove()
            except Exception:
                pass

        self.camera_plane_handle = self.server.scene.add_mesh_trimesh(
            name=name,
            mesh=plane_mesh,
            position=(0.0, 0.0, 0.0),  # 이미 world 좌표로 생성했으므로 position은 0
        )

        # GUI 토글 (최초 1회만 생성)
        if not hasattr(self, "_camera_plane_toggle"):
            with self.server.gui.add_folder("Plane", expand_by_default=False):
                toggle = self.server.gui.add_checkbox("Show Camera Plane", True)

                @toggle.on_update
                def _(_ev) -> None:
                    if hasattr(self, "camera_plane_handle") and self.camera_plane_handle is not None:
                        self.camera_plane_handle.visible = toggle.value

            self._camera_plane_toggle = True

        return self.camera_plane_handle    

    # ------------------------------ Content ------------------------------
    def _ensure_playback(self, num_frames: int) -> None:
        self.num_frames = int(num_frames)
        with self.server.gui.add_folder("Playback"):
            gui_timestep = self.server.gui.add_slider(
                "Timestep", min=0, max=max(0, self.num_frames - 1), step=1, initial_value=0
            )
            gui_next = self.server.gui.add_button("Next Frame")
            gui_prev = self.server.gui.add_button("Prev Frame")
            gui_fps = self.server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=self.framerate)
            gui_play = self.server.gui.add_checkbox("Playing", True)
            render_png = self.server.gui.add_checkbox("Render to PNG", False)

            self.gui_timestep = gui_timestep
            self.gui_playing = gui_play

            @gui_fps.on_update
            def _(_):
                self.framerate = int(gui_fps.value)

            @gui_next.on_click
            def _(_):
                gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

            @gui_prev.on_click
            def _(_):
                gui_timestep.value = (gui_timestep.value - 1) % self.num_frames

            @render_png.on_update
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                if client is None:
                    return
                img = client.get_render(height=1440, width=2560)  # 2x 720p
                cv2.imwrite(os.path.join(self.render_path, f"{int(gui_timestep.value):05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Root for per-frame content
        self.server.scene.add_frame("/frames/t", show_axes=False)

    def add_hand_and_object(
        self,
        obj_name: Optional[str],
        hand_keypoint_dict: Dict[int, np.ndarray],  # (21, 3) world
        obj_trajectory_dict: Optional[Dict[int, np.ndarray]] = None,  # 4x4 or {"T":4x4}
        edge_list: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        frames = sorted(set(list(hand_keypoint_dict.keys()) + (list(obj_trajectory_dict.keys()) if obj_trajectory_dict else [])))
        if not frames:
            raise ValueError("No frames found in inputs.")
        self._ensure_playback(num_frames=len(frames))

        # Prepare object mesh once
        obj_mesh: Optional[trimesh.Trimesh] = None
        if obj_name is not None:
            if get_initial_mesh is None:
                raise RuntimeError("get_initial_mesh not available; run inside project or supply mesh manually.")
            obj_mesh, _ = get_initial_mesh(obj_name, return_type = "trimesh", simplify=True)
        # Build per-frame nodes
        for local_idx, fidx in enumerate(frames):
            node = self.server.scene.add_frame(f"/frames/t/{local_idx}", show_axes=False, visible=False)
            self.frame_nodes[local_idx] = node

            # 1) Object mesh (transformed)
            if obj_mesh is not None and obj_trajectory_dict is not None and fidx in obj_trajectory_dict:
                Ti = obj_trajectory_dict[fidx]
                if isinstance(Ti, dict) and "T" in Ti:
                    Ti = Ti["T"]
                Ti = _homogenize(np.asarray(Ti))
                mesh_t = copy.deepcopy(obj_mesh).apply_transform(Ti)
                self.server.scene.add_mesh_trimesh(name=f"/frames/t/{local_idx}/object", mesh=mesh_t)

            # 2) Hand keypoints + bones
            if fidx in hand_keypoint_dict:
                kp = np.asarray(hand_keypoint_dict[fidx], dtype=np.float32)  # (21, 3)
                # Points
                colors = np.tile(np.array([[0, 120, 255]], dtype=np.uint8), (kp.shape[0], 1))  # orange-ish
                self.server.scene.add_point_cloud(
                    f"/frames/t/{local_idx}/hand_points",
                    points=kp,
                    colors=colors,
                    point_shape="circle",
                    point_size=0.01,
                )
                # Lines
                edges = edge_list if edge_list is not None else MEDIA_EDGES
                segs = _edge_segments_from_keypoints(kp, edges)  # (E, 2, 3)
                if len(segs) > 0:
                    line_colors = np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (segs.shape[0], 2, 1))
                    self.server.scene.add_line_segments(
                        f"/frames/t/{local_idx}/hand_bones",
                        points=segs,
                        colors=line_colors,
                        line_width=2.0,
                    )

        # Initial visibility
        if 0 in self.frame_nodes:
            self.frame_nodes[0].visible = True

        # Hook up slider to toggle frames
        prev_local = 0

        @self.gui_timestep.on_update
        def _(_):
            nonlocal prev_local
            cur = int(self.gui_timestep.value)
            if prev_local == cur:
                return
            if prev_local in self.frame_nodes:
                self.frame_nodes[prev_local].visible = False
            if cur in self.frame_nodes:
                self.frame_nodes[cur].visible = True
            prev_local = cur

    # ------------------------------ Loop ------------------------------
    def spin(self) -> None:
        """Simple playback loop. Call after setup."""
        if self.gui_timestep is None:
            raise RuntimeError("Call add_hand_and_object(...) before spin().")
        while True:
            if self.gui_playing and self.gui_playing.value and self.num_frames > 0:
                self.gui_timestep.value = (int(self.gui_timestep.value) + self.framerate) % self.num_frames
            time.sleep(0.4)


# ------------------------------ Convenience API ------------------------------

def visualize_keypoint_object(
    obj_name: Optional[str],
    cam_params: Dict[str, Dict[str, np.ndarray]],
    hand_keypoint_dict: Dict[int, np.ndarray],  # frame -> (21,3) in world
    obj_trajectory_dict: Dict[int, np.ndarray],  # frame -> 4x4 or {"T":4x4}
    *,
    cam_imgs: Optional[Dict[str, List[np.ndarray]]] = None,
    cam_size: Tuple[int, int] = (2048, 1536),
    target_ids_for_plane: Optional[List[str]] = ["22684737","23022627","22645029","23173281","22641023","22641005"]
,
    port: int = 11375,
    up_direction: str = "+y",
    dark_mode: bool = False,
) -> None:
    """
    Drop-in function compatible with your existing import path.

    Example:
        visualize_keypoint_object(
            obj_name="pringles",
            cam_params=cam_params,
            hand_keypoint_dict=hand_kps,
            obj_trajectory_dict=obj_traj,
            cam_imgs=None,
            target_ids_for_plane=["22684737","23022627",...],
        )
    """
    viz = KeypointObjectCameraVisualizer(port=port, up_direction=up_direction, dark_mode=dark_mode)
    W, H = int(cam_size[0]), int(cam_size[1])
    viz.draw_cameras(cam_params, width=W, height=H, cam_imgs=cam_imgs, downsample_factor=12)
    if target_ids_for_plane is not None:
        viz.add_camera_plane_from_ids(cam_params, target_ids_for_plane, drop=0.18, size=10.0, step=0.25)
    viz.add_hand_and_object(obj_name=obj_name, hand_keypoint_dict=hand_keypoint_dict, obj_trajectory_dict=obj_trajectory_dict)

    
    viz.spin()
