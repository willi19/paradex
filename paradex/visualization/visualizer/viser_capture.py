"""Temporary capture-specific Viser extensions.

Keep these overrides out of :mod:`viser` while the generic visualizer remains
backward compatible with its existing users.
"""

from __future__ import annotations

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from paradex.visualization.visualizer.viser import ViserViewer


class CaptureViserViewer(ViserViewer):
    """Bright, uncluttered viewer for reconstructed capture sequences."""

    def __init__(self, up_direction: np.ndarray, scene_title: str | None = None) -> None:
        super().__init__(up_direction=up_direction, scene_title=scene_title)

    def load_server(self) -> None:
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(dark_mode=False)
        self.server.scene.set_up_direction(self.up_direction)
        self.server.scene.set_background_image(np.full((2, 2, 3), 250, dtype=np.uint8))
        self.server.scene.world_axes

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            near_slider = client.gui.add_slider(
                "Near", min=0.01, max=10.0, step=0.001, initial_value=client.camera.near
            )
            far_slider = client.gui.add_slider(
                "Far", min=1, max=1000.0, step=0.001, initial_value=client.camera.far
            )

            @near_slider.on_update
            def _(_) -> None:
                client.camera.near = near_slider.value

            @far_slider.on_update
            def _(_) -> None:
                client.camera.far = far_slider.value

            self.load_saved_view(client, verbose=False)

    def add_lights(self) -> None:
        self.server.scene.enable_default_lights(True)

    def add_player(self) -> None:
        """Suppress the generic trajectory/video controls for capture playback.

        ``ViserViewer``'s Render Full Video button only knows about trajectories
        registered through ``add_traj``. Capture playback updates meshes directly,
        so its button would always report "No frames to render." The capture
        script creates its own visible Capture controls instead.
        """
        self.gui_playing = self.server.gui.add_checkbox("_Capture Internal Play", False, visible=False)

    def add_floor(
        self,
        height: float = 0.0,
        cell_color: tuple[int, int, int] = (200, 200, 200),
        section_color: tuple[int, int, int] = (128, 128, 0),
    ) -> None:
        self.floor_size = 1.0
        self.grid_handles = self.server.scene.add_grid(
            name="floor/grid",
            width=self.floor_size * 2,
            height=self.floor_size * 2,
            plane="xy",
            position=(0.0, 0.0, height),
            cell_size=0.1,
            cell_color=cell_color,
            section_color=section_color,
        )

    def update_floor(self) -> None:
        if hasattr(self, "floor_box_handle"):
            self.floor_box_handle.visible = self.floor_visible.value
        if hasattr(self, "grid_handles"):
            self.grid_handles.visible = self.grid_visible.value

    def add_object(
        self,
        name: str,
        obj: trimesh.Trimesh,
        obj_T: np.ndarray,
        opacity: float = 1.0,
        show_axes: bool = False,
        double_sided: bool = True,
    ) -> None:
        frame_handle = self.server.scene.add_frame(
            f"/objects/{name}_frame",
            position=obj_T[:3, 3],
            wxyz=R.from_matrix(obj_T[:3, :3]).as_quat()[[3, 0, 1, 2]],
            show_axes=show_axes,
            axes_length=0.05,
            axes_radius=0.002,
        )
        obj_name = f"/objects/{name}_frame/{name}"
        self._set_view_state_path_for_object(name)

        if double_sided or opacity < 0.999:
            color = np.array([200, 200, 200], dtype=np.uint8)
            vertex_colors = getattr(obj.visual, "vertex_colors", None)
            if vertex_colors is not None:
                colors = np.asarray(vertex_colors)
                if colors.ndim == 2 and colors.shape[0] > 0 and colors.shape[1] >= 3:
                    color = colors[:, :3].mean(axis=0).astype(np.uint8)
            mesh_handle = self.server.scene.add_mesh_simple(
                name=obj_name,
                vertices=obj.vertices,
                faces=obj.faces,
                color=color,
                side="double" if double_sided else "front",
            )
            if opacity < 0.999:
                mesh_handle.opacity = float(np.clip(opacity, 0.0, 1.0))
        else:
            # GLB preserves the orange OBJ's MTL texture, unlike simple meshes.
            mesh_handle = self.server.scene.add_mesh_trimesh(name=obj_name, mesh=obj)

        self.obj_dict[name] = {
            "mesh": obj,
            "transform": obj_T,
            "frame": frame_handle,
            "handle": mesh_handle,
        }
        self.frame_nodes[name] = frame_handle
