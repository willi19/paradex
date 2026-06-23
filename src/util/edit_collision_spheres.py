"""
Curobo collision-sphere editor (viser) for allegro_right_A.urdf.

- Drag spheres with the gizmo to position them inside link frames.
- Adjust the selected sphere's radius with the slider in "Selected sphere".
- Save → curobo-format YAML keyed by link.

Usage:
    python src/util/edit_collision_spheres.py
    python src/util/edit_collision_spheres.py --load existing.yml
"""
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
import yaml
import yourdfpy

DEFAULT_URDF = "rsc/robot/allegro_v5/allegro_right_A.urdf"
DEFAULT_OUTPUT = "rsc/robot/spheres/allegro_v5_right.yml"
DEFAULT_RADIUS = 0.012

# Pairs of links whose collision should be ignored. Stored one-way per the
# convention used by existing curobo yamls — the engine treats the pair as
# unordered. Edit here if the kinematics change.
SELF_COLLISION_IGNORE_PAIRS = [
    ("palm_link",  "link_1_0"),
    ("palm_link",  "link_5_0"),
    ("palm_link",  "link_9_0"),
    ("palm_link",  "link_12_0"),
    ("link_12_0",  "link_14_0"),
    ("link_14_0",  "link_15_0"),
    ("link_15_0",  "link_15_0_tip"),
    ("link_1_0",   "link_2_0"),
    ("link_2_0",   "link_3_0_tip"),
    ("link_5_0",   "link_6_0"),
    ("link_6_0",   "link_7_0_tip"),
    ("link_9_0",   "link_10_0"),
    ("link_10_0",  "link_11_0_tip"),
]


def build_ignore_map(pairs):
    out: Dict[str, List[str]] = {}
    for a, b in pairs:
        out.setdefault(a, []).append(b)
    return out

MESH_COLOR = (200, 200, 210, 90)
LINK_HIGHLIGHT_COLOR = (255, 200, 60, 220)
SPHERE_COLOR = (255, 80, 80, 200)
SPHERE_SELECTED_COLOR = (60, 200, 255, 220)


def _icosphere(radius: float, color) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    m.visual.face_colors = list(color)
    return m


@dataclass(eq=False)
class Sphere:
    link: str
    center: np.ndarray
    radius: float
    mesh_node: Optional[viser.MeshHandle] = None
    gizmo: Optional[viser.TransformControlsHandle] = None
    list_folder: Optional[viser.GuiFolderHandle] = None


class SphereEditor:
    def __init__(self, server: viser.ViserServer, urdf_path: str, output_path: str):
        self.server = server
        self.output_path = output_path

        self.urdf = yourdfpy.URDF.load(urdf_path)
        self.urdf.update_cfg(np.zeros(len(self.urdf.actuated_joint_names)))
        self.link_T: Dict[str, np.ndarray] = {
            ln: np.asarray(self.urdf.get_transform(ln, self.urdf.base_link))
            for ln in self.urdf.link_map
        }

        self.spheres: List[Sphere] = []
        self.selected: Optional[Sphere] = None
        self._link_mesh_records: Dict[str, List[dict]] = {}

        self._build_scene()
        self._build_gui()

    # ---------------------------------------------------------------- scene
    def _build_scene(self):
        for link_name, T in self.link_T.items():
            self.server.scene.add_frame(
                f"/robot/{link_name}", show_axes=False,
                position=tuple(T[:3, 3]),
                wxyz=tuple(vtf.SO3.from_matrix(T[:3, :3]).wxyz),
            )
        scene = self.urdf.scene
        parents = scene.graph.transforms.parents
        for gname, mesh in scene.geometry.items():
            link_name = parents.get(gname)
            if link_name not in self.link_T:
                continue
            T_world = np.asarray(scene.graph.get(gname)[0])
            T_local = np.linalg.inv(self.link_T[link_name]) @ T_world
            self._link_mesh_records.setdefault(link_name, []).append({
                "name": f"/robot/{link_name}/__mesh__{gname}",
                "mesh": mesh,
                "pos": tuple(T_local[:3, 3]),
                "wxyz": tuple(vtf.SO3.from_matrix(T_local[:3, :3]).wxyz),
                "handle": None,
            })

    def _render_links(self, highlight: Optional[str]):
        for link_name, records in self._link_mesh_records.items():
            color = LINK_HIGHLIGHT_COLOR if link_name == highlight else MESH_COLOR
            for rec in records:
                if rec["handle"] is not None:
                    rec["handle"].remove()
                m = rec["mesh"].copy()
                m.visual.face_colors = list(color)
                rec["handle"] = self.server.scene.add_mesh_trimesh(
                    rec["name"], mesh=m, position=rec["pos"], wxyz=rec["wxyz"],
                )

    # ---------------------------------------------------------------- gui
    def _build_gui(self):
        gui = self.server.gui
        gui.set_panel_label("Curobo Sphere Editor")

        links = sorted(self.link_T.keys())
        with gui.add_folder("Workflow"):
            self.link_dd = gui.add_dropdown("Link", options=links, initial_value=links[0])
            self.link_dd.on_update(lambda _: self._render_links(self.link_dd.value))
            self.default_radius = gui.add_number(
                "Default radius (m)", initial_value=DEFAULT_RADIUS,
                min=0.001, max=0.2, step=0.001,
            )
            add_btn = gui.add_button("➕ Add sphere to selected link")
            add_btn.on_click(lambda _: self.add_sphere(self.link_dd.value))
            clear_btn = gui.add_button("🗑 Clear ALL spheres")
            clear_btn.on_click(lambda _: self.clear_all())

        with gui.add_folder("Selected sphere"):
            self.sel_info = gui.add_markdown("*(none)*")
            self.sel_radius = gui.add_slider(
                "radius (m)", min=0.002, max=0.08, step=0.001,
                initial_value=DEFAULT_RADIUS, disabled=True,
            )
            self.sel_radius.on_update(lambda _: self._on_radius_slider())
            sel_del = gui.add_button("❌ delete selected")
            sel_del.on_click(lambda _: self._delete_selected())

        with gui.add_folder("I/O"):
            self.out_input = gui.add_text("Output path", initial_value=self.output_path)
            save_btn = gui.add_button("💾 Save YAML")
            save_btn.on_click(lambda _: self.save())
            self.load_input = gui.add_text("Load YAML path", initial_value="")
            load_btn = gui.add_button("📂 Load")
            load_btn.on_click(lambda _: self.load(self.load_input.value))

        self.info_md = gui.add_markdown("0 spheres")
        self.list_root = gui.add_folder("Spheres")
        self._render_links(links[0])
        self._refresh_info()

    def _refresh_info(self):
        n = len(self.spheres)
        nl = len(set(s.link for s in self.spheres))
        self.info_md.content = f"**{n}** spheres across **{nl}** link(s)"

    def _refresh_sel_panel(self):
        if self.selected is None:
            self.sel_info.content = "*(none)*"
            self.sel_radius.disabled = True
            return
        s = self.selected
        self.sel_info.content = (
            f"link: `{s.link}`<br>"
            f"center: {np.round(s.center, 4).tolist()}"
        )
        self.sel_radius.disabled = False
        if self.sel_radius.value != s.radius:
            self.sel_radius.value = s.radius

    # ---------------------------------------------------------------- ops
    def add_sphere(self, link: str, center=(0.0, 0.0, 0.0), radius: Optional[float] = None):
        r = float(radius if radius is not None else self.default_radius.value)
        s = Sphere(link=link, center=np.asarray(center, dtype=float), radius=r)
        self.spheres.append(s)
        self._spawn_3d(s, color=SPHERE_COLOR)
        self._spawn_list_entry(s)
        self.select(s)
        self._refresh_info()

    def _spawn_3d(self, s: Sphere, color):
        s.mesh_node = self.server.scene.add_mesh_trimesh(
            f"/robot/{s.link}/sphere_{id(s)}",
            mesh=_icosphere(s.radius, color),
            position=tuple(s.center), wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        s.gizmo = self.server.scene.add_transform_controls(
            f"/robot/{s.link}/gizmo_{id(s)}", scale=0.03,
            disable_rotations=True, depth_test=False,
            position=tuple(s.center), wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=False,
        )
        s.gizmo.on_update(lambda _: self._on_gizmo_move(s))

    def _redraw_sphere(self, s: Sphere, color):
        if s.mesh_node is not None:
            s.mesh_node.remove()
        s.mesh_node = self.server.scene.add_mesh_trimesh(
            f"/robot/{s.link}/sphere_{id(s)}",
            mesh=_icosphere(s.radius, color),
            position=tuple(s.center), wxyz=(1.0, 0.0, 0.0, 0.0),
        )

    def _on_gizmo_move(self, s: Sphere):
        s.center = np.asarray(s.gizmo.position, dtype=float)
        if s.mesh_node is not None:
            s.mesh_node.position = tuple(s.center)
        if self.selected is s:
            self._refresh_sel_panel()

    def _spawn_list_entry(self, s: Sphere):
        with self.list_root:
            s.list_folder = self.server.gui.add_folder(
                f"{s.link}  r={s.radius:.3f}"
            )
        with s.list_folder:
            sel_btn = self.server.gui.add_button("👁 select")
            sel_btn.on_click(lambda _: self.select(s))
            del_btn = self.server.gui.add_button("❌ delete")
            del_btn.on_click(lambda _: self.remove_sphere(s))

    def remove_sphere(self, s: Sphere):
        if s not in self.spheres:
            return
        if s.mesh_node is not None: s.mesh_node.remove()
        if s.gizmo is not None: s.gizmo.remove()
        if s.list_folder is not None: s.list_folder.remove()
        self.spheres.remove(s)
        if self.selected is s:
            self.selected = None
        self._refresh_info()
        self._refresh_sel_panel()

    def clear_all(self):
        for s in list(self.spheres):
            self.remove_sphere(s)

    def select(self, s: Sphere):
        prev = self.selected
        if prev is not None and prev is not s and prev in self.spheres:
            self._redraw_sphere(prev, SPHERE_COLOR)
            if prev.gizmo is not None:
                prev.gizmo.visible = False
        self.selected = s
        self._redraw_sphere(s, SPHERE_SELECTED_COLOR)
        if s.gizmo is not None:
            s.gizmo.visible = True
        self._refresh_sel_panel()

    def _on_radius_slider(self):
        if self.selected is None:
            return
        new_r = float(self.sel_radius.value)
        if new_r == self.selected.radius:
            return
        self.selected.radius = new_r
        self._redraw_sphere(self.selected, SPHERE_SELECTED_COLOR)
        if self.selected.list_folder is not None:
            self.selected.list_folder.label = f"{self.selected.link}  r={new_r:.3f}"

    def _delete_selected(self):
        if self.selected is not None:
            self.remove_sphere(self.selected)

    # ---------------------------------------------------------------- io
    def save(self):
        by_link: Dict[str, list] = {}
        for s in self.spheres:
            by_link.setdefault(s.link, []).append({
                "center": [float(x) for x in s.center],
                "radius": float(s.radius),
            })
        payload = {
            "collision_spheres": by_link,
            "self_collision_ignore": build_ignore_map(SELF_COLLISION_IGNORE_PAIRS),
        }
        out = self.out_input.value
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(payload, f, default_flow_style=None, sort_keys=False)
        print(f"[saved] {out} — {len(self.spheres)} spheres, "
              f"{len(by_link)} link(s), "
              f"{len(SELF_COLLISION_IGNORE_PAIRS)} ignore pair(s)")

    def load(self, path: str):
        if not path or not os.path.exists(path):
            print(f"[load] not found: {path!r}")
            return
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        self.clear_all()
        for link, items in (data.get("collision_spheres") or {}).items():
            if link not in self.link_T:
                print(f"[load] skip unknown link: {link}")
                continue
            for item in items or []:
                self.add_sphere(link,
                                center=item.get("center", [0, 0, 0]),
                                radius=item.get("radius", DEFAULT_RADIUS))
        print(f"[loaded] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--load", default=None, help="optional existing YAML to start from")
    args = parser.parse_args()

    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    editor = SphereEditor(server, args.urdf, args.out)
    if args.load:
        editor.load(args.load)

    print("Editor ready. Open the URL above and Ctrl+C to quit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
