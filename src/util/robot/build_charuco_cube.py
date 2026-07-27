"""Build a textured 0.3 m ChArUco cube mesh (OBJ+MTL+PNG) for the handeye target.

Four side faces carry ChArUco boards (keys 4-7 from the proposed config, 6x6 @ 50 mm
= exactly the 0.3 m face). Top + mount(bottom) faces are white. Faces are rendered by
OpenCV's own CharucoBoard.generateImage (detector-consistent) into a texture atlas;
the cube is a 24-vertex box with per-face UVs. Output goes to rsc/object/charuco_cube/
and the franka_adapter.urdf 'cube' link is repointed at it.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from cv2 import aruco
import trimesh
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "src/calibration/proposed_boards/charuco_info_new.json"
OUT = REPO / "rsc/object/charuco_cube"
OUT.mkdir(parents=True, exist_ok=True)

_DICT = {"6X6_250": aruco.DICT_6X6_250, "6X6_1000": aruco.DICT_6X6_1000}
SIDE = 0.3
FACE_PX = 300          # per-face texture resolution (rough is fine — it's just boards on a cube)
SIDE_BOARDS = ["4", "5", "6", "7"]   # config keys for the 4 charuco faces


def face_image(info):
    d = aruco.getPredefinedDictionary(_DICT[info["dict_type"]])
    b = aruco.CharucoBoard((info["numX"], info["numY"]),
                           info["checkerLength"], info["markerLength"],
                           d, np.array(info["markerIDs"]))
    b.setLegacyPattern(bool(info.get("setLegacyPattern")))
    m = FACE_PX // (info["numX"] * 3)          # small quiet-zone margin
    img = b.generateImage((FACE_PX, FACE_PX), marginSize=m, borderBits=1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def build():
    cfg = json.load(open(CONFIG))
    faces = [face_image(cfg[k]) for k in SIDE_BOARDS]
    white = np.full((FACE_PX, FACE_PX, 3), 255, np.uint8)
    # atlas: 3 cols x 2 rows -> [f4 f5 f6 / f7 white(top) white(mount)]
    cells = faces + [white, white]              # 6 cells
    rows = [np.hstack(cells[0:3]), np.hstack(cells[3:6])]
    atlas = np.vstack(rows)                      # (2*FACE_PX, 3*FACE_PX)
    Image.fromarray(atlas).save(OUT / "atlas.png")

    # atlas cell (col,row) -> uv rect; PIL origin top-left, trimesh uv origin bottom-left
    def cell_uv(col, row):
        u0, u1 = col / 3.0, (col + 1) / 3.0
        v_top, v_bot = 1.0 - row / 2.0, 1.0 - (row + 1) / 2.0
        # corners in order matching face corner order (bl, br, tr, tl)
        return np.array([[u0, v_bot], [u1, v_bot], [u1, v_top], [u0, v_top]])

    h = SIDE / 2
    # face: (4 corners CCW outward, atlas cell). corner order = bl,br,tr,tl in face's uv sense
    face_def = [
        # +X (charuco f4)
        (np.array([[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]]), (0, 0)),
        # -X (f5)
        (np.array([[-h, h, -h], [-h, -h, -h], [-h, -h, h], [-h, h, h]]), (1, 0)),
        # +Y (f6)
        (np.array([[h, h, -h], [-h, h, -h], [-h, h, h], [h, h, h]]), (2, 0)),
        # -Y (f7)
        (np.array([[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]]), (0, 1)),
        # +Z (top, white)
        (np.array([[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]]), (1, 1)),
        # -Z (mount, white)
        (np.array([[-h, h, -h], [h, h, -h], [h, -h, -h], [-h, -h, -h]]), (2, 1)),
    ]

    V, F, UV = [], [], []
    for corners, (col, row) in face_def:
        base = len(V)
        V.extend(corners)
        UV.extend(cell_uv(col, row))
        F.append([base, base + 1, base + 2])
        F.append([base, base + 2, base + 3])
    V = np.array(V)
    F = np.array(F)
    UV = np.array(UV)

    tex = Image.open(OUT / "atlas.png")
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=UV, image=tex))
    mesh.export(OUT / "charuco_cube.obj")
    print(f"wrote {OUT/'charuco_cube.obj'} (+ .mtl, atlas.png)")
    print(f"  verts {len(V)}, faces {len(F)}, uv in [{UV.min():.2f},{UV.max():.2f}], extents {mesh.extents.round(3)}")
    return mesh


def repoint_urdf():
    f = REPO / "rsc/robot/franka/franka_adapter.urdf"
    txt = f.read_text()
    old = '''      <geometry>
        <box size="0.3 0.3 0.3"/>
      </geometry>'''
    new = '''      <geometry>
        <mesh filename="../../object/charuco_cube/charuco_cube.obj"/>
      </geometry>'''
    if old in txt:
        f.write_text(txt.replace(old, new))
        print("URDF cube link -> charuco_cube.obj")
    else:
        print("URDF cube box geometry not found (already repointed?)")


if __name__ == "__main__":
    build()
    repoint_urdf()
