"""View the ChArUco-cube support assembly in 3D (Viser): the 3D-print joint parts
(corner brackets + robot mount) with the 4 ChArUco boards glued on, so you can see
HOW a board seats on the corner brackets.

Reuses corner_bracket() / robot_mount() from build_cube_frame.py, places the 8 corner
brackets (into-the-cube octants) at the cube corners, lays the 4 boards on the
TOP / FRONT / LEFT / RIGHT faces (BACK = robot mount, BOTTOM = open), and mounts the
robot mount on the BACK face. Boards are semi-transparent so the pads show through.

    python src/util/robot/view_cube_support.py            # opens http://localhost:8080
    python src/util/robot/view_cube_support.py --no-view  # just build + print bounds
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO))

_bcf = importlib.util.spec_from_file_location("bcf", Path(__file__).parent / "build_cube_frame.py")
bcf = importlib.util.module_from_spec(_bcf); _bcf.loader.exec_module(bcf)

C = bcf.CUBE          # cube edge (mm)
T = bcf.THICK
BOARD_T = 3.0         # board thickness (mm)
BOARD = C - 4.0       # board footprint on a face (mm)
MM = 0.001            # mm -> m for viser


def reflect_translate(sx, sy, sz, tx, ty, tz):
    M = np.diag([sx, sy, sz, 1.0]).astype(float)
    M[:3, 3] = (tx, ty, tz)
    return M


def board_box(dx, dy, dz, cx, cy, cz, rgba):
    b = trimesh.creation.box((dx, dy, dz))
    b.apply_translation((cx, cy, cz))
    b.visual.vertex_colors = rgba
    return b


def build_parts():
    """Return [(name, mesh_in_meters, is_board), ...] for the whole assembly."""
    parts = []
    bracket = bcf.corner_bracket()
    bracket.visual.vertex_colors = [90, 160, 224, 255]      # blue
    # 8 corners: pads must point INTO the cube, so reflect each axis at the far face.
    for cx in (0, C):
        for cy in (0, C):
            for cz in (0, C):
                sx, sy, sz = (1 if cx == 0 else -1), (1 if cy == 0 else -1), (1 if cz == 0 else -1)
                m = bracket.copy()
                m.apply_transform(reflect_translate(sx, sy, sz, cx, cy, cz))
                parts.append((f"bracket_{cx}_{cy}_{cz}", m, False))

    # robot mount on the BACK face (y = C): modular hub + 4 legs (each its own print).
    # built in the mount's xy-plane, then rotated so plate normal -> +y, tabs -> -y (into cube).
    Rx = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    onto_back = np.eye(4)
    onto_back[:3, :3] = Rx[:3, :3]
    onto_back[:3, 3] = (C / 2, C - T, C / 2)
    for sx, sy in bcf.DIAGS:
        q = bcf.mount_hub_quarter(sx, sy); q.visual.vertex_colors = [120, 200, 120, 255]  # green
        q.apply_transform(onto_back)
        parts.append((f"hub_q_{sx}_{sy}", q, False))
    for sx, sy in bcf.DIAGS:
        leg = bcf.mount_leg(sx, sy); leg.visual.vertex_colors = [90, 200, 150, 255]  # teal-green
        leg.apply_transform(onto_back)
        parts.append((f"mount_leg_{sx}_{sy}", leg, False))

    # 4 boards, glued flat on the outer face surface (semi-transparent)
    face_rgba = [235, 235, 235, 120]
    t = BOARD_T
    parts.append(("board_TOP",   board_box(BOARD, BOARD, t, C / 2, C / 2, C + t / 2, face_rgba), True))
    parts.append(("board_FRONT", board_box(BOARD, t, BOARD, C / 2, -t / 2, C / 2, face_rgba), True))
    parts.append(("board_LEFT",  board_box(t, BOARD, BOARD, -t / 2, C / 2, C / 2, face_rgba), True))
    parts.append(("board_RIGHT", board_box(t, BOARD, BOARD, C + t / 2, C / 2, C / 2, face_rgba), True))

    # scale mm -> m
    out = []
    for name, m, is_board in parts:
        m = m.copy(); m.apply_scale(MM)
        out.append((name, m, is_board))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-view", action="store_true", help="just build + print, no viser")
    args = ap.parse_args()

    parts = build_parts()
    allv = np.vstack([m.vertices for _, m, _ in parts])
    print(f"cube edge {C:.0f} mm | {len(parts)} parts "
          f"({sum(not b for _, _, b in parts)} print parts + {sum(b for _, _, b in parts)} boards)")
    print(f"assembly bounds (m): {allv.min(0).round(3)} .. {allv.max(0).round(3)}")
    for name, m, _ in parts:
        print(f"  {name:22s} bbox {(m.extents).round(3)} m")
    if args.no_view:
        return

    from paradex.visualization.visualizer.viser import ViserViewer
    viewer = ViserViewer(port_number=args.port)
    viewer.add_grid(size=1.0, cell_size=0.05, height=0.0)
    for name, m, _ in parts:
        viewer.add_trimesh(name, m, np.eye(4))
    print(f"\nopen http://localhost:{args.port}  (blue = corner brackets, green = robot mount, "
          f"translucent = boards). Ctrl-C to quit.")
    viewer.start_viewer(use_thread=False)


if __name__ == "__main__":
    main()
