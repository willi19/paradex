"""3D-print parts to build the ChArUco cube: 4 side boards (a square tube), open bottom,
and a robot mount on top (robot-side). Boards are GLUED to flat pads (3 mm boards, no screws).

Parts:
  corner_bracket.stl  x8  - L bracket (2 perpendicular pads) joining two adjacent side boards
                            at 90 deg, at the 8 vertical-edge corners (4 top + 4 bottom).
  robot_mount.stl     x1  - top cross: central plate (robot-adapter interface) + 4 arms out to
                            the 4 top corners, tying the tube together and mounting it to the robot.

Cube edge 320 mm (board incl. 10 mm margin). Prints under a ~250 mm bed except robot_mount arms
(split at the marked plane or print the hub + 4 arms separately).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pathlib import Path
import numpy as np
import trimesh

CUBE = 320.0          # cube edge (mm)
PAD = 90.0            # glue pad reach from the corner (mm) -- bigger = more board contact
THICK = 5.0           # pad / arm thickness (mm)
LIP = 4.0             # seating lip height (board 3 mm + a bit)
WALL_H = 320.0        # side board height (= cube edge)

# robot mount
HUB = 100.0           # central plate (mm) = Board_base top plate footprint (stacks on it)
HUB_HOLE = 4.5        # M4 clearance for the robot-adapter bolts
ARM_W = 40.0          # arm width (mm)

# hub<->leg tongue-in-groove interlock (the hub ear is a flat TONGUE that slides into a
# SLOT in the leg; one vertical bolt goes through leg-top + tongue + leg-bottom so the holes
# can never be off). bolts only clamp; the slot carries the load & sets the position.
TONGUE_W = 30.0       # hub tongue width (mm), < ARM_W so it seats inside the leg slot walls
SOCK_LEN = 44.0       # leg socket / tongue overlap length along the diagonal (mm)
WALL = 5.0            # slot wall thickness above/below the tongue (mm)
CLR = 0.5             # slot clearance per side (mm)
# corner-bracket <-> leg tongue-in-groove (a flat tongue off the bracket's BACK pad slides into
# a slot at the leg's outer end; both lie in the back-face plane on the corner->center diagonal).
OTB = 40.0            # bracket tongue length from the corner (mm)
OTW = 22.0            # bracket tongue width (mm)

OUT = Path(__file__).resolve().parents[3] / "rsc/object/charuco_cube/support"


def box(dx, dy, dz, tx=0, ty=0, tz=0):
    b = trimesh.creation.box((dx, dy, dz)); b.apply_translation((dx/2+tx, dy/2+ty, dz/2+tz)); return b


def boxc(dx, dy, dz, cx, cy, cz):
    b = trimesh.creation.box((dx, dy, dz)); b.apply_translation((cx, cy, cz)); return b


def corner_bracket():
    # 3 perpendicular glue pads (the 3 cube faces meeting at a corner) -> more stable than 2.
    # corner at origin, faces are x=0, y=0, z=0; the two side boards glue to two of them, the
    # third pad braces the corner (and ties to the robot mount / base). + edge lips on all 3 edges.
    parts = [
        box(THICK, PAD, PAD),                        # pad on x=0 face
        box(PAD, THICK, PAD),                        # pad on y=0 face
        box(PAD, PAD, THICK),                        # pad on z=0 face
        box(THICK + LIP, THICK + LIP, PAD),          # lip along z edge
        box(THICK + LIP, PAD, THICK + LIP),          # lip along y edge
        box(PAD, THICK + LIP, THICK + LIP),          # lip along x edge
    ]
    # NOTE: no tongue here -- the leg's forked end wraps this bracket's y=0 (back) pad from both
    # faces (see mount_leg's outer fork), so the pad itself is what gets clamped.
    return trimesh.boolean.union(parts)


# --- robot mount: MODULAR (hub + 4 legs) so each part fits a ~250 mm bed ---
# The one-piece cross spanned ~350 mm (too big). Now: a small central HUB (robot-adapter
# interface) whose 4 diagonal ears are flat TONGUES, + 4 identical LEGS. Each leg has a
# blind SLOT that the hub tongue slides into (tongue-in-groove -> sets XY + rotation), then
# runs out to a corner and ends in downward L-tabs that grab the two adjacent wall tops.
# One vertical bolt per leg pins leg-top + tongue + leg-bottom together (holes self-align).
HR = 70.0                                    # hub half-diagonal: DIAMOND corners sit at (+-HR,0)/(0,+-HR)
HUB_DIAG = np.hypot(HUB / 2, HUB / 2)       # (legacy) hub corner distance from center
CORNER_DIAG = np.hypot(CUBE / 2, CUBE / 2)  # top corner distance from center
SOCK0 = HUB_DIAG - 6                         # inner edge of the leg socket (diagonal radius)
SOCK1 = SOCK0 + SOCK_LEN                     # outer edge / blind end of the slot
RB = (SOCK0 + SOCK1) / 2                     # bolt radius (through the overlap)
BLOCK_H = THICK + 2 * CLR + 2 * WALL         # leg socket block height
DIAGS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _rotz(ang):
    return trimesh.transformations.rotation_matrix(ang, [0, 0, 1])


def _diag_bar(r0, r1, ang, w, z0=0.0, h=THICK):
    """Box of width w, height h (bottom at z0), spanning radius r0..r1 along direction `ang`."""
    b = box(r1 - r0, w, h, r0, -w / 2, z0)
    b.apply_transform(_rotz(ang))
    return b


def _vbolt(ang, r=RB):
    """Vertical bolt hole through a tongue/slot overlap at diagonal radius r (matches both parts)."""
    c = trimesh.creation.cylinder(radius=HUB_HOLE / 2, height=4 * BLOCK_H, sections=24)
    c.apply_translation((r * np.cos(ang), r * np.sin(ang), 0))
    return c


def tri_prism(a, b, c, h):
    """Triangular prism from 2D pts a,b,c extruded z 0..h (watertight)."""
    V = np.array([[*a, 0], [*b, 0], [*c, 0], [*a, h], [*b, h], [*c, h]], float)
    F = [[0, 2, 1], [3, 4, 5],                       # bottom, top
         [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]
    m = trimesh.Trimesh(vertices=V, faces=F, process=True)
    m.fix_normals()
    return m


def mount_hub_quarter(sx=1, sy=1):
    """One QUARTER of the DIAMOND hub (print x4): a right triangle (center + two axis corners),
    so the hub square is 45deg to the legs and each tongue leaves a flat diamond EDGE (not a
    pointy corner). The 4 quarters butt along the x/y axes; the robot adapter's 4 bolts clamp
    them. Each quarter keys to its leg via the tongue-in-groove + vertical bolt."""
    ang = np.arctan2(sy, sx)
    parts = [
        tri_prism((0, 0), (sx * HR, 0), (0, sy * HR), THICK),   # diamond quadrant
        _diag_bar(HR / np.sqrt(2) - 6, SOCK1 - 4, ang, TONGUE_W),  # tongue off the diamond edge
    ]
    holes = [_vbolt(ang)]                                        # leg bolt
    c = trimesh.creation.cylinder(radius=HUB_HOLE / 2, height=4 * THICK, sections=24)
    c.apply_translation((sx * HR * 0.4, sy * HR * 0.4, THICK / 2))  # this quarter's robot bolt
    holes.append(c)
    return trimesh.boolean.difference([trimesh.boolean.union(parts), *holes])


def mount_hub():
    """Full hub = the 4 quarters butted together (for a combined preview / STL)."""
    return trimesh.boolean.union([mount_hub_quarter(sx, sy) for sx, sy in DIAGS])


def mount_leg(sx=1, sy=1):
    """One leg for corner (sx,sy): slotted socket (hub tongue slides in) -> bar -> L-tabs."""
    ang = np.arctan2(sy, sx)
    cx, cy = sx * CUBE / 2, sy * CUBE / 2
    block = _diag_bar(SOCK0, SOCK1, ang, ARM_W, z0=-WALL - CLR, h=BLOCK_H)    # inner socket block (hub)
    oblock = _diag_bar(CORNER_DIAG - OTB - 8, CORNER_DIAG, ang, ARM_W,        # outer socket block (bracket)
                       z0=-WALL - CLR, h=BLOCK_H)
    parts = [
        block, oblock,
        _diag_bar(SOCK1, CORNER_DIAG - OTB - 8, ang, ARM_W, z0=0, h=THICK),  # bar between the two sockets
        boxc(THICK, PAD, PAD, cx - sx * THICK / 2, cy - sy * PAD / 2, -PAD / 2),   # tab on one wall
        boxc(PAD, THICK, PAD, cx - sx * PAD / 2, cy - sy * THICK / 2, -PAD / 2),   # tab on the other
    ]
    # inner blind slot: hub tongue enters from inboard, closed near SOCK1
    islot = _diag_bar(SOCK0 - 14, SOCK1 - 4, ang, TONGUE_W + 2 * CLR, z0=-CLR, h=THICK + 2 * CLR)
    # outer FORK: full-width gap so the two prongs wrap the ONE back pad (its 2 faces); open at corner
    oslot = _diag_bar(CORNER_DIAG - OTB, CORNER_DIAG + 20, ang, ARM_W + 2, z0=-CLR, h=THICK + 2 * CLR)
    return trimesh.boolean.difference(
        [trimesh.boolean.union(parts), islot, oslot, _vbolt(ang), _vbolt(ang, CORNER_DIAG - OTB / 2)])


def mount_assembly():
    """Hub + 4 legs unioned (for a combined preview / STL)."""
    return trimesh.boolean.union([mount_hub()] + [mount_leg(sx, sy) for sx, sy in DIAGS])


def preview(mesh, png, title):
    fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolor="#5aa0e0",
                                         edgecolor="k", linewidths=0.05, alpha=0.9))
    b = mesh.bounds
    for lim, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        lim(b[0, i], b[1, i])
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=30, azim=45); ax.set_title(title)
    fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, mesh, title in [("corner_bracket", corner_bracket(), "corner bracket x8 (glue 2 walls, 90deg)"),
                              ("mount_hub_quarter", mount_hub_quarter(), "hub QUARTER (x4, tongue)"),
                              ("mount_leg", mount_leg(), "robot mount LEG (slot + bolt, x4)"),
                              ("robot_mount", mount_assembly(), "robot mount assembled (4 quarters + 4 legs)")]:
        mesh.export(OUT / f"{name}.stl")
        preview(mesh, OUT / f"{name}_preview.png", title)
        print(f"{name}: bbox {mesh.extents.round(1)} mm | watertight {mesh.is_watertight} "
              f"| volume {mesh.volume/1000:.1f} cm^3 -> {OUT/(name+'.stl')}")


if __name__ == "__main__":
    main()
