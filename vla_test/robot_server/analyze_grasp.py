"""Overlay where the gripper actually was onto what the policy saw.

For every query in a session log this projects the robot's own FK pose into the
frame that was sent to the policy at that instant, so gripper and object are
compared at the SAME moment — the only comparison that means anything. Frames
come from `robot_client.py --save-frames`.

The projection chain is `pixel = K @ dist( E @ C2R @ p_robot )`, verified on this
rig by projecting link6 and the TCP onto a live frame and checking they land on
the wrist flange and the palm. Note the direction: despite the name, the matrix
from load_current_C2R() maps ROBOT -> the calibration world frame that the
extrinsics are expressed in, so it is applied as `E @ C2R`, not `E @ inv(C2R)`.

A pixel offset converts to millimetres only near the plane the scale was taken
at; the mm/px printed here is computed by projecting a 50 mm robot-frame step at
the gripper's own height, so it is valid for objects at roughly that height and
degrades with depth difference. For an assumption-free number, triangulate the
object from two calibrated cameras instead.

Usage:
  python vla_test/robot_server/analyze_grasp.py ~/logs/deploy/<session>
  python vla_test/robot_server/analyze_grasp.py <session> --serial 26053260
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yourdfpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.calibration.utils import load_current_camparam, load_current_C2R
from paradex.vla.client import INSPIRE_LIMITS_RAD, fk_link6

TCP_OFFSET_M = 0.1352          # measured link6 -> TCP on this rig


def build_projector(serial):
    intr, extr = load_current_camparam()
    if serial not in intr:
        raise SystemExit(f"camera {serial} has no intrinsics in the current calibration")
    K = np.asarray(intr[serial]["original_intrinsics"], dtype=np.float64)
    dist = np.asarray(intr[serial]["dist_params"], dtype=np.float64)
    E4 = np.eye(4)
    E4[:3, :] = np.asarray(extr[serial], dtype=np.float64)
    robot_to_cam = E4 @ load_current_C2R()

    def project(p_robot):
        pc = (robot_to_cam @ np.append(np.asarray(p_robot, dtype=np.float64), 1.0))[:3]
        if pc[2] <= 1e-6:
            return None
        uv, _ = cv2.projectPoints(pc.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)
        return uv.reshape(2)

    return project


def detect_can(bgr):
    """Largest dark blob on the bright table — the pepsi can."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 110))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    h, w = mask.shape
    mask[: int(0.10 * h), :] = 0          # robot base / background clutter
    n, _, stats, cen = cv2.connectedComponentsWithStats(mask, 8)
    best, best_a = None, 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 200 < a < 0.05 * h * w and a > best_a:
            best, best_a = i, a
    return (np.asarray(cen[best]), best_a) if best is not None else (None, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--serial", default="26053260")
    p.add_argument("--urdf", default="vla_test/robot_server/xarm_inspire/xarm_inspire.urdf")
    p.add_argument("--out", default=None, help="annotated frame dir (default <session>/annotated)")
    args = p.parse_args()

    sdir = Path(args.session).expanduser()
    if not (sdir / "session.jsonl").is_file():
        raise SystemExit(f"no session.jsonl in {sdir}")
    frames_dir = sdir / "frames"
    if not frames_dir.is_dir():
        raise SystemExit(
            f"no frames in {sdir}. Re-run robot_client with --save-frames "
            "(on by default) — without the images there is nothing to align against.")
    out = Path(args.out) if args.out else sdir / "annotated"
    out.mkdir(exist_ok=True)

    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        urdf_path = REPO_ROOT / urdf_path
    urdf = yourdfpy.URDF.load(str(urdf_path))
    project = build_projector(args.serial)

    recs = [json.loads(l) for l in open(sdir / "session.jsonl")]
    queries = [r for r in recs if r["event"] == "query"]
    print(f"{len(queries)} queries, frames in {frames_dir}\n")

    rows = []
    for r in queries:
        fname = r.get("frame_file")
        if not fname or not (frames_dir / fname).is_file():
            continue
        bgr = cv2.imread(str(frames_dir / fname))
        h, w = bgr.shape[:2]
        sx, sy = w / 2048.0, h / 1536.0

        q = np.asarray(r["state"]["state.arm_joint_position"], dtype=np.float64)
        pos, rot = fk_link6(urdf, q)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rot.reshape(3, 1))[0]
        T[:3, 3] = pos
        tcp = (T @ np.array([0, 0, TCP_OFFSET_M, 1.0]))[:3]

        uv_t = project(tcp)
        if uv_t is None:
            continue
        tcp_px = uv_t * [sx, sy]
        can_px, area = detect_can(bgr)

        # local scale, taken at the gripper's own height
        step = project(tcp + np.array([0.05, 0.0, 0.0]))
        mmpp = 50.0 / max(np.linalg.norm((step - uv_t) * [sx, sy]), 1e-6)

        hand = np.asarray(r["state"]["state.dexhand_position"], dtype=np.float64)
        frac_open = float((hand / INSPIRE_LIMITS_RAD).mean())

        vis = bgr.copy()
        cv2.circle(vis, tuple(np.int32(tcp_px)), 7, (0, 0, 255), 2)
        cv2.putText(vis, "TCP", tuple(np.int32(tcp_px) + [9, -9]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        off_mm = None
        if can_px is not None:
            cv2.circle(vis, tuple(np.int32(can_px)), 7, (0, 255, 0), 2)
            cv2.arrowedLine(vis, tuple(np.int32(tcp_px)), tuple(np.int32(can_px)),
                            (255, 255, 0), 2, tipLength=0.15)
            off_mm = float(np.linalg.norm(can_px - tcp_px) * mmpp)
        cv2.putText(vis, f"q{r['idx']}  open {frac_open:.2f}"
                    + (f"  off {off_mm:.0f}mm" if off_mm is not None else "  can not found"),
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(out / fname.replace(".jpg", ".png")), vis)
        rows.append((r["idx"], frac_open, off_mm, tcp, mmpp))

    if not rows:
        raise SystemExit("no frames could be matched to log records")

    opens = np.array([x[1] for x in rows])
    k = int(opens.argmin())
    idx, fo, off, tcp, mmpp = rows[k]
    print(f"most-closed hand: query {idx}, {fo:.2f} open")
    print(f"  TCP (robot)        {np.round(tcp, 3).tolist()}")
    print(f"  gripper->can       "
          + (f"{off:.0f} mm  (scale {mmpp:.2f} mm/px at the gripper's height)"
             if off is not None else "can not detected in that frame"))
    found = [x[2] for x in rows if x[2] is not None]
    if found:
        print(f"  across the run: median {np.median(found):.0f} mm, min {min(found):.0f} mm")
    print(f"\nannotated frames -> {out}")


if __name__ == "__main__":
    main()
