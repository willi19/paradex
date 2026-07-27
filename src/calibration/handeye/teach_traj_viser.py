"""Build a hand-eye calibration trajectory by "teaching" the Franka INSIDE Viser.

No real robot / franka library needed: pose the arm with 7 joint sliders, watch the
charuco cube against the camera rig, and hit Save to record each pose. yourdfpy gives
the EEF pose via FK. Writes <i>_qpos.npy (7 joints) + <i>_pose.npy (fr3_link8 4x4)
into system/current/hecalib/franka/ — same layout as franka_teaching.py, which
handeye/capture.py replays.

Cameras are drawn from a hand-eye session's C2R (placed in robot base frame). Since
that C2R was solved for the xarm base, camera positions are APPROXIMATE for the franka
(good enough to reason about visibility); recalibrate once the franka is mounted.

Usage:
    python src/calibration/handeye/teach_traj_viser.py
    python src/calibration/handeye/teach_traj_viser.py --no-cameras
    python src/calibration/handeye/teach_traj_viser.py --cam-session 20260625_174310
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).resolve().parents[3]))
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.utils.system import config_dir
from paradex.calibration.utils import handeye_calib_path

REPO = Path(__file__).resolve().parents[3]
URDF = str(REPO / "rsc/robot/franka/franka_adapter.urdf")
EEF_LINK = "fr3_link8"
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # franka ready pose
OUT_DIR = os.path.join(config_dir, "hecalib", "franka")


def load_cameras(viewer, session):
    base = os.path.join(handeye_calib_path, session, "0")
    c2r_p = os.path.join(base, "C2R.npy")
    cp = os.path.join(base, "cam_param")
    if not (os.path.exists(c2r_p) and os.path.exists(os.path.join(cp, "extrinsics.json"))):
        print(f"(no C2R/cam_param in {session}; skipping cameras)")
        return
    inv_c2r = np.linalg.inv(np.load(c2r_p))  # C2R: robot->colmap ; inv: colmap->robot
    ext = json.load(open(os.path.join(cp, "extrinsics.json")))
    intr = json.load(open(os.path.join(cp, "intrinsics.json")))
    n = 0
    for s, v in ext.items():
        if s not in intr:
            continue
        E = np.array(v).reshape(3, 4)
        wfc = np.eye(4)
        wfc[:3, :3] = E[:3, :3].T
        wfc[:3, 3] = -E[:3, :3].T @ E[:3, 3]
        T = inv_c2r @ wfc
        K = np.array(intr[s]["intrinsics_undistort"]).reshape(3, 3)
        viewer.add_camera(s, T, K, color=(0.0, 0.9, 0.3), size=0.06)
        n += 1
    print(f"placed {n} cameras (robot frame, approx for franka)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-session", default=None, help="handeye session for cameras (default: latest w/ C2R)")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    fk = yourdfpy.URDF.load(URDF, build_scene_graph=True)
    jnames = fk.actuated_joint_names
    lo = np.array([j.limit.lower for j in fk.actuated_joints])
    hi = np.array([j.limit.upper for j in fk.actuated_joints])
    home = np.clip(HOME, lo, hi)

    viewer = ViserViewer(port_number=args.port)
    viewer.add_grid(size=2.0, cell_size=0.1, height=0.0)
    viewer.add_robot("franka", URDF)
    robot = viewer.robot_dict["franka"]

    if not args.no_cameras:
        session = args.cam_session
        if session is None:
            for s in sorted(os.listdir(handeye_calib_path), reverse=True):
                if os.path.exists(os.path.join(handeye_calib_path, s, "0", "C2R.npy")):
                    session = s
                    break
        if session:
            load_cameras(viewer, session)

    saved = []  # list of (qpos(7,), ee(4,4))

    def current_cfg():
        return np.array([s.value for s in sliders])

    def eef_pose(cfg):
        fk.update_cfg(cfg)
        return np.array(fk.get_transform(EEF_LINK, fk.base_link))

    with viewer.server.gui.add_folder("Joints"):
        sliders = [viewer.server.gui.add_slider(
            jnames[i], min=float(lo[i]), max=float(hi[i]), step=0.01, initial_value=float(home[i]))
            for i in range(len(jnames))]
    with viewer.server.gui.add_folder("Trajectory"):
        btn_save = viewer.server.gui.add_button("Save pose")
        btn_del = viewer.server.gui.add_button("Delete last")
        btn_write = viewer.server.gui.add_button("Write to hecalib/franka")
        status = viewer.server.gui.add_text("saved", initial_value="0 poses")

    def refresh_status():
        status.value = f"{len(saved)} poses"

    def update(_=None):
        cfg = current_cfg()
        robot.update_cfg(cfg)

    for s in sliders:
        s.on_update(update)
    update()

    @btn_save.on_click
    def _(_=None):
        cfg = current_cfg()
        saved.append((cfg.copy(), eef_pose(cfg)))
        refresh_status()
        print(f"saved pose {len(saved)-1}: qpos={np.round(cfg,3)}")

    @btn_del.on_click
    def _(_=None):
        if saved:
            saved.pop()
            refresh_status()

    @btn_write.on_click
    def _(_=None):
        os.makedirs(OUT_DIR, exist_ok=True)
        for i, (q, ee) in enumerate(saved):
            np.save(os.path.join(OUT_DIR, f"{i}_qpos.npy"), q)
            np.save(os.path.join(OUT_DIR, f"{i}_pose.npy"), ee)  # match franka_teaching.py
        status.value = f"wrote {len(saved)} poses -> {OUT_DIR}"
        print(status.value)

    print(f"Pose with joint sliders; Save each pose; Write when done -> {OUT_DIR}")
    print("Ctrl-C to quit.")
    viewer.start_viewer(use_thread=False)


if __name__ == "__main__":
    main()
