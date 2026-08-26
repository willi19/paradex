"""Capture one multi-camera image set and overlay the Franka at its live joint pose.

This is the end-to-end check of the calibration chain: intrinsics -> extrinsics ->
C2R. If the rendered arm lands on the real arm in every view, all three are good;
a consistent offset in one view usually means that camera's extrinsic, while an
offset that grows with distance from the robot base points at C2R.

Unlike ``src/object6d/validate_c2r.py`` (existing images, zero qpos) this captures
now and reads the arm's actual ``qpos`` from the running franka_daemon, so the
overlay can be checked in any pose without staging a dataset first.

Prereqs: capture-PC daemons up, and ``./cpp/franka_daemon/run_daemon.sh`` running.

Usage:
    python src/validate/robot/franka_overlay.py
    python src/validate/robot/franka_overlay.py --save_path mingi_erasethis/franka_ovl
    python src/validate/robot/franka_overlay.py --reuse <existing capture dir>
"""
import argparse
import os
import time

import cv2
import numpy as np

from paradex.calibration.utils import handeye_calib_path, load_camparam, load_c2r
from paradex.utils.file_io import find_latest_directory
from paradex.image.grid import make_image_grid
from paradex.image.overlay import overlay_mask
from paradex.image.projection import BatchRenderer
from paradex.utils.path import rsc_path, shared_dir, assert_shared_data_mounted
from paradex.visualization.robot import RobotModule


def capture(save_path, arm=None):
    """Grab one synchronized image per camera into ``<save_path>/raw/images``.

    Reads the arm pose on both sides of the shot: the overlay is only meaningful if
    the arm held still through it, and a pose read after the fact silently renders
    the wrong configuration if it did not.
    """
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

    rel = os.path.relpath(save_path, os.path.expanduser("~"))
    rcc = remote_camera_controller("franka_overlay")
    before = after = None
    try:
        before = arm.get_data()["qpos"] if arm is not None else None
        rcc.start("image", False, os.path.join(rel, "raw"))
        rcc.stop()
        after = arm.get_data()["qpos"] if arm is not None else None
    finally:
        rcc.end()
    if before is not None and after is not None:
        drift = float(np.abs(np.asarray(after) - np.asarray(before)).max())
        print(f"    qpos drift across the shot: {drift:.6f} rad")
        if drift > 1e-3:
            raise RuntimeError(
                f"the arm moved {drift:.4f} rad during the capture — the overlay would "
                f"be rendered in a pose the photo never had. Hold it still and retry.")
    return np.asarray(before) if before is not None else None

    # The daemons write asynchronously; wait for files rather than guessing a sleep.
    img_dir = os.path.join(save_path, "raw", "images")
    for _ in range(100):
        if os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 0:
            time.sleep(1.0)          # let the stragglers land
            break
        time.sleep(0.2)
    return img_dir


# Inspire reports 6 raw channels, 0 = fully closed .. 1000 = fully open, ordered
# [little, ring, middle, index, thumb_bend, thumb_rot]. The URDF's actuated joints
# come out in a different order and in radians measured from open, so each channel is
# mapped by name and scaled by that joint's own upper limit.
INSPIRE_TO_URDF = {
    "right_little_1_joint": 0,
    "right_ring_1_joint":   1,
    "right_middle_1_joint": 2,
    "right_index_1_joint":  3,
    "right_thumb_1_joint":  4,   # bend
    "right_thumb_2_joint":  5,   # rotation/opposition
}


def read_hand():
    """Inspire's 6 raw joint values, or None if the hand is not reachable."""
    try:
        from paradex.io.robot_controller import get_hand
        hand = get_hand("inspire")
    except Exception as e:
        print(f"    hand unavailable ({type(e).__name__}: {e}) — fingers render at zero")
        return None
    try:
        for _ in range(30):
            d = hand.get_data()
            if d is not None and d.get("qpos") is not None:
                return np.asarray(d["qpos"], dtype=float)
            time.sleep(0.1)
    finally:
        hand.end()
    return None


def read_qpos():
    """Live joint angles from the franka_daemon, or None if it is not answering."""
    from paradex.io.robot_controller.franka_controller import FrankaController
    from paradex.utils.system import network_info

    arm = FrankaController(network_info["franka"])
    try:
        for _ in range(50):
            data = arm.get_data()
            if data is not None:
                return np.array(data["qpos"], dtype=float)
            time.sleep(0.1)
    finally:
        arm.end()
    return None


def _urdf_upper_limits(urdf_path):
    import xml.etree.ElementTree as ET
    out = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("upper") is not None:
            out[j.get("name")] = float(lim.get("upper"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save_path", default=None,
                    help="capture dir under shared_data (default: franka_overlay/<ts>)")
    ap.add_argument("--reuse", default=None,
                    help="skip capture and overlay onto this existing capture dir")
    ap.add_argument("--qpos", type=float, nargs=7, default=None,
                    help="use these joint angles instead of reading the daemon")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--links", default="all",
                    help="which geometry to draw: 'all', 'hand' (hand + the last arm "
                         "link, which is what you want to judge hand alignment without "
                         "the arm covering it), or a comma-separated name prefix list")
    ap.add_argument("--no_hand", action="store_true",
                    help="skip the Inspire read and draw the fingers open")
    ap.add_argument("--urdf", default=os.path.join(
        shared_dir, "AutoDex/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf"),
        help="robot URDF (default: the fr3_inspire the planner actually runs with)")
    args = ap.parse_args()

    assert_shared_data_mounted("a franka overlay capture")

    captured_qpos = None
    if args.reuse:
        root = args.reuse if os.path.isabs(args.reuse) else os.path.join(shared_dir, args.reuse)
        img_dir = os.path.join(root, "raw", "images")
        print(f"[1] reusing {img_dir}")
    else:
        rel = args.save_path or os.path.join("franka_overlay",
                                             time.strftime("%Y%m%d_%H%M%S"))
        root = rel if os.path.isabs(rel) else os.path.join(shared_dir, rel)
        print(f"[1] capturing -> {root}")
        from paradex.io.robot_controller.franka_controller import FrankaController
        from paradex.utils.system import network_info
        _arm = FrankaController(network_info["franka"])
        try:
            for _ in range(50):
                if _arm.get_data() is not None:
                    break
                time.sleep(0.1)
            captured_qpos = capture(root, _arm)
        finally:
            _arm.end()
        img_dir = os.path.join(root, "raw", "images")

    raw = {}
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith((".png", ".jpg")):
            img = cv2.imread(os.path.join(img_dir, f))
            if img is not None:
                raw[os.path.splitext(f)[0]] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"    {len(raw)} images")
    if not raw:
        raise RuntimeError(f"no images in {img_dir}")

    qpos_file = os.path.join(root, "qpos.npy")
    if args.qpos:
        qpos = np.array(args.qpos)
    elif args.reuse and os.path.exists(qpos_file):
        # The arm has almost certainly moved since; the images are of the old pose.
        qpos = np.load(qpos_file)
        print("    (qpos from the capture, not the arm's current pose)")
    elif not args.reuse and captured_qpos is not None:
        qpos = captured_qpos
    else:
        qpos = read_qpos()
    if qpos is None:
        raise RuntimeError("franka_daemon is not publishing state — is run_daemon.sh up?")
    hand_file = os.path.join(root, "hand_qpos.npy")
    if args.no_hand:
        hand_raw = None
    elif args.reuse and os.path.exists(hand_file):
        hand_raw = np.load(hand_file)
    else:
        hand_raw = read_hand()
        if hand_raw is not None and not args.reuse:
            np.save(hand_file, hand_raw)
    if not args.reuse:
        np.save(qpos_file, qpos)
    print(f"[2] qpos = {np.round(qpos, 4)}")

    # C2R maps robot -> the world frame of the extrinsic that produced it, and
    # COLMAP fixes that frame arbitrarily per run: two solves of the same rig differ
    # here by a rigid transform (measured: 54 deg / 32 mm between two of ours, with a
    # 0.28 mm rigid-fit residual). Pairing C2R with load_current_camparam() therefore
    # renders a perfectly shaped robot in the wrong place. Take both from the same
    # handeye run's own camparam snapshot.
    he = find_latest_directory(handeye_calib_path)
    he_root = os.path.join(handeye_calib_path, he)
    he_idx = sorted(os.listdir(he_root))[0]
    intrinsic, extrinsic = load_camparam(os.path.join(he_root, he_idx))
    c2r = load_c2r(os.path.join(he_root, he_idx))
    print(f"[3] handeye {he} | {len(intrinsic)} cams | C2R t = {np.round(c2r[:3, 3], 4)}")

    robot = RobotModule(args.urdf)
    names = robot.get_joint_names()
    cfg = np.zeros(len(names))
    arm = [i for i, nm in enumerate(names) if nm.startswith("fr3_joint")]
    for k, i in enumerate(arm[:len(qpos)]):
        cfg[i] = qpos[k]

    if hand_raw is not None:
        upper = _urdf_upper_limits(args.urdf)
        for nm, ch in INSPIRE_TO_URDF.items():
            if nm in names and ch < len(hand_raw):
                # raw 1000 = open = 0 rad; raw 0 = closed = the joint's upper limit
                cfg[names.index(nm)] = (1.0 - hand_raw[ch] / 1000.0) * upper.get(nm, 0.0)
        print(f"    hand raw = {np.round(hand_raw, 1)}")
    robot.update_cfg(cfg)

    if args.links == "all":
        mesh = robot.get_robot_mesh()
    else:
        prefixes = (("base_link", "right_", "fr3_link7") if args.links == "hand"
                    else tuple(x.strip() for x in args.links.split(",") if x.strip()))
        parts = []
        for g in robot.scene.geometry:
            if not g.startswith(prefixes):
                continue
            lm = robot.scene.geometry[g].copy()
            lm.apply_transform(robot.scene.graph.get(g)[0])
            parts.append(lm)
        if not parts:
            raise RuntimeError(f"--links {args.links!r} matched no geometry; "
                               f"available: {sorted(robot.scene.geometry)}")
        import trimesh
        mesh = trimesh.util.concatenate(parts)
        print(f"    drawing {len(parts)} geometries ({args.links})")
    print(f"[4] mesh {len(mesh.vertices)} verts")

    # BatchRenderer projects with `intrinsics_undistort`, so the photo has to be
    # undistorted too. Overlaying on the raw frame instead puts the robot visibly off
    # wherever it sits near the image border, which reads exactly like a bad C2R.
    images = {}
    for c, img in raw.items():
        if c not in intrinsic:
            continue
        K = np.array(intrinsic[c]["original_intrinsics"], dtype=float).reshape(3, 3)
        Ku = np.array(intrinsic[c]["intrinsics_undistort"], dtype=float).reshape(3, 3)
        d = np.array(intrinsic[c]["dist_params"], dtype=float).reshape(1, -1)
        images[c] = cv2.undistort(img, K, d, None, Ku)

    cams = [c for c in intrinsic if c in images and c in extrinsic]
    extr_c2r = {}
    for cam in cams:
        cam_from_world = np.eye(4)
        cam_from_world[:3, :] = np.array(extrinsic[cam])[:3, :4]
        extr_c2r[cam] = (cam_from_world @ c2r)[:3, :]

    print(f"[5] rendering {len(cams)} cams")
    _, mask_dict, _ = BatchRenderer({c: intrinsic[c] for c in cams}, extr_c2r).render(mesh)

    tag = "" if args.links == "all" else "_" + args.links.replace(",", "+")
    out_dir = os.path.join(root, "franka_overlay" + tag)
    os.makedirs(out_dir, exist_ok=True)
    overlays = []
    for cam in sorted(cams):
        ovl = overlay_mask(images[cam].copy(), mask_dict[cam], color=(0, 255, 0),
                           alpha=args.alpha)
        overlays.append(ovl)
        cv2.imwrite(os.path.join(out_dir, f"{cam}.jpg"), cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR))
    grid_path = os.path.join(out_dir, "grid.jpg")
    cv2.imwrite(grid_path, cv2.cvtColor(make_image_grid(overlays), cv2.COLOR_RGB2BGR))
    print(f"[6] {out_dir}\n    {grid_path}")


if __name__ == "__main__":
    main()
