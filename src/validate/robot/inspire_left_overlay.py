"""Interactive debug tool for the Inspire LEFT hand URDF.

Captures a single multi-camera snapshot plus the current xarm + inspire-left
joint state, then opens an OpenCV trackbar GUI that lets you slide the four
thumb joints and the wrist mount offset (xyz) while the URDF mesh is
re-rendered and alpha-overlaid on the captured images in real time.

Usage:
    # Fresh capture (default — saves to ~/shared_data/inspire_left_debug/<ts>)
    python src/validate/robot/inspire_left_overlay.py

    # Reuse an existing capture directory (skips robot connection)
    python src/validate/robot/inspire_left_overlay.py --load <abs_path>

Keys in GUI:
    q : quit
    s : save current trackbar state to <save_dir>/tuned.json
    r : reset trackbars to captured values
"""

import argparse
import json
import os
import re
import shutil
import tempfile
import time

import cv2
import numpy as np
import trimesh

from paradex.utils.path import home_path, shared_dir, rsc_path
from paradex.calibration.utils import (
    load_camparam, load_current_camparam, load_c2r, load_current_C2R,
    save_current_camparam, save_current_C2R, get_cammtx,
)
from paradex.image.image_dict import ImageDict
from paradex.visualization.robot import RobotModule


THUMB_LINK_PREFIX = "left_thumb"
CROP_HALF = 250  # 500×500 crop centered on the hand
OVERLAY_ALPHA = 0.5  # mesh opacity (overlay_mask image-weight = 1-OVERLAY_ALPHA)

# BGR colors per finger group; matches link-name substring.
FINGER_COLORS = {
    "thumb":  (0,   0,   255),   # red
    "index":  (0,   255, 0),     # green
    "middle": (0,   255, 255),   # yellow
    "ring":   (255, 255, 0),     # cyan
    "little": (255, 0,   255),   # magenta
    "base":   (180, 180, 180),   # palm/base in gray
}


URDF_SRC = "/home/robot/shared_data/AutoDex/content/assets/robot/inspire_left_description/xarm_inspire_left.urdf"

# Motor index (0..5) → URDF joint name; matches inspire's wire order.
# motor[0]=little, [1]=ring, [2]=middle, [3]=index, [4]=thumb_bend(_2), [5]=thumb_rot(_1)
MOTOR_ORDER = [
    "left_little_1_joint",
    "left_ring_1_joint",
    "left_middle_1_joint",
    "left_index_1_joint",
    "left_thumb_2_joint",
    "left_thumb_1_joint",
]
LEFT_LIMIT = {
    "left_little_1_joint": 1.6,
    "left_ring_1_joint":   1.6,
    "left_middle_1_joint": 1.6,
    "left_index_1_joint":  1.6,
    "left_thumb_2_joint":  0.55,
    "left_thumb_1_joint":  1.15,
}

# Slider scales (rad per 1000 trackbar steps).
THUMB_SCALES = {
    "left_thumb_1_joint": 1.15,
    "left_thumb_2_joint": 0.55,
    "left_thumb_3_joint": 0.55,   # URDF mimic is 0.6×thumb_2 → allow [0,0.55]
    "left_thumb_4_joint": 0.66,   # URDF mimic is 0.8×thumb_2; but right is 1.2× → wider range
}
WRIST_RANGE_MM = 100  # slider 0..200 maps to -100..+100 mm


def motor_to_joint(motor):
    """motor (6,) 0..1000 → dict for the 6 actuated hand joints (rad)."""
    out = {}
    for i, jn in enumerate(MOTOR_ORDER):
        out[jn] = LEFT_LIMIT[jn] * (1.0 - motor[i] / 1000.0)
    return out


def make_urdf_without_thumb_mimic(src_path, dst_path):
    """Copy URDF to dst, removing the <mimic …/> tag from thumb_3/4 joints so
    they become independently actuated. Other finger mimics are kept."""
    with open(src_path, "r") as f:
        xml = f.read()

    # Find each <joint name="left_thumb_{3,4}_joint" …> block and drop its mimic line.
    def strip_mimic(match):
        block = match.group(0)
        return re.sub(r"\s*<mimic[^/]*/>\s*", "\n    ", block)

    xml = re.sub(
        r'<joint\s+name="left_thumb_3_joint"[\s\S]*?</joint>',
        strip_mimic, xml)
    xml = re.sub(
        r'<joint\s+name="left_thumb_4_joint"[\s\S]*?</joint>',
        strip_mimic, xml)

    with open(dst_path, "w") as f:
        f.write(xml)


def capture_snapshot(save_dir):
    """Trigger remote multi-camera capture + read current arm/hand qpos."""
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.io.robot_controller import get_arm, get_hand

    os.makedirs(save_dir, exist_ok=True)
    # capture PCs save under their own ~/<rel_path>/images; main PC sees the
    # same files via the shared_data mount.
    rel_path = os.path.relpath(save_dir, home_path)

    save_current_camparam(save_dir)  # snapshot calibration alongside the images
    save_current_C2R(save_dir)       # and the handeye transform that matches it

    rcc = remote_camera_controller("image_main.py")
    try:
        rcc.start("image", False, rel_path)
        time.sleep(0.5)
        rcc.stop()
    finally:
        rcc.end()

    arm = get_arm("xarm")
    arm_qpos = np.array(arm.get_data()["qpos"], dtype=np.float64)
    # XArmController doesn't expose .end(); process exit cleans it up.

    hand = get_hand("inspire_left")
    hand_motor = np.array(hand.get_qpos(), dtype=np.float64)
    if hasattr(hand, "end"):
        try: hand.end()
        except Exception: pass

    np.save(os.path.join(save_dir, "arm_qpos.npy"), arm_qpos)
    np.save(os.path.join(save_dir, "hand_motor.npy"), hand_motor)

    print(f"[capture] saved snapshot to {save_dir}")
    return arm_qpos, hand_motor


def resolve_snapshot_dir(path):
    """If `path` itself has arm_qpos.npy, use it; otherwise treat `path` as a
    parent and pick the most-recently-modified subdirectory that does."""
    if os.path.isfile(os.path.join(path, "arm_qpos.npy")):
        return path
    candidates = []
    if os.path.isdir(path):
        for name in os.listdir(path):
            sub = os.path.join(path, name)
            if os.path.isfile(os.path.join(sub, "arm_qpos.npy")):
                candidates.append(sub)
    if not candidates:
        raise FileNotFoundError(
            f"No snapshot (arm_qpos.npy) found at {path} or any subdirectory"
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    chosen = candidates[0]
    print(f"[load] picked latest snapshot: {chosen}")
    return chosen


def load_snapshot(save_dir):
    arm_qpos = np.load(os.path.join(save_dir, "arm_qpos.npy"))
    hand_motor = np.load(os.path.join(save_dir, "hand_motor.npy"))
    return arm_qpos, hand_motor


def load_images(save_dir):
    """Read all camera PNGs from <save_dir>/images/*.png → {serial: bgr}."""
    img_dir = os.path.join(save_dir, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"no images at {img_dir}")
    images = {}
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        serial = os.path.splitext(fn)[0]
        img = cv2.imread(os.path.join(img_dir, fn))
        if img is not None:
            images[serial] = img
    if not images:
        raise RuntimeError(f"no readable images in {img_dir}")
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default=None,
                    help="absolute path of an existing snapshot directory")
    args = ap.parse_args()

    if args.load:
        save_dir = resolve_snapshot_dir(os.path.expanduser(args.load))
        arm_qpos, hand_motor = load_snapshot(save_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(shared_dir, "inspire_left_debug", ts)
        os.makedirs(save_dir, exist_ok=True)
        arm_qpos, hand_motor = capture_snapshot(save_dir)

    def reload_assets(snap_dir):
        """Load camparam + c2r + undistorted images from a snapshot dir."""
        if os.path.isdir(os.path.join(snap_dir, "cam_param")):
            intr, extr = load_camparam(snap_dir)
        else:
            intr, extr = load_current_camparam()
        if os.path.isfile(os.path.join(snap_dir, "C2R.npy")):
            c = load_c2r(snap_dir)
        else:
            c = load_current_C2R()
        ud = os.path.join(snap_dir, "undistort")
        if os.path.isdir(os.path.join(ud, "images")):
            d = ImageDict.from_path(ud)
            d.set_camparam(intr, extr)
        else:
            d = ImageDict.from_path(snap_dir)
            d.set_camparam(intr, extr)
            d = d.undistort(save_path=ud)
        return intr, extr, c, dict(d.images)

    intrinsic, extrinsic, c2r, images = reload_assets(save_dir)
    # restrict to cameras for which we have both an image and calib
    common = [s for s in images if s in intrinsic and s in extrinsic]
    if not common:
        raise RuntimeError("no camera has both image and calib loaded")
    images = {s: images[s] for s in common}
    intrinsic = {s: intrinsic[s] for s in common}
    extrinsic = {s: extrinsic[s] for s in common}
    print(f"[overlay] {len(images)} cameras: {common}")

    # ---- URDF (mimic-stripped on thumb_3/4) ----
    tmp = tempfile.mkdtemp(prefix="inspire_left_overlay_")
    urdf_path = os.path.join(tmp, "xarm_inspire_left_nomimic34.urdf")
    # need to also copy mesh dirs relative to original urdf? yourdfpy resolves
    # <mesh filename="package://..."/> via mesh search path; here meshes live
    # under rsc/robot/inspire/. yourdfpy with default loader looks relative to
    # the urdf file, so copy meshes alongside.
    # Easier: write the modified URDF next to the original.
    urdf_path = os.path.join(os.path.dirname(URDF_SRC), ".tmp_inspire_left_overlay.urdf")
    make_urdf_without_thumb_mimic(URDF_SRC, urdf_path)

    rm = RobotModule(urdf_path)
    actuated = rm.get_joint_names()
    print(f"[overlay] {len(actuated)} actuated joints: {actuated}")

    # ---- initial cfg from captured state ----
    init_hand = motor_to_joint(hand_motor)
    cfg = {}
    for i in range(6):
        cfg[f"joint{i+1}"] = float(arm_qpos[i])
    for jn in MOTOR_ORDER:
        cfg[jn] = init_hand[jn]
    # thumb_3/4 are now actuated (mimic stripped); set from URDF's mimic ratios
    cfg["left_thumb_3_joint"] = init_hand["left_thumb_2_joint"] * 0.60
    cfg["left_thumb_4_joint"] = init_hand["left_thumb_2_joint"] * 0.80

    print(f"[init] motor: {hand_motor.tolist()}")
    print(f"[init] arm qpos (rad): {np.round(arm_qpos, 4).tolist()}")
    print(f"[init] thumb cfg (rad): "
          f"t1={cfg['left_thumb_1_joint']:.3f}, t2={cfg['left_thumb_2_joint']:.3f}, "
          f"t3={cfg['left_thumb_3_joint']:.3f}, t4={cfg['left_thumb_4_joint']:.3f}")

    # overlay_mask() mutates in place, so keep an untouched copy and restore
    # before each project_mesh() call. Persistent ImageDict keeps BatchRenderer cached.
    orig_images = {s: img.copy() for s, img in images.items()}
    imgdict = ImageDict({s: img.copy() for s, img in orig_images.items()},
                        intrinsic, extrinsic, path=None)
    cammtx = get_cammtx(intrinsic, extrinsic)  # 3x4 per camera in world frame

    def build_hand_groups(cur_cfg, wrist_delta):
        """Return (groups_dict, hand_pivot_world).
        Groups: thumb, index, middle, ring, little, base (palm).
        hand_pivot_world: centroid of all hand meshes' bbox in camera world."""
        rm.urdf.update_cfg(cur_cfg)
        T_link6 = rm.urdf.get_transform("link6", rm.urdf.base_link)
        R6 = T_link6[:3, :3]
        T_extra = np.eye(4)
        T_extra[:3, 3] = R6 @ np.asarray(wrist_delta, dtype=np.float64)

        scene = rm.urdf.scene
        groups = {}
        all_pts = []
        for geom_name, mesh in scene.geometry.items():
            lname = geom_name.lower()
            if "left_thumb" in lname:   key = "thumb"
            elif "left_index" in lname: key = "index"
            elif "left_middle" in lname:key = "middle"
            elif "left_ring" in lname:  key = "ring"
            elif "left_little" in lname:key = "little"
            elif "base_link" in lname or "left_base" in lname: key = "base"
            else: continue  # skip xarm links
            T_world_geom = scene.graph.get(geom_name)[0]
            T_final = c2r @ T_extra @ T_world_geom
            m = mesh.copy()
            m.apply_transform(T_final)
            groups[key] = m if key not in groups else trimesh.util.concatenate([groups[key], m])
            all_pts.append(m.bounds.mean(axis=0))
        pivot = np.mean(all_pts, axis=0) if all_pts else np.zeros(3)
        return groups, pivot

    def project_point(p_world, cm_3x4):
        x = np.append(p_world, 1.0)
        uvw = cm_3x4 @ x
        return uvw[:2] / uvw[2]

    def crop_around(img, uv, half=CROP_HALF):
        H, W = img.shape[:2]
        u, v = int(uv[0]), int(uv[1])
        u0, v0 = max(0, u - half), max(0, v - half)
        u1, v1 = min(W, u + half), min(H, v + half)
        out = np.zeros((2 * half, 2 * half, 3), dtype=img.dtype)
        if u0 < u1 and v0 < v1:
            ox, oy = u0 - (u - half), v0 - (v - half)
            out[oy:oy + (v1 - v0), ox:ox + (u1 - u0)] = img[v0:v1, u0:u1]
        return out

    cammtx = get_cammtx(intrinsic, extrinsic)

    # ---- trackbars ----
    cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("overlay", 1280, 720)
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("controls", 600, 280)

    def to_slider(rad, scale): return int(np.clip(rad / scale, 0.0, 1.0) * 1000)

    init_sliders = {
        "thumb_1": to_slider(cfg["left_thumb_1_joint"], THUMB_SCALES["left_thumb_1_joint"]),
        "thumb_2": to_slider(cfg["left_thumb_2_joint"], THUMB_SCALES["left_thumb_2_joint"]),
        "thumb_3": to_slider(cfg["left_thumb_3_joint"], THUMB_SCALES["left_thumb_3_joint"]),
        "thumb_4": to_slider(cfg["left_thumb_4_joint"], THUMB_SCALES["left_thumb_4_joint"]),
        "wrist_x": WRIST_RANGE_MM,  # midpoint = 0 offset
        "wrist_y": WRIST_RANGE_MM,
        "wrist_z": WRIST_RANGE_MM,
    }
    for name, init in init_sliders.items():
        maxv = 2 * WRIST_RANGE_MM if name.startswith("wrist") else 1000
        cv2.createTrackbar(name, "controls", init, maxv, lambda _v: None)

    def reset_sliders():
        for name, init in init_sliders.items():
            cv2.setTrackbarPos(name, "controls", init)

    def read(name): return cv2.getTrackbarPos(name, "controls")

    def recapture():
        """Trigger a fresh multi-camera snapshot + reload everything."""
        nonlocal c2r, cammtx, save_dir, arm_qpos, hand_motor, intrinsic, extrinsic
        ts2 = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(shared_dir, "inspire_left_debug", ts2)
        os.makedirs(save_dir, exist_ok=True)
        print(f"[recapture] capturing to {save_dir} (GUI will freeze) ...", flush=True)
        arm_qpos, hand_motor = capture_snapshot(save_dir)
        print("[recapture] capture done, undistorting + reloading ...", flush=True)
        intrinsic, extrinsic, c2r, new_images = reload_assets(save_dir)
        common2 = [s for s in new_images if s in intrinsic and s in extrinsic]
        new_images = {s: new_images[s] for s in common2}
        intrinsic = {s: intrinsic[s] for s in common2}
        extrinsic = {s: extrinsic[s] for s in common2}
        cammtx = get_cammtx(intrinsic, extrinsic)
        orig_images.clear()
        orig_images.update({s: img.copy() for s, img in new_images.items()})
        imgdict.images = {s: img.copy() for s, img in new_images.items()}
        imgdict.intrinsic = intrinsic
        imgdict.extrinsic = extrinsic
        imgdict._cache = {}  # invalidate BatchRenderer (extrinsics may differ)
        init_hand2 = motor_to_joint(hand_motor)
        for i in range(6): cfg[f"joint{i+1}"] = float(arm_qpos[i])
        for jn in MOTOR_ORDER: cfg[jn] = init_hand2[jn]
        cfg["left_thumb_3_joint"] = init_hand2["left_thumb_2_joint"] * 0.60
        cfg["left_thumb_4_joint"] = init_hand2["left_thumb_2_joint"] * 0.80
        for n, scale in (("thumb_1", THUMB_SCALES["left_thumb_1_joint"]),
                         ("thumb_2", THUMB_SCALES["left_thumb_2_joint"]),
                         ("thumb_3", THUMB_SCALES["left_thumb_3_joint"]),
                         ("thumb_4", THUMB_SCALES["left_thumb_4_joint"])):
            init_sliders[n] = to_slider(cfg[f"left_{n}_joint"], scale)
        init_sliders["wrist_x"] = WRIST_RANGE_MM
        init_sliders["wrist_y"] = WRIST_RANGE_MM
        init_sliders["wrist_z"] = WRIST_RANGE_MM
        reset_sliders()
        print(f"[recapture] {save_dir} ({len(common2)} cams)")

    # ---- main loop ----
    while True:
        # 1) trackbars → cfg + wrist offset
        cfg["left_thumb_1_joint"] = read("thumb_1") / 1000.0 * THUMB_SCALES["left_thumb_1_joint"]
        cfg["left_thumb_2_joint"] = read("thumb_2") / 1000.0 * THUMB_SCALES["left_thumb_2_joint"]
        cfg["left_thumb_3_joint"] = read("thumb_3") / 1000.0 * THUMB_SCALES["left_thumb_3_joint"]
        cfg["left_thumb_4_joint"] = read("thumb_4") / 1000.0 * THUMB_SCALES["left_thumb_4_joint"]
        dx = (read("wrist_x") - WRIST_RANGE_MM) * 0.001
        dy = (read("wrist_y") - WRIST_RANGE_MM) * 0.001
        dz = (read("wrist_z") - WRIST_RANGE_MM) * 0.001
        wrist_delta = np.array([dx, dy, dz])

        # 2) hand groups in camera-world frame + hand centroid for crop
        groups, hand_pivot = build_hand_groups(cfg, wrist_delta)

        # 3) restore originals, then overlay each group with its own color.
        # overlay_mask mutates imgdict.images in place, so layered calls
        # accumulate the colored masks into a single multi-color overlay.
        for s in orig_images:
            np.copyto(imgdict.images[s], orig_images[s])
        for key, mesh in groups.items():
            imgdict.project_mesh(mesh, color=FINGER_COLORS[key],
                                 alpha=1.0 - OVERLAY_ALPHA)

        # 4) per-camera crop centered on hand centroid → grid
        crops = {}
        for serial in imgdict.images:
            uv = project_point(hand_pivot, cammtx[serial])
            crops[serial] = crop_around(imgdict.images[serial], uv, half=CROP_HALF)
        grid = ImageDict(crops, intrinsic=None, extrinsic=None, path=None).merge()
        hud = [
            f"t1={np.degrees(cfg['left_thumb_1_joint']):6.1f}deg  "
            f"t2={np.degrees(cfg['left_thumb_2_joint']):6.1f}deg  "
            f"t3={np.degrees(cfg['left_thumb_3_joint']):6.1f}deg  "
            f"t4={np.degrees(cfg['left_thumb_4_joint']):6.1f}deg",
            f"wrist(link6) dx={dx*1000:+6.1f}  dy={dy*1000:+6.1f}  dz={dz*1000:+6.1f}  [mm]",
        ]
        for i, line in enumerate(hud):
            y = 22 + i * 22
            cv2.putText(grid, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(grid, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("overlay", grid)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            recapture()
        elif k == ord('r'):
            reset_sliders()
        elif k == ord('s'):
            dump = {
                "cfg": {k_: float(v) for k_, v in cfg.items()},
                "wrist_offset_mm_link6": [float(dx * 1000), float(dy * 1000), float(dz * 1000)],
            }
            out = os.path.join(save_dir, "tuned.json")
            with open(out, "w") as f:
                json.dump(dump, f, indent=2)
            print(f"[overlay] saved {out}")

    cv2.destroyAllWindows()
    try: os.remove(urdf_path)
    except OSError: pass


if __name__ == "__main__":
    main()
