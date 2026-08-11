"""Live 3D view of the VIVE+MANUS teleop INPUT (fused hand), before driving any
robot. Reads ViveManusROSReceiver.get_data() and renders wrist frame + finger
skeletons in a Viser web viewer. No arm/hand controller, no recording.

Prerequisites (same as teleop, minus the robot):
  - source /opt/ros/humble/setup.bash
  - source <vive-teleop>/ros2_ws/install/setup.bash        (manus_ros2_msgs)
  - VIVE publisher running  : /vive_trackers/right_hand/pose
  - MANUS publisher running : /manus_glove_0, /manus_glove_1  (both gloves)

Run:
  python src/dataset_acquisition/hri/visualize_teleop_input.py --port 8081
  open http://localhost:8081
"""
import time
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R
import viser

from paradex.io.teleop.vive.receiver import ViveManusROSReceiver

# Skeleton connectivity (wrist -> each finger chain). Matches the joint key
# names produced by the receiver's MANUS->named-pose conversion.
FINGER_CHAINS = {
    "thumb":  ["wrist", "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"],
    "index":  ["wrist", "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip"],
    "middle": ["wrist", "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip"],
    "ring":   ["wrist", "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip"],
    "pinky":  ["wrist", "pinky_metacarpal", "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip"],
}
# viser native RGB is 0-255.
FINGER_COLORS = {
    "thumb":  (255, 90, 90),
    "index":  (90, 220, 90),
    "middle": (90, 160, 255),
    "ring":   (255, 200, 60),
    "pinky":  (210, 110, 255),
}


def _matrix_to_wxyz(matrix):
    quat_xyzw = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def draw_hand(server, frame, prefix):
    if frame is None or "wrist" not in frame:
        return
    all_joints = sorted({j for chain in FINGER_CHAINS.values() for j in chain})
    present = [j for j in all_joints if j in frame]
    points = np.array([frame[j][:3, 3] for j in present], dtype=np.float32)
    colors = np.tile(np.array([[235, 235, 235]], dtype=np.uint8), (len(points), 1))
    server.scene.add_point_cloud(
        f"{prefix}/joints", points=points, colors=colors,
        point_size=0.008, point_shape="circle",
    )

    wrist = frame["wrist"]
    server.scene.add_frame(
        f"{prefix}/wrist", axes_length=0.06, axes_radius=0.004,
        position=wrist[:3, 3], wxyz=_matrix_to_wxyz(wrist),
    )

    for finger, chain in FINGER_CHAINS.items():
        pts = [frame[j][:3, 3] for j in chain if j in frame]
        if len(pts) >= 2:
            server.scene.add_spline_catmull_rom(
                f"{prefix}/{finger}", points=np.array(pts, dtype=np.float32),
                line_width=3.0, color=FINGER_COLORS[finger],
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-side", default="right", choices=["right", "bimanual"])
    ap.add_argument("--port", type=int, default=8081,
                    help="Viser web port (8080 is often taken by VS Code).")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    receiver = ViveManusROSReceiver(hand_side=args.hand_side)
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction((0.0, 0.0, 1.0))
    server.scene.add_grid("grid", width=2.0, height=2.0)
    status = server.gui.add_text("status", initial_value="waiting for data...")
    print(f"Teleop-input viewer: http://localhost:{args.port}  (Ctrl-C to quit)")

    period = 1.0 / max(args.fps, 1.0)
    n_ok = 0
    try:
        while True:
            data = receiver.get_data()
            right = data.get("Right") if data else None
            left = data.get("Left") if data else None
            fresh = right is not None or (args.hand_side == "bimanual" and left is not None)

            if fresh:
                n_ok += 1
                draw_hand(server, right, "right")
                if args.hand_side == "bimanual":
                    draw_hand(server, left, "left")
                status.value = f"streaming (frames: {n_ok})"
            else:
                status.value = ("no fresh data — check VIVE + BOTH MANUS gloves "
                                "publishers and ROS sourcing")
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.end()


if __name__ == "__main__":
    main()
