import os
import time
import copy
import glob
import pickle
from typing import Tuple, Dict, Any, Optional

import numpy as np
import trimesh
import viser
import cv2

try:
    import torch
except ImportError:  # Optional dependency; only needed if pickle stores torch tensors
    torch = None

# ======================================================
# Paths (update these for your dataset/object)
# ======================================================
DATA_ROOT = "/home/temp_id/shared_data/capture/hri_inspire_left"
OBJECT_NAME = "smallbowl"
EPISODE = "1"
URDF_PATH = "/home/temp_id/paradex/rsc/robot/xarm_inspire_left_new.urdf"

BASE_PATH = os.path.join(DATA_ROOT, OBJECT_NAME, EPISODE)
TACTILE_PATH = os.path.join(BASE_PATH, "raw/hand/tactile.npy")
HAND_POSITION_PATH = os.path.join(BASE_PATH, "raw/hand")
ARM_POSITION_PATH = os.path.join(BASE_PATH, "raw/arm/position.npy")
VIDEO_PATH = os.path.join(BASE_PATH, "videos/22645029.avi")

OBJECT_TRACKING_DIR = os.path.join(BASE_PATH, "object_tracking")

OBJECT_MESH_PATH = os.path.join("/home/temp_id/shared_data/mesh", "servingbowl_small", "servingbowl_small.obj")
FRAME_OFFSET = 63  # Positive => hand/arm/tactile advance relative to object.

# ======================================================
# Tactile → (link, vertex ids)
# ======================================================
# TACTILE_VERTEX_MAP = {
#     "little_tip":    ("pinky_intermediate", [5006, 5191, 5045, 5041]),
#     "little_nail":   ("pinky_intermediate", [5207, 5481, 5380, 5246]),
#     "little_pad":    ("pinky_proximal",     [4816, 4820, 4803, 4806]),

#     "ring_tip":      ("ring_intermediate",  [3845, 3843, 4265, 4260]),
#     "ring_nail":     ("ring_intermediate",  [3964, 4080, 4368, 4458]),
#     "ring_pad":      ("ring_proximal",      [4820, 4816, 4806, 4803]),

#     "middle_tip":    ("middle_intermediate",[3736, 3738, 4118, 4121]),
#     "middle_nail":   ("middle_intermediate",[4171, 4265, 3883, 3789]),
#     "middle_pad":    ("middle_proximal",    [4822, 4815, 4819, 4824]),

#     "index_tip":     ("index_intermediate", [3843, 3976, 4379, 4265]),
#     "index_nail":    ("index_intermediate", [4458, 4471, 4094, 4080]),
#     "index_pad":     ("index_proximal",     [4822, 4815, 4819, 4824]),

#     "thumb_tip":     ("thumb_distal",       [713, 706, 667, 677]),
#     "thumb_nail":    ("thumb_distal",       [992, 993, 941, 942]),
#     "thumb_middle":  ("thumb_proximal",     [235, 236, 237, 240]),
#     "thumb_pad":     ("thumb_proximal",     [554, 389, 371, 532]),

#     "palm":          ("hand_base_link",     [25768, 25761, 11752, 25770]),
# }

TACTILE_VERTEX_MAP = {
    "little_tip":    ("left_little_2", [30136, 32377, 10140, 21072]),
    "little_nail":   ("left_little_2", [19218, 8443, 26413, 15619]),
    "little_pad":    ("left_little_1", [53841, 15868, 16563, 51124]),

    "ring_tip":      ("left_ring_2",   [15360, 29482, 1307, 6517]),
    "ring_nail":     ("left_ring_2",   [20197, 18901, 9956, 11658]),
    "ring_pad":      ("left_ring_1",   [41463, 53892, 19863, 39231]),

    "middle_tip":    ("left_middle_2", [16989, 36909, 28443, 20206]),
    "middle_nail":   ("left_middle_2", [24530, 32740, 34230, 26026]),
    "middle_pad":    ("left_middle_1", [36682, 18750, 19228, 38119]),

    "index_tip":     ("left_index_2",  [29503, 1307, 6517, 15171]),
    "index_nail":    ("left_index_2",  [20197, 12013, 4598, 13379]),
    "index_pad":     ("left_index_1",  [36688, 18750, 19219, 37862]),

    "thumb_tip":     ("left_thumb_4",  [9676, 23934, 26403, 28719]),
    "thumb_nail":    ("left_thumb_4",  [18621, 16421, 35555, 37778]),
    "thumb_middle":  ("left_thumb_2",  [15649, 22156, 14346, 5837]),
    "thumb_pad":     ("left_thumb_2",  [19300, 18427, 10008, 8949]),

    "palm":          ("base_link",     [68864, 61448, 68207, 67649]),
}

TACTILE_LAYOUT = {
    "little_tip":    (3000, 3, 3),
    "little_nail":   (3018, 12, 8),
    "little_pad":    (3210, 10, 8),
    "ring_tip":      (3370, 3, 3),
    "ring_nail":     (3388, 12, 8),
    "ring_pad":      (3580, 10, 8),
    "middle_tip":    (3740, 3, 3),
    "middle_nail":   (3758, 12, 8),
    "middle_pad":    (3950, 10, 8),
    "index_tip":     (4110, 3, 3),
    "index_nail":    (4128, 12, 8),
    "index_pad":     (4320, 10, 8),
    "thumb_tip":     (4480, 3, 3),
    "thumb_nail":    (4498, 12, 8),
    "thumb_middle":  (4690, 3, 3),
    "thumb_pad":     (4708, 12, 8),
    "palm":          (4900, 8, 14),
}

# ======================================================
# Inspire action → hand 6DOF
# ======================================================
def inspire_action_to_qpos(action: np.ndarray) -> np.ndarray:
    # Map Inspire's integer action space (0-1000) into radian joint angles.
    limits = {
        "pinky": 1.6,
        "ring": 1.6,
        "middle": 1.6,
        "index": 1.6,
        "thumb_pitch": 0.55,
        "thumb_yaw": 1.15,
    }
    qpos = np.zeros((action.shape[0], 6), dtype=float)
    qpos[:, 0] = limits["thumb_yaw"]   * (1.0 - action[:, 5] / 1000.0)
    qpos[:, 1] = limits["thumb_pitch"] * (1.0 - action[:, 4] / 1000.0)
    qpos[:, 2] = limits["index"]       * (1.0 - action[:, 3] / 1000.0)
    qpos[:, 3] = limits["middle"]      * (1.0 - action[:, 2] / 1000.0)
    qpos[:, 4] = limits["ring"]        * (1.0 - action[:, 1] / 1000.0)
    qpos[:, 5] = limits["pinky"]       * (1.0 - action[:, 0] / 1000.0)
    return qpos

# ======================================================
# Hand 6DOF → URDF joint dict (mimic 적용)
# ======================================================
def hand6_to_urdf_joints(hand6: np.ndarray) -> Dict[str, float]:
    # Expand the 6-DOF hand vector into per-joint targets, including mimic joints.
    ty, tp, i, m, r, p = hand6
    joints = {}

    # Thumb
    joints["thumb_proximal_yaw_joint"]   = ty
    joints["thumb_proximal_pitch_joint"] = tp
    joints["thumb_intermediate_joint"]   = 1.334 * tp
    joints["thumb_distal_joint"]         = 0.667 * tp

    # Index
    joints["index_proximal_joint"]     = i
    joints["index_intermediate_joint"] = 1.06399 * i - 0.04545

    # Middle
    joints["middle_proximal_joint"]     = m
    joints["middle_intermediate_joint"] = 1.06399 * m - 0.04545

    # Ring
    joints["ring_proximal_joint"]     = r
    joints["ring_intermediate_joint"] = 1.06399 * r - 0.04545

    # Pinky
    joints["pinky_proximal_joint"]     = p
    joints["pinky_intermediate_joint"] = 1.06399 * p - 0.04545

    return joints

# ======================================================
# Utils
# ======================================================
def load_series(data_dir: str, candidates: Tuple[str, ...]):
    # Pick the first existing file from candidates; useful when naming varies by run.
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return np.load(path)
    raise FileNotFoundError

def get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

def interpolate_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    # Linear interpolation to align sequences of different lengths to the video frame count.
    if len(seq) == target_len:
        return seq
    x_src = np.linspace(0, len(seq) - 1, len(seq))
    x_tgt = np.linspace(0, len(seq) - 1, target_len)
    flat = seq.reshape(len(seq), -1)
    out = np.stack(
        [np.interp(x_tgt, x_src, flat[:, i]) for i in range(flat.shape[1])],
        axis=1
    )
    return out.reshape((target_len,) + seq.shape[1:])


def to_numpy_array(x) -> np.ndarray:
    # Convert numpy/torch inputs to numpy on CPU.
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "detach"):
        return np.asarray(x.detach())
    return np.asarray(x)


def _maybe_stack_array(payload: Any) -> Optional[np.ndarray]:
    """Handle payloads that are already stacked arrays."""
    if isinstance(payload, (list, tuple)):
        return None
    arr = to_numpy_array(payload)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr.astype(float)
    return None


def _build_T_from_frame(frame: Any, idx: int) -> np.ndarray:
    """Extract or build a 4x4 transform from a single frame entry."""
    if isinstance(frame, dict):
        for k in ("T", "pose", "world_T_obj", "T_world_obj", "T_object_world"):
            if k in frame:
                T = to_numpy_array(frame[k])
                break
        else:
            if "obj_R" in frame and "obj_t" in frame:
                R = to_numpy_array(frame["obj_R"])
                t = to_numpy_array(frame["obj_t"]).reshape(3)
                T = np.eye(4, dtype=float)
                T[:3, :3] = R
                T[:3, 3] = t
            else:
                raise ValueError(f"Frame {idx} missing T/obj_R/obj_t")
    else:
        T = to_numpy_array(frame)

    T = np.asarray(T)
    if T.shape == (4, 4):
        return T.astype(float)
    if T.ndim == 3 and T.shape[0] == 1 and T.shape[1:] == (4, 4):
        return T[0].astype(float)
    if T.size == 16:
        return T.reshape(4, 4).astype(float)
    if T.shape == (3, 4):  # pad last row
        padded = np.eye(4, dtype=float)
        padded[:3, :] = T
        return padded
    raise ValueError(f"Frame {idx} transform shape {T.shape} cannot be interpreted as 4x4")


def load_object_trajectory(track_dir: str) -> np.ndarray:
    # Load the first pickle in the tracking folder into an array of shape [T, 4, 4].
    candidates = sorted(glob.glob(os.path.join(track_dir, "*.pickle")))
    if not candidates:
        raise FileNotFoundError(f"No pickle found in {track_dir}")

    with open(candidates[0], "rb") as f:
        payload = pickle.load(f)

    # If already stacked tensor/array
    stacked = _maybe_stack_array(payload)
    if stacked is not None:
        return stacked

    if isinstance(payload, dict):
        for key in ("T", "poses", "traj", "trajectory"):
            if key in payload:
                stacked = _maybe_stack_array(payload[key])
                if stacked is not None:
                    return stacked
        frames = payload.get("frames", payload)
        if isinstance(frames, dict):
            frames = [frames[k] for k in sorted(frames.keys())]
    else:
        frames = payload

    # Torch tensor or numpy array with shape (T,4,4)
    if torch is not None and isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(float)
    if isinstance(frames, np.ndarray):
        arr = np.asarray(frames)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(float)

    T_list = []
    for idx, frame in enumerate(frames):
        T_list.append(_build_T_from_frame(frame, idx))

    if not T_list:
        raise ValueError("No transforms loaded from object pickle")

    return np.stack(T_list, axis=0)


def load_object_mesh(mesh_path: str) -> trimesh.Trimesh:
    # Load and merge the object mesh into a single Trimesh.
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    if isinstance(mesh, list):
        return trimesh.util.concatenate(mesh)
    raise ValueError(f"Unexpected mesh type: {type(mesh)}")

# ======================================================
# Tactile
# ======================================================
def unpack_tactile_frame(raw, index):
    # Convert flat tactile array into a dict of named patches shaped by sensor layout.
    out = {}
    for k, (off, r, c) in index.items():
        block = raw[off:off + r * c]
        out[k] = block.reshape(r, c)
    return out

def build_tactile_index_from_layout(layout):
    idx, off = {}, 0
    for k, (_, r, c) in layout.items():
        idx[k] = (off, r, c)
        off += r * c
    return idx

# ======================================================
# Robot mesh
# ======================================================
def get_mesh(robot_wrapper, state):
    # Run FK then merge per-link meshes into trimesh objects placed in world coordinates.
    robot_wrapper.compute_forward_kinematics(state)
    robot_obj = robot_info(URDF_PATH, down_sample=True)

    out = {}
    for ln, meshes in robot_obj.mesh_dict.items():
        if not meshes:
            continue
        T = robot_wrapper.get_link_pose(robot_wrapper.get_link_index(ln))
        merged = []
        for m in meshes:
            mm = copy.deepcopy(m)
            mm.transform(T)
            merged.append(
                trimesh.Trimesh(
                    vertices=np.asarray(mm.vertices),
                    faces=np.asarray(mm.triangles),
                    process=False,
                )
            )
        out[ln] = trimesh.util.concatenate(merged)
    return out

# ======================================================
# Contact arrow
# ======================================================
def length_to_color(length, max_len=0.025):
    """
    length: 화살표 길이
    return: (R, G, B, A) in uint8
    """
    t = np.clip(length / max_len, 0.0, 1.0)

    # Blue → Cyan → Green → Yellow → Red
    if t < 0.25:
        r, g, b = 0, int(4*t*255), 255
    elif t < 0.5:
        r, g, b = 0, 255, int((1-4*(t-0.25))*255)
    elif t < 0.75:
        r, g, b = int(4*(t-0.5)*255), 255, 0
    else:
        r, g, b = 255, int((1-4*(t-0.75))*255), 0

    return np.array([r, g, b, 255], dtype=np.uint8)

def compute_contact_arrow(tm, vids):
    # Compute arrow origin/normal by averaging the chosen vertices on a link mesh.
    v = tm.vertices[vids]
    n = tm.vertex_normals[vids]
    c = v.mean(axis=0)
    n = n.mean(axis=0)
    n /= np.linalg.norm(n) + 1e-8
    return c, n


def make_arrow_mesh(start, direction, length, color_rgba):
    # Build a cylinder+cone arrow mesh oriented along the contact normal.
    if length < 1e-6:
        return None

    shaft = trimesh.creation.cylinder(radius=0.001, height=length * 0.7)
    head  = trimesh.creation.cone(radius=0.002, height=length * 0.3)

    shaft.apply_translation([0, 0, length * 0.35])
    head.apply_translation([0, 0, length * 0.85])

    arrow = trimesh.util.concatenate([shaft, head])

    arrow.apply_transform(
        trimesh.geometry.align_vectors([0, 0, 1], direction)
    )
    arrow.apply_translation(start)

    # 🔥 색상 적용 (vertex color)
    arrow.visual.vertex_colors = np.tile(
        color_rgba,
        (arrow.vertices.shape[0], 1)
    )

    return arrow

# ======================================================
# Imports for robot
# ======================================================
from paradex.robot.robot_wrapper_deprecated import RobotWrapper
from paradex.robot.robot_module import robot_info

# ======================================================
# Main
# ======================================================
def main():
    # Load data and align tactile/action/arm/object sequences to the video timeline.
    c2r = np.load(os.path.join(BASE_PATH, "C2R.npy"))

    tactile_seq = np.load(TACTILE_PATH)
    # action_seq = load_series(HAND_POSITION_PATH, ("action.npy", "position.npy"))
    action_seq = load_series(HAND_POSITION_PATH, ("position.npy", "action.npy"))
    arm_seq = np.load(ARM_POSITION_PATH)
    object_seq = load_object_trajectory(OBJECT_TRACKING_DIR)
    
    print("hand action: ", len(action_seq))
    print("arm: ", len(arm_seq))
    print("object: ", len(object_seq))

    object_mesh_template = load_object_mesh(OBJECT_MESH_PATH)
    # object_mesh_template.apply_scale()

    tactile_index = build_tactile_index_from_layout(TACTILE_LAYOUT)
    hand_qpos = inspire_action_to_qpos(action_seq)

    # T = get_video_frame_count(VIDEO_PATH)
    T = len(object_seq)
    
    print("video frames: ", T)
    
    tactile_i = interpolate_sequence(tactile_seq, T)
    hand_i = interpolate_sequence(hand_qpos, T)
    arm_i = interpolate_sequence(arm_seq, T)
    object_i = interpolate_sequence(object_seq, T)
    
    print("first object transform:\n", object_i[0])

    server = viser.ViserServer()
    robot_wrapper = RobotWrapper(URDF_PATH)

    # Floor
    server.scene.add_grid(
        name="/floor",
        width=2.0,          # 바닥 가로 크기 (m)
        height=2.0,         # 바닥 세로 크기 (m)
        cell_size=0.05,     # 그리드 한 칸 크기
        position=(0.0, 0.0, 0.0),
    )
    # World axes (X=red, Y=green, Z=blue) for orientation reference.
    axes = trimesh.creation.axis(origin_size=0.02, axis_length=0.15)
    server.scene.add_mesh_trimesh("/axes", axes)

    robot_handles = {}
    arrow_handles = {k: None for k in TACTILE_VERTEX_MAP}
    object_handle = None

    # GUI
    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider("Frame", 0, T - 1, 1, 0)
        gui_play = server.gui.add_checkbox("Play", True)
        gui_fps = server.gui.add_slider("FPS", 1, 60, 1, 30)

    joint_names = robot_wrapper.joint_names
    # print(joint_names)

    # Give the object a neutral color if none is present.
    if not hasattr(object_mesh_template.visual, "vertex_colors") or len(object_mesh_template.visual.vertex_colors) == 0:
        object_mesh_template.visual.vertex_colors = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (len(object_mesh_template.vertices), 1))

    def render_frame(t):
        nonlocal object_handle
        # Render robot links, tactile arrows, and object mesh for frame t.
        seq_t = max(0, min(T - 1, t + FRAME_OFFSET))
        arm_q = arm_i[seq_t]
        hand6 = hand_i[seq_t]

        hand_joint_dict = hand6_to_urdf_joints(hand6)

        q = np.zeros(18)
        for i, jn in enumerate(joint_names):
            if jn in hand_joint_dict:
                q[i-1] = hand_joint_dict[jn]
            elif i <= len(arm_q):
                q[i-1] = arm_q[i-1]

        tactile = unpack_tactile_frame(tactile_i[seq_t], tactile_index)
        meshes = get_mesh(robot_wrapper, q)
        # meshes.apply_transform(c2r)

        with server.atomic():
            for ln, tm in meshes.items():
                if ln in robot_handles:
                    robot_handles[ln].remove()
                robot_handles[ln] = server.scene.add_mesh_trimesh(
                    f"/robot/{ln}", tm
                )

            for name, (link, vids) in TACTILE_VERTEX_MAP.items():
                if name not in tactile:
                    continue
                p = tactile[name].mean()
                length = np.clip(p / 1000.0, 0, 1) * 0.025
                color = length_to_color(length, max_len=0.025)

                c, n = compute_contact_arrow(meshes[link], vids)
                arrow = make_arrow_mesh(c, n, length, color)
                if arrow is None:
                    if arrow_handles[name]:
                        arrow_handles[name].remove()
                        arrow_handles[name] = None
                else:
                    if arrow_handles[name]:
                        arrow_handles[name].remove()
                    arrow_handles[name] = server.scene.add_mesh_trimesh(
                        f"/contact/{name}", arrow
                    )

            # Object overlay
            if object_handle:
                object_handle.remove()
            obj_mesh = copy.deepcopy(object_mesh_template)
            
            temp_mtx = np.array([[0, -1, 0, -0.10],
                                 [1, 0, 0, 0],
                                 [0, 0, 1, -0.30],
                                 [0, 0, 0, 1]], dtype=float)
            
            transform_mtx = c2r @ object_i[t]
            t1, t2, t3 = transform_mtx[:3, 3]
            transform_mtx[:3, 3] = np.array([-t2-0.075, t1+0.03, t3 - 0.070], dtype=float)

            # obj_mesh.apply_transform(temp_mtx @ transform_mtx)
            # obj_mesh.apply_transform(transform_mtx)

            
            obj_mesh.apply_transform(object_i[t])
            object_handle = server.scene.add_mesh_trimesh(
                "/object", obj_mesh
            )

    render_frame(0)

    @gui_frame.on_update
    def _(_):
        render_frame(gui_frame.value)

    while True:
        if gui_play.value:
            gui_frame.value = (gui_frame.value + 1) % T
            render_frame(gui_frame.value)
        time.sleep(1.0 / gui_fps.value)

if __name__ == "__main__":
    main()
