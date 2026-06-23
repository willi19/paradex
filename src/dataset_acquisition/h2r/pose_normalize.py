"""Per-object 6D pose normalization — the same technique the H2R grasp
pipeline (hrdex_afford / build_scene_assets) applies to capture data.

A normalization mode rolls / canonicalizes the object pose so the optimizer
and the paired dataset see a consistent object orientation:
  none      : no change
  x_roll    : roll about the aligned object's local +X so +Y points world +X
              (cylindrical objects)
  axis_sign : 24-fold cuboid-symmetry canonicalization (box-like objects)
  sphere    : replace the aligned rotation with identity (spherical objects)

All functions are numpy-only. `pose_C` is the object pose as returned by the
6D estimator; `c2r` is the 4x4 camera->robot matrix (C2R.npy); `T_align` is
`<mesh>_viser_align.npy` (identity if absent).
"""
import numpy as np

MODES = ("none", "x_roll", "axis_sign", "sphere")


def _canonicalize_axis_sign(vx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Pick a consistent sign for the X axis: prefer world +Z, then +X, +Y."""
    for c in (2, 0, 1):
        if vx[c] > eps:
            return vx
        if vx[c] < -eps:
            return -vx
    return vx


def normalize_x_axis_roll(pose_R_aligned: np.ndarray,
                          target_world_axis: np.ndarray = np.array([1., 0., 0.]),
                          canonicalize_flip: bool = True) -> np.ndarray:
    """Roll about the aligned object's local +X so the aligned +Y points along
    `target_world_axis`. Translation unchanged."""
    out = pose_R_aligned.copy()
    R = out[:3, :3]
    vx = R[:, 0]
    nx = np.linalg.norm(vx)
    if nx < 1e-9:
        return out
    vx = vx / nx
    if canonicalize_flip:
        vx = _canonicalize_axis_sign(vx)
    proj = target_world_axis - np.dot(target_world_axis, vx) * vx
    pn = np.linalg.norm(proj)
    if pn < 1e-6:
        return out                       # X colinear with target: roll ambiguous
    vy = proj / pn
    vz = np.cross(vx, vy)
    out[:3, :3] = np.column_stack([vx, vy, vz])
    return out


def _axis_flip_group() -> np.ndarray:
    """Klein four-group: identity + 180° rotations about each principal axis."""
    diags = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                     dtype=float)
    return np.stack([np.diag(d) for d in diags], axis=0)


_CUBOID_ROTS = _axis_flip_group()


def normalize_cuboid_symmetry(pose_R_aligned: np.ndarray,
                              target_R: np.ndarray = np.eye(3)) -> np.ndarray:
    """Pick the cuboid-symmetry variant whose rotation is closest to target_R."""
    out = pose_R_aligned.copy()
    R_cur = out[:3, :3]
    Rvs = np.einsum("ij,kjl->kil", R_cur, _CUBOID_ROTS)
    scores = np.einsum("kii->k", Rvs @ target_R.T)
    out[:3, :3] = R_cur @ _CUBOID_ROTS[int(np.argmax(scores))]
    return out


def normalize_pose_x_in_world(pose_C, c2r, T_align):
    r2c = np.linalg.inv(c2r)
    pose_R_aligned = r2c @ pose_C @ np.linalg.inv(T_align)
    return c2r @ normalize_x_axis_roll(pose_R_aligned) @ T_align


def normalize_pose_cuboid_in_world(pose_C, c2r, T_align):
    r2c = np.linalg.inv(c2r)
    pose_R_aligned = r2c @ pose_C @ np.linalg.inv(T_align)
    return c2r @ normalize_cuboid_symmetry(pose_R_aligned) @ T_align


def normalize_pose_full_in_world(pose_C, c2r, T_align):
    r2c = np.linalg.inv(c2r)
    pose_R_aligned = r2c @ pose_C @ np.linalg.inv(T_align)
    pose_R_aligned[:3, :3] = np.eye(3)
    return c2r @ pose_R_aligned @ T_align


def normalize_object_pose(mode: str, pose_C: np.ndarray, c2r: np.ndarray,
                          T_align: np.ndarray) -> np.ndarray:
    """Dispatch on `mode`; returns the normalized 4x4 object pose (camera/world
    frame, same frame as the input pose_C)."""
    pose_C = np.asarray(pose_C, dtype=np.float64)
    c2r = np.asarray(c2r, dtype=np.float64)
    T_align = np.asarray(T_align, dtype=np.float64)
    if mode == "none":
        return pose_C.copy()
    if mode == "x_roll":
        return normalize_pose_x_in_world(pose_C, c2r, T_align)
    if mode == "axis_sign":
        return normalize_pose_cuboid_in_world(pose_C, c2r, T_align)
    if mode == "sphere":
        return normalize_pose_full_in_world(pose_C, c2r, T_align)
    raise ValueError(f"unknown normalize mode: {mode} (choices: {MODES})")
