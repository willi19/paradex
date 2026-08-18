"""Opt-in geometric retargeting for the Allegro V5 hand.

The legacy ``allegro_v5`` retargeter remains the safety fallback.  This module
adds an AnyTeleop-inspired position objective: it fits the four Allegro
fingertips to the wrist-relative MANUS fingertip geometry while retaining the
legacy command as a regularizer and warm start.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from paradex.retargetor.hand_regargetor import allegro_v5, clip_allegro_v5_safe_action


_REPO_ROOT = Path(__file__).resolve().parents[2]
_URDF_PATH = _REPO_ROOT / "rsc" / "robot" / "allegro_v5" / "allegro_right_A.urdf"
_TIP_LINKS = ("link_3_0_tip", "link_7_0_tip", "link_11_0_tip", "link_15_0_tip")
_MANUS_TIP_NAMES = ("index_tip", "middle_tip", "ring_tip", "thumb_tip")


def _transform_from_origin(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError("Allegro V5 URDF contains an invalid joint origin")
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(axis * float(angle)).as_matrix()
    return transform


@dataclass(frozen=True)
class _UrdfJoint:
    parent: str
    origin: np.ndarray
    axis: np.ndarray | None
    action_index: int | None


class AllegroV5TipKinematics:
    """Minimal URDF FK needed by the position-space retargeting objective."""

    def __init__(self, urdf_path: str | Path = _URDF_PATH):
        root = ET.parse(urdf_path).getroot()
        self._by_child: dict[str, _UrdfJoint] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            parent = joint.find("parent")
            if child is None or parent is None:
                continue
            name = joint.get("name", "")
            match = re.fullmatch(r"joint_(\d+)_0", name)
            axis_element = joint.find("axis")
            axis = None
            action_index = None
            if match is not None:
                action_index = int(match.group(1))
                axis = np.fromstring(
                    axis_element.get("xyz", "0 0 1") if axis_element is not None else "0 0 1",
                    sep=" ",
                    dtype=np.float64,
                )
                axis_norm = np.linalg.norm(axis)
                if axis.shape != (3,) or axis_norm < 1.0e-9:
                    raise ValueError(f"Allegro V5 URDF contains invalid axis for {name}")
                axis = axis / axis_norm
            self._by_child[child.get("link", "")] = _UrdfJoint(
                parent=parent.get("link", ""),
                origin=_transform_from_origin(joint.find("origin")),
                axis=axis,
                action_index=action_index,
            )

    def _link_transform(
        self, link: str, qpos: np.ndarray, cache: dict[str, np.ndarray]
    ) -> np.ndarray:
        if link == "palm_link":
            return np.eye(4, dtype=np.float64)
        if link in cache:
            return cache[link]
        joint = self._by_child.get(link)
        if joint is None:
            raise KeyError(f"No URDF joint creates Allegro V5 link {link!r}")
        transform = self._link_transform(joint.parent, qpos, cache) @ joint.origin
        if joint.action_index is not None:
            transform = transform @ _axis_angle_transform(
                joint.axis, qpos[joint.action_index]
            )
        cache[link] = transform
        return transform

    def fingertip_positions(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (16,) or not np.all(np.isfinite(qpos)):
            raise ValueError("Allegro V5 qpos must be a finite 16-vector")
        cache: dict[str, np.ndarray] = {}
        return np.vstack(
            [self._link_transform(link, qpos, cache)[:3, 3] for link in _TIP_LINKS]
        )


def manus_wrist_relative_tips(frame: Mapping[str, np.ndarray]) -> np.ndarray | None:
    """Return index/middle/ring/thumb tip positions in the MANUS wrist frame."""
    try:
        wrist_inv = np.linalg.inv(np.asarray(frame["wrist"], dtype=np.float64))
        tips = np.vstack(
            [
                (wrist_inv @ np.asarray(frame[name], dtype=np.float64))[:3, 3]
                for name in _MANUS_TIP_NAMES
            ]
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    return tips if np.all(np.isfinite(tips)) else None


def _similarity_rotation_and_scale(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Find the proper rotation and scale mapping centered source to target."""
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    source_norm_sq = float(np.sum(source_centered**2))
    if source_norm_sq < 1.0e-10:
        raise ValueError("MANUS fingertip geometry is degenerate")
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum((source_centered @ rotation) * target_centered) / source_norm_sq)
    if not np.isfinite(scale) or scale <= 1.0e-6:
        raise ValueError("MANUS-to-Allegro calibration scale is invalid")
    return rotation, scale


class AllegroV5AnyTeleopRetargeter:
    """Stateful, opt-in geometric retargeter with legacy-safe fallback.

    The first valid frame calibrates a wrist-local similarity transform against
    the legacy retargeter's command.  Later frames preserve relative fingertip
    geometry, while the legacy command keeps the redundant Allegro pose stable.
    Call ``reset`` at the start of each operator session to recalibrate.
    """

    def __init__(
        self,
        direct_retargeter: Callable[[Mapping[str, np.ndarray]], np.ndarray] = allegro_v5,
        regularization: float = 0.08,
        max_nfev: int = 32,
    ):
        if regularization < 0.0 or max_nfev < 1:
            raise ValueError("regularization must be non-negative and max_nfev positive")
        self._direct_retargeter = direct_retargeter
        self._regularization = float(regularization)
        self._max_nfev = int(max_nfev)
        self._kinematics = AllegroV5TipKinematics()
        self._rotation: np.ndarray | None = None
        self._scale: float | None = None
        self._previous_action: np.ndarray | None = None

    def reset(self) -> None:
        self._rotation = None
        self._scale = None
        self._previous_action = None

    def _calibrate_if_needed(self, human_tips: np.ndarray, direct_action: np.ndarray) -> None:
        if self._rotation is not None:
            return
        robot_tips = self._kinematics.fingertip_positions(direct_action)
        self._rotation, self._scale = _similarity_rotation_and_scale(human_tips, robot_tips)

    def __call__(self, frame: Mapping[str, np.ndarray]) -> np.ndarray:
        direct_action = clip_allegro_v5_safe_action(self._direct_retargeter(frame))
        human_tips = manus_wrist_relative_tips(frame)
        if human_tips is None:
            self._previous_action = direct_action.copy()
            return direct_action

        try:
            self._calibrate_if_needed(human_tips, direct_action)
            target = (human_tips - human_tips.mean(axis=0)) @ self._rotation
            target *= self._scale
            lower = clip_allegro_v5_safe_action(np.full(16, -np.inf))
            upper = clip_allegro_v5_safe_action(np.full(16, np.inf))
            joint_range = np.maximum(upper - lower, 1.0e-6)
            initial = np.clip(
                self._previous_action if self._previous_action is not None else direct_action,
                lower,
                upper,
            )

            def residual(qpos: np.ndarray) -> np.ndarray:
                actual = self._kinematics.fingertip_positions(qpos)
                geometry_error = (actual - actual.mean(axis=0) - target).reshape(-1)
                regularizer = np.sqrt(self._regularization) * (qpos - direct_action) / joint_range
                return np.concatenate((geometry_error / max(self._scale, 1.0e-6), regularizer))

            result = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                max_nfev=self._max_nfev,
                method="trf",
            )
            if not result.success or result.x.shape != (16,) or not np.all(np.isfinite(result.x)):
                raise RuntimeError("Allegro V5 geometric optimization did not converge")
            action = clip_allegro_v5_safe_action(result.x)
        except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
            action = direct_action

        self._previous_action = action.copy()
        return action
